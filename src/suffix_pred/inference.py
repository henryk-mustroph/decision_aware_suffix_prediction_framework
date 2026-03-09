import os
import random
from collections.abc import Iterator

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

class Decoder:
    """
    Shared base class for activity-sequence decoders/evaluators.
    """

    def __init__(self,
                 model,
                 dataset,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS"):
        
        self.model = model
        self.dataset = dataset
        self.concept_name = concept_name

        # Index of activity in categorical dataset attributes
        self.concept_name_id = [i for i, cat in enumerate(self.dataset.all_categories[0]) if cat[0] == self.concept_name][0]

        # EOS id for activity
        self.eos_id = [v for k, v in self.dataset.all_categories[0][self.concept_name_id][2].items() if k == eos_value][0]

        self._activity_id_to_label = {
            v: k for k, v in self.dataset.all_categories[0][self.concept_name_id][2].items()
        }

        self.cases = self._get_cases_from_dataset()

    def _get_cases_from_dataset(self):
        cases = {}
        for padded_case in self.dataset:
            case_id = padded_case[0]
            categorical_tensors = padded_case[1]
            suffix = categorical_tensors[self.concept_name_id][-self.dataset.min_suffix_size :]
            if torch.all(suffix == self.eos_id).item():
                cases[case_id] = padded_case
        return cases

    def _prepare_static_inputs(self, cats_static, nums_static):
        static_cat = None
        static_num = None

        if cats_static is not None and cats_static.numel() > 0:
            static_cat = cats_static
        if nums_static is not None and nums_static.numel() > 0:
            static_num = nums_static

        if static_cat is None and static_num is None:
            return None

        return (static_cat, static_num)

    def _iterate_case(self, case) -> Iterator[tuple]:
        (_, categorical_tensors, numerical_tensors, _, zero_mask, static_cats, static_nums, *_,) = case

        current_prefix = ([torch.zeros_like(cat_attribute).unsqueeze(0) for cat_attribute in categorical_tensors],
                          [torch.zeros_like(num_attribute).unsqueeze(0) for num_attribute in numerical_tensors])

        current_suffix = ([torch.clone(cat_attribute).unsqueeze(0) for cat_attribute in categorical_tensors],
                          [torch.clone(num_attribute).unsqueeze(0) for num_attribute in numerical_tensors])

        zero_mask_default = torch.zeros_like(zero_mask).unsqueeze(0).unsqueeze(0)
        static_atts = self._prepare_static_inputs(static_cats, static_nums)

        prefix_length = 0
        for i in range(
            categorical_tensors[self.concept_name_id].shape[0] - self.dataset.min_suffix_size - 1):
            for j in range(len(current_prefix[0])):
                current_prefix[0][j][0] = torch.roll(current_prefix[0][j][0], -1)
                current_prefix[0][j][0, -1] = categorical_tensors[j][i]

                current_suffix[0][j][0] = torch.roll(current_suffix[0][j][0], -1)
                current_suffix[0][j][0, -1] = 0

            for j in range(len(current_prefix[1])):
                current_prefix[1][j][0] = torch.roll(current_prefix[1][j][0], -1)
                current_prefix[1][j][0, -1] = numerical_tensors[j][i]

                current_suffix[1][j][0] = torch.roll(current_suffix[1][j][0], -1)
                current_suffix[1][j][0, -1] = 0

            zero_mask_default[0, 0] = torch.roll(zero_mask_default[0, 0], -1)
            zero_mask_default[0, 0, -1] = zero_mask[i]

            if prefix_length or categorical_tensors[self.concept_name_id][i]:
                prefix_length += 1
                current_mask = zero_mask_default[0].clone()
                yield (prefix_length,
                       current_prefix,
                       current_mask,
                       static_atts,
                       current_suffix)

    def _sample_categorical_predictions(self, cat_means, cat_variances):
        sampled_predictions = {}

        for key in cat_means.keys():
            if not key.endswith("_mean"):
                continue

            feature_name = key[:-5]
            logits = cat_means[key]

            if self.use_variance_cat:
                var_key = f"{feature_name}_var"
                if var_key in cat_variances:
                    logvar = torch.clamp(cat_variances[var_key], min=-6.0, max=6.0)
                    std = torch.exp(0.5 * logvar)
                    logits = torch.normal(logits, std)

            if self.sample_argmax:
                sampled = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                sampled = torch.multinomial(probs, num_samples=1, replacement=True)

            sampled_predictions[f"{feature_name}_mean"] = sampled

        return sampled_predictions

    def _activity_label(self, activity_id: int):
        return self._activity_id_to_label.get(activity_id, None)

    def _decode_activity_suffix(self, suffix_case):
        activity_tensor = suffix_case[0][self.concept_name_id][0]
        result = []
        for i in range(activity_tensor.shape[0]):
            token = int(activity_tensor[i].item())
            if token == 0 or token == self.eos_id:
                continue
            result.append(self._activity_label(token))
        return result

    def _decode_activity_suffix_ids(self, suffix_case):
        activity_tensor = suffix_case[0][self.concept_name_id][0]
        result = []
        for i in range(activity_tensor.shape[0]):
            token = int(activity_tensor[i].item())
            if token == 0 or token == self.eos_id:
                continue
            result.append(token)
        return result

    def _decode_activity_prefix(self, prefix_case):
        activity_tensor = prefix_case[0][self.concept_name_id][0]
        seq_labels = []
        for i in range(activity_tensor.shape[0]):
            token = int(activity_tensor[i].item())
            if token == 0:
                continue
            seq_labels.append(self._activity_label(token))
        return seq_labels

    def _ensure_eval_mode(self):
        if hasattr(self.model, "eval") and callable(self.model.eval):
            self.model.eval()

    def _require_model_method(self, method_name: str):
        method = getattr(self.model, method_name, None)
        if method is None or not callable(method):
            raise TypeError( f"Model used with {self.__class__.__name__} must implement callable '{method_name}(...)'.")

# Monte Carlo Suffix sampling
class MCSA(Decoder):
    """
    Adopted Monte-Carlo Suffix Sampling Algorithm: sampling suffixes for evaluation.

    - Evaluates only activity sequence suffixes.
    - Uses probabilistic categorical sampling (with optional log-variance noise).
    """

    def __init__(
        self,
        model,
        dataset,
        concept_name: str = "concept:name",
        eos_value: str = "EOS",
        samples_per_case: int = 100,
        sample_argmax: bool = False,
        use_variance_cat: bool = True,
        variational_dropout_sampling: bool = True,
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            concept_name=concept_name,
            eos_value=eos_value,
        )
        self.samples_per_case = samples_per_case
        self.sample_argmax = sample_argmax
        self.use_variance_cat = use_variance_cat
        self.variational_dropout_sampling = variational_dropout_sampling
        self._require_model_method("inference")

    def sample_suffix(self, prefix, prefix_len, static_inputs, mask, include_model_states=False):
        prediction, (h, c), z = self.model.inference(
            prefix=prefix,
            static_inputs=static_inputs,
            mask=mask,
        )

        max_iteration = (
            self.dataset.encoder_decoder.window_size
            - self.dataset.encoder_decoder.min_suffix_size
            - prefix_len
        )

        sampled_suffix = []
        model_states = [] if include_model_states else None

        i = 0
        while i <= max_iteration:
            cat_predictions = self._sample_categorical_predictions(prediction[0][0], prediction[1][0])

            activity_key = f"{self.concept_name}_mean"
            if activity_key not in cat_predictions:
                # fallback if naming differs
                activity_key = [k for k in cat_predictions.keys() if k.endswith("_mean")][0]

            activity_id = int(cat_predictions[activity_key].item())
            if activity_id == self.eos_id:
                break

            sampled_suffix.append(self._activity_label(activity_id))

            if include_model_states:
                model_states.append((h, c))

            # Keep numeric inputs empty: model handles fallback if required.
            next_event = (list(cat_predictions.values()), [])

            if self.variational_dropout_sampling:
                prediction, (h, c) = self.model.inference(last_event=next_event,
                                                          hx=(h, c),
                                                          z=z)
                
            else:
                prediction, (h, c) = self.model.inference(last_event=next_event,
                                                          hx=(h, c),
                                                          z=None)

            i += 1

        if include_model_states:
            return sampled_suffix, model_states
        return sampled_suffix

    def predict_probabilistic_suffix(self, prefix, prefix_len, static_inputs, mask, include_model_states=False):
        suffixes = []
        for _ in range(self.samples_per_case):
            suffixes.append(
                self.sample_suffix(
                    prefix=prefix,
                    prefix_len=prefix_len,
                    static_inputs=static_inputs,
                    mask=mask,
                    include_model_states=include_model_states)
            )
        return suffixes

    def evaluate(self, random_order=False, include_model_states=False):
        """
        Sequential activity-only probabilistic evaluation.

        Yields:
        - case_id,
        - prefix_len,
        - prefix,            # activity label sequence
        - target_suffix,     # activity label sequence
        - sampled_suffixes,  # list of sampled activity sequences
        """
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, zero_mask, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                suffix_activity_sequence = self._decode_activity_suffix(suffix)
                
                sampled_suffixes = self.predict_probabilistic_suffix(prefix=prefix,
                                                                     prefix_len=prefix_len,
                                                                     static_inputs=statics,
                                                                     mask=zero_mask,
                                                                     include_model_states=include_model_states)
                yield (case_name,
                       prefix_len,
                       prefix_activity,
                       suffix_activity_sequence,
                       sampled_suffixes)


# Arg-max activity sampling for camargo:
class Mode(Decoder):
    """
    Deterministic arg-max activity suffix decoding (Camargo-style inference).
    """

    def __init__(
        self,
        model,
        dataset,
        concept_name: str = "concept:name",
        eos_value: str = "EOS",
    ):
        super().__init__(
            model=model,
            dataset=dataset,
            concept_name=concept_name,
            eos_value=eos_value,
        )
        if not callable(self.model):
            raise TypeError(
                "Model used with Mode must be callable and return activity probabilities."
            )

        # Optional feature projection for models (e.g. C-LSTM) that consume a subset
        # of dataset dynamic features defined in model.model_feat.
        self._cat_feature_indices = None
        self._num_feature_indices = None
        self._init_model_feature_projection()

    def _init_model_feature_projection(self):
        model_feat = getattr(self.model, "model_feat", None)
        if model_feat is None:
            return

        dataset_cat_names = [cat[0] for cat in self.dataset.all_categories[0]]
        dataset_num_names = [num[0] for num in self.dataset.all_categories[1]]
        model_cat_names, model_num_names = model_feat

        missing_cat = [name for name in model_cat_names if name not in dataset_cat_names]
        missing_num = [name for name in model_num_names if name not in dataset_num_names]
        if missing_cat or missing_num:
            raise ValueError(
                "Model features are missing in dataset categories for Mode decoding. "
                f"Missing categorical: {missing_cat}, missing numerical: {missing_num}."
            )

        self._cat_feature_indices = [dataset_cat_names.index(name) for name in model_cat_names]
        self._num_feature_indices = [dataset_num_names.index(name) for name in model_num_names]

    def _project_prefix_for_model(self, prefix):
        if self._cat_feature_indices is None and self._num_feature_indices is None:
            return prefix

        prefix_cats, prefix_nums = prefix
        selected_cats = [prefix_cats[i] for i in self._cat_feature_indices]
        selected_nums = [prefix_nums[i] for i in self._num_feature_indices]
        return (selected_cats, selected_nums)

    def _roll_prefix_with_activity(self, prefix, suffix, step_idx: int, activity_id: int):
        prefix_cats, prefix_nums = prefix
        suffix_cats, suffix_nums = suffix

        new_cats = []
        for i, cat in enumerate(prefix_cats):
            shifted = torch.roll(cat.clone(), shifts=-1, dims=1)
            if i == self.concept_name_id:
                shifted[:, -1] = activity_id
            else:
                if step_idx < suffix_cats[i].shape[1]:
                    shifted[:, -1] = suffix_cats[i][:, step_idx]
                else:
                    shifted[:, -1] = 0
            new_cats.append(shifted)

        new_nums = []
        for i, num in enumerate(prefix_nums):
            shifted = torch.roll(num.clone(), shifts=-1, dims=1)
            if step_idx < suffix_nums[i].shape[1]:
                shifted[:, -1] = suffix_nums[i][:, step_idx]
            else:
                shifted[:, -1] = num[:, -1]
            new_nums.append(shifted)

        return (new_cats, new_nums)

    def decode_suffix(self, prefix, suffix, prefix_len):
        max_iteration = (
            self.dataset.encoder_decoder.window_size
            - self.dataset.encoder_decoder.min_suffix_size
            - prefix_len
        )

        current_prefix = ([t.clone() for t in prefix[0]], [t.clone() for t in prefix[1]])
        decoded = []

        for step_idx in range(max_iteration + 1):
            model_prefix = self._project_prefix_for_model(current_prefix)
            probs = self.model(model_prefix)
            activity_id = int(torch.argmax(probs, dim=-1).item())

            if activity_id == self.eos_id:
                break

            decoded.append(self._activity_label(activity_id))
            current_prefix = self._roll_prefix_with_activity(
                prefix=current_prefix,
                suffix=suffix,
                step_idx=step_idx,
                activity_id=activity_id,
            )

        return decoded

    def evaluate(self, random_order=False, include_model_states=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, _, _, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)
                decoded_suffixes = [self.decode_suffix(prefix=prefix, suffix=suffix, prefix_len=prefix_len)]
                yield (
                    case_name,
                    prefix_len,
                    prefix_activity,
                    target_suffix,
                    decoded_suffixes,
                )


# beam search for activity sequences of Taymouri et. al.
class Beam(Decoder):
    """
    Fixed-width beam-search activity suffix decoding (Taymouri-style inference).
    """

    def __init__(
        self,
        model,
        dataset,
        concept_name: str = "concept:name",
        eos_value: str = "EOS",
        beam_width: int = 3):
        
        super().__init__(
            model=model,
            dataset=dataset,
            concept_name=concept_name,
            eos_value=eos_value,
        )
        self.beam_width = beam_width
        self._require_model_method("beam_search")

        # Feature projection (same pattern as Mode decoder)
        self._cat_feature_indices = None
        self._num_feature_indices = None
        self._init_model_feature_projection()

    def _init_model_feature_projection(self):
        model_feat = getattr(self.model, "model_feat", None)
        if model_feat is None:
            return

        dataset_cat_names = [cat[0] for cat in self.dataset.all_categories[0]]
        dataset_num_names = [num[0] for num in self.dataset.all_categories[1]]
        model_cat_names, model_num_names = model_feat

        missing_cat = [name for name in model_cat_names if name not in dataset_cat_names]
        missing_num = [name for name in model_num_names if name not in dataset_num_names]
        if missing_cat or missing_num:
            raise ValueError(
                "Model features are missing in dataset categories for Beam decoding. "
                f"Missing categorical: {missing_cat}, missing numerical: {missing_num}."
            )

        self._cat_feature_indices = [dataset_cat_names.index(name) for name in model_cat_names]
        self._num_feature_indices = [dataset_num_names.index(name) for name in model_num_names]

    def _project_prefix_for_model(self, prefix):
        if self._cat_feature_indices is None and self._num_feature_indices is None:
            return prefix

        prefix_cats, prefix_nums = prefix
        selected_cats = [prefix_cats[i] for i in self._cat_feature_indices]
        selected_nums = [prefix_nums[i] for i in self._num_feature_indices]
        return (selected_cats, selected_nums)

    def _decode_beam_ids(self, beam_ids_1d):
        """Convert a single beam's token id list to activity labels, stopping at EOS/pad."""
        decoded = []
        for token in beam_ids_1d:
            token = int(token)
            if token == 0 or token == self.eos_id:
                break
            decoded.append(self._activity_label(token))
        return decoded

    def decode_suffix(self, prefix, prefix_len):
        """Return all beam candidates as a list of decoded activity-label sequences."""
        max_iteration = (
            self.dataset.encoder_decoder.window_size
            - self.dataset.encoder_decoder.min_suffix_size
            - prefix_len
        )

        model_prefix = self._project_prefix_for_model(prefix)
        beam_ids = self.model.beam_search(
            prefixes=model_prefix,
            beam_width=self.beam_width,
            max_len=max_iteration + 1,
            eos_id=self.eos_id,
        )

        # beam_ids shape: [1, beam_width, max_len] (batch=1 per call)
        all_beams = beam_ids[0]  # [beam_width, max_len]
        decoded_beams = []
        for k in range(all_beams.shape[0]):
            decoded_beams.append(self._decode_beam_ids(all_beams[k].tolist()))
        return decoded_beams

    def evaluate(self, random_order=False, include_model_states=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, _, _, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)
                decoded_suffixes = self.decode_suffix(prefix=prefix, prefix_len=prefix_len)
                yield (case_name,
                       prefix_len,
                       prefix_activity,
                       target_suffix,
                       decoded_suffixes)


def get_evaluator(kind: str, model, dataset, **kwargs):
    """
    Factory helper to build suffix evaluators.

    Supported kinds:
    - "mcsa": probabilistic Monte-Carlo sampling
    - "mode": deterministic arg-max decoding
    - "beam": fixed-width beam-search decoding
    """
    kind_normalized = kind.strip().lower()

    if kind_normalized == "mcsa":
        return MCSA(model=model, dataset=dataset, **kwargs)
    if kind_normalized == "mode":
        return Mode(model=model, dataset=dataset, **kwargs)
    if kind_normalized == "beam":
        return Beam(model=model, dataset=dataset, **kwargs)

    raise ValueError(f"Unknown evaluator kind '{kind}'. Supported: 'mcsa', 'mode', 'beam'.")