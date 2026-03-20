from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# performance imports for torch: torch kernel uses one core only.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 
import torch
import torch.nn.functional as F
from tqdm.notebook import tqdm

# required files:
# use the decision labeling same as for training but online.
from data_processing.decision_labeling import DecisionLabeler
# use the inference modes implemented also in the standard case
from .inference import Mode, Beam, MCSA

@dataclass
class DecisionGuidanceConfig:
    """
    Parameters for local decision-guided reweighting at inference.
    """
    epsilon: float = 1e-3
    beta_max: float = 2.0
    alpha: float = 0.1
    support_threshold: float = 0.05

class DecisionRuleGuidedMixin:
    """
    Shared decision-guidance and reasoning helpers for all decoders.
    """
    def _init_decision_guidance(self,
                                decision_labeler: DecisionLabeler,
                                guidance_config: Optional[DecisionGuidanceConfig] = None,
                                decision_places_bundle_path: Optional[str] = None) -> None:
        
        self.decision_labeler = decision_labeler
        self.guidance_config = guidance_config or DecisionGuidanceConfig()
        self._reasoning_bundle_guards = self._load_reasoning_guards(decision_places_bundle_path)

        self._activity_name_to_id = dict(self.dataset.all_categories[0][self.concept_name_id][2])

        self._cat_feature_names = [str(cat[0]) for cat in self.dataset.all_categories[0]]
        self._num_feature_names = [str(num[0]) for num in self.dataset.all_categories[1]]

        self._static_cat_feature_names: List[str] = []
        self._static_num_feature_names: List[str] = []
        if hasattr(self.dataset, "all_static_categories") and self.dataset.all_static_categories is not None:
            if len(self.dataset.all_static_categories) >= 1:
                self._static_cat_feature_names = [str(cat[0]) for cat in self.dataset.all_static_categories[0]]
            if len(self.dataset.all_static_categories) >= 2:
                self._static_num_feature_names = [str(num[0]) for num in self.dataset.all_static_categories[1]]

        self.last_reasoning_trace: List[Dict[str, Any]] = []
        self.last_conflicts: int = 0
        self.last_decision_steps: int = 0
        self.last_explained_steps: int = 0

    @staticmethod
    def _load_reasoning_guards(decision_places_bundle_path: Optional[str]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        if not decision_places_bundle_path:
            return {}
        with open(decision_places_bundle_path, "r", encoding="utf-8") as handle:
            bundle = json.load(handle)
        out: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for entry in bundle:
            place_name = str(entry.get("place_name", ""))
            guards = entry.get("guards", {})
            if place_name:
                out[place_name] = guards
        return out

    def _decode_event_attrs_from_prefix_last(self, prefix: Tuple[List[torch.Tensor], List[torch.Tensor]]) -> Dict[str, Any]:
        cats, nums = prefix
        attrs: Dict[str, Any] = {}

        for i, tensor in enumerate(cats):
            raw_id = int(tensor[0, -1].item())
            if raw_id == 0:
                continue
            feature_name = self._cat_feature_names[i]
            id_to_label = {v: k for k, v in self.dataset.all_categories[0][i][2].items()}
            attrs[feature_name] = str(id_to_label.get(raw_id, raw_id))

        for i, tensor in enumerate(nums):
            feature_name = self._num_feature_names[i]
            attrs[feature_name] = float(tensor[0, -1].item())

        return attrs

    def _decode_event_attrs_from_predicted_ids(self,
                                               predicted_cat_ids: Dict[str, int],
                                               predicted_num_values: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        attrs: Dict[str, Any] = {}

        for feature_name, token_id in predicted_cat_ids.items():
            cat_idx = None
            for i, name in enumerate(self._cat_feature_names):
                if name == feature_name:
                    cat_idx = i
                    break
            if cat_idx is None or token_id == 0:
                continue
            id_to_label = {v: k for k, v in self.dataset.all_categories[0][cat_idx][2].items()}
            attrs[feature_name] = str(id_to_label.get(token_id, token_id))

        if predicted_num_values is not None:
            for feature_name, val in predicted_num_values.items():
                attrs[feature_name] = float(val)

        return attrs

    def _decode_event_attrs_from_suffix_step(self,
                                             suffix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                             step_idx: int,
                                             activity_id: int,
                                             static_inputs: Any,) -> Dict[str, Any]:
        """
        Build event attributes using predicted activity and GT non-activity attrs at step_idx.
        """
        suffix_cats, suffix_nums = suffix
        cat_ids: Dict[str, int] = {}
        num_vals: Dict[str, float] = {}

        for i, feature_name in enumerate(self._cat_feature_names):
            if i == self.concept_name_id:
                cat_ids[feature_name] = int(activity_id)
                continue
            if i >= len(suffix_cats) or step_idx >= suffix_cats[i].shape[1]:
                continue
            cat_ids[feature_name] = int(suffix_cats[i][0, step_idx].item())

        for i, feature_name in enumerate(self._num_feature_names):
            if i >= len(suffix_nums) or step_idx >= suffix_nums[i].shape[1]:
                continue
            num_vals[feature_name] = float(suffix_nums[i][0, step_idx].item())

        next_event_attrs = self._decode_event_attrs_from_predicted_ids(predicted_cat_ids=cat_ids,
                                                                       predicted_num_values=num_vals)
        next_event_attrs.update(self._extract_static_attrs(static_inputs))
        return self.decision_labeler._filter_attributes(next_event_attrs)

    def _roll_prefix_with_activity_from_suffix(self,
                                               prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                               suffix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                               step_idx: int,
                                               activity_id: int) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Roll prefix and append predicted activity with GT non-activity attributes at step_idx.
        """
        prefix_cats, prefix_nums = prefix
        suffix_cats, suffix_nums = suffix

        new_cats: List[torch.Tensor] = []
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

        new_nums: List[torch.Tensor] = []
        for i, num in enumerate(prefix_nums):
            shifted = torch.roll(num.clone(), shifts=-1, dims=1)
            if step_idx < suffix_nums[i].shape[1]:
                shifted[:, -1] = suffix_nums[i][:, step_idx]
            else:
                shifted[:, -1] = num[:, -1]
            new_nums.append(shifted)

        return (new_cats, new_nums)

    def _extract_static_attrs(self, static_inputs: Any) -> Dict[str, Any]:
        if static_inputs is None:
            return {}

        out: Dict[str, Any] = {}
        cat_static = None
        num_static = None

        if isinstance(static_inputs, tuple) and len(static_inputs) == 2:
            cat_static, num_static = static_inputs
        else:
            return out

        if cat_static is not None and hasattr(cat_static, "numel") and cat_static.numel() > 0:
            for i, feature_name in enumerate(self._static_cat_feature_names):
                token_id = int(cat_static[0, i].item()) if cat_static.dim() > 1 else int(cat_static[i].item())
                if token_id == 0:
                    continue
                id_to_label = {v: k for k, v in self.dataset.all_static_categories[0][i][2].items()}
                out[feature_name] = str(id_to_label.get(token_id, token_id))

        if num_static is not None and hasattr(num_static, "numel") and num_static.numel() > 0:
            for i, feature_name in enumerate(self._static_num_feature_names):
                val = float(num_static[0, i].item()) if num_static.dim() > 1 else float(num_static[i].item())
                out[feature_name] = val

        return out

    def _build_initial_past_events(self,
                                   prefix: Tuple[List[torch.Tensor], List[torch.Tensor]],
                                   static_inputs: Any) -> List[Dict[str, Any]]:
        cats, _ = prefix
        activity_tensor = cats[self.concept_name_id][0]
        static_attrs = self._extract_static_attrs(static_inputs)

        events: List[Dict[str, Any]] = []
        for t in range(activity_tensor.shape[0]):
            token = int(activity_tensor[t].item())
            if token == 0:
                continue
            attrs = self._decode_event_attrs_from_prefix_last(([cat[:, : t + 1] for cat in prefix[0]], [num[:, : t + 1] for num in prefix[1]]))
            
            attrs.update(static_attrs)
            attrs = self.decision_labeler._filter_attributes(attrs)
            events.append(attrs)

        return events

    def _get_decision_context(self,
                           current_input_activity: str,
                           past_events: List[Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, float], float]]:
        
        transitions = self.decision_labeler.transition_by_label.get(str(current_input_activity), [])
        if len(transitions) == 0:
            return None

        trans = transitions[0]
        dps = self.decision_labeler._places_after_transition(trans)
        if len(dps) == 0:
            return None

        place_name = str(dps[0])
        z_i = self.decision_labeler._predict_shallow(place_name, past_events)
        c_i = max(z_i.values(), default=0.0)
        return place_name, z_i, float(c_i)

    def _masked_distribution(self, base_probs: torch.Tensor, z_i: Dict[str, float], c_i: float, step_idx: int) -> torch.Tensor:
        if len(z_i) == 0:
            return base_probs

        cfg = self.guidance_config
        beta_r = float(c_i) * float(cfg.beta_max) * math.exp(-float(cfg.alpha) * float(step_idx))
        if beta_r <= 0.0:
            return base_probs

        z_vec = torch.zeros_like(base_probs)
        for label, prob in z_i.items():
            token_id = self._activity_name_to_id.get(str(label), None)
            if token_id is None:
                continue
            z_vec[token_id] = float(prob)

        mask = torch.pow(float(cfg.epsilon) + (1.0 - float(cfg.epsilon)) * z_vec, beta_r)
        weighted = base_probs * mask
        denom = weighted.sum()
        if float(denom.item()) <= 0.0:
            return base_probs
        return weighted / denom

    def _is_conflict(self, chosen_activity: str, z_i: Dict[str, float]) -> Tuple[bool, List[str]]:
        if len(z_i) == 0:
            return False, []
        tau = float(self.guidance_config.support_threshold)
        supported = [a for a, p in z_i.items() if float(p) >= tau]
        if len(supported) == 0:
            return False, supported
        return str(chosen_activity) not in set(map(str, supported)), supported

    @staticmethod
    def _coerce_float_or_none(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            f = float(value)
        except (TypeError, ValueError):
            return None
        if math.isnan(f):
            return None
        return f

    def _guard_matches(self, feature_row: Dict[str, Any], guard: Dict[str, Any]) -> bool:
        intervals = guard.get("intervals", {}) or {}
        allowed = guard.get("categorical_allowed", {}) or {}
        excluded = guard.get("categorical_excluded", {}) or {}

        for feat, bounds in intervals.items():
            val = self._coerce_float_or_none(feature_row.get(feat, None))
            if val is None:
                return False
            low = self._coerce_float_or_none(bounds.get("low", None))
            high = self._coerce_float_or_none(bounds.get("high", None))
            if low is not None and not (val > low):
                return False
            if high is not None and not (val <= high):
                return False

        for feat, values in allowed.items():
            val = feature_row.get(feat, None)
            if val is None:
                return False
            if str(val) not in {str(v) for v in values}:
                return False

        for feat, values in excluded.items():
            val = feature_row.get(feat, None)
            if val is None:
                continue
            if str(val) in {str(v) for v in values}:
                return False

        return True

    def _best_matching_guard(self,
                             place_name: str,
                             next_activity_label: str,
                             past_events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        
        guards = (self._reasoning_bundle_guards.get(place_name, {}).get(str(next_activity_label), []))
        
        if len(guards) == 0:
            return None

        feature_row = self.decision_labeler._build_feature_row(past_events)
        matches = [g for g in guards if self._guard_matches(feature_row, g)]
        if len(matches) == 0:
            return None

        matches.sort(key=lambda g: (float(g.get("score", 0.0)), float(g.get("prob_model", 0.0)), float(g.get("support", 0.0))),
                     reverse=True)
        
        return matches[0]

    def _build_attribute_checks(self,
                                feature_row: Dict[str, Any],
                                guard: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Build per-attribute membership checks used in local reasoning output.
        """
        
        checks: List[Dict[str, Any]] = []

        intervals = guard.get("intervals", {}) or {}
        allowed = guard.get("categorical_allowed", {}) or {}
        excluded = guard.get("categorical_excluded", {}) or {}

        for feat, bounds in intervals.items():
            val = self._coerce_float_or_none(feature_row.get(feat, None))
            low = self._coerce_float_or_none(bounds.get("low", None))
            high = self._coerce_float_or_none(bounds.get("high", None))

            in_rule_set = val is not None
            if in_rule_set and low is not None:
                in_rule_set = bool(val > low)
            if in_rule_set and high is not None:
                in_rule_set = bool(val <= high)

            checks.append({"attr": str(feat),
                           "value": None if val is None else float(val),
                           "in_rule_set": bool(in_rule_set),
                           "rule_type": "interval",
                           "low": None if low is None else float(low),
                           "high": None if high is None else float(high)})

        for feat, values in allowed.items():
            val = feature_row.get(feat, None)
            allowed_set = {str(v) for v in values}
            in_rule_set = val is not None and str(val) in allowed_set
            
            checks.append({"attr": str(feat),
                           "value": None if val is None else str(val),
                           "in_rule_set": bool(in_rule_set),
                           "rule_type": "categorical_allowed",
                           "allowed": sorted(list(allowed_set))})

        for feat, values in excluded.items():
            val = feature_row.get(feat, None)
            excluded_set = {str(v) for v in values}
            in_rule_set = val is None or str(val) not in excluded_set
            
            checks.append({"attr": str(feat),
                           "value": None if val is None else str(val),
                           "in_rule_set": bool(in_rule_set),
                           "rule_type": "categorical_excluded",
                           "excluded": sorted(list(excluded_set))})

        return checks

    def _append_reasoning_step(self,
                               step_idx: int,
                               input_activity: str,
                               selected_activity: str,
                               decision_context: Optional[Tuple[str, Dict[str, float], float]],
                               past_events: List[Dict[str, Any]],
                               model_prob: Optional[float] = None) -> None:
        
        if decision_context is None:
            return

        place_name, z_i, c_i = decision_context
        conflict, supported = self._is_conflict(selected_activity, z_i)
        sorted_decisions = sorted(z_i.items(), key=lambda kv: float(kv[1]), reverse=True)
        top_decision_event = str(sorted_decisions[0][0]) if len(sorted_decisions) > 0 else None
        top_decision_prob = float(sorted_decisions[0][1]) if len(sorted_decisions) > 0 else None

        self.last_decision_steps += 1
        if conflict:
            self.last_conflicts += 1

        matched_guard = None
        attribute_checks: List[Dict[str, Any]] = []
        explanation_status = "conflict_not_supported" if conflict else "no_matching_rule"
        explained = False

        if not conflict:
            matched_guard = self._best_matching_guard(place_name, selected_activity, past_events)
            if matched_guard is not None:
                feature_row = self.decision_labeler._build_feature_row(past_events)
                attribute_checks = self._build_attribute_checks(feature_row, matched_guard)
                self.last_explained_steps += 1
                explanation_status = "explained"
                explained = True

        self.last_reasoning_trace.append({"step": int(step_idx),
                                          "place": place_name,
                                          "input_event": str(input_activity),
                                          "next_event": str(selected_activity),
                                          "model_prob": float(model_prob) if model_prob is not None else None,
                                          "confidence": float(c_i),
                                          "decision_top_event": top_decision_event,
                                          "decision_top_prob": top_decision_prob,
                                          "supported_set": [str(a) for a in supported],
                                          "conflict": bool(conflict),
                                          "explained": bool(explained),
                                          "explanation_status": str(explanation_status),
                                          "decision_distribution": {k: float(v) for k, v in z_i.items()},
                                          "attribute_checks": attribute_checks,
                                          "matched_rule": None if matched_guard is None else {"rule": matched_guard.get("rule", ""),
                                                                                              "raw_rule": matched_guard.get("raw_rule", ""),
                                                                                              "prob_model": float(matched_guard.get("prob_model", 0.0)),
                                                                                              "support": int(matched_guard.get("support", 0)),
                                                                                              "score": float(matched_guard.get("score", 0.0))},
                                          })

    def summarize_last_reasoning(self) -> Dict[str, Any]:
        rate = 0.0
        if self.last_decision_steps > 0:
            rate = float(self.last_conflicts) / float(self.last_decision_steps)
        explained_rate = 0.0
        if self.last_decision_steps > 0:
            explained_rate = float(self.last_explained_steps) / float(self.last_decision_steps)
        
        return {"decision_steps": int(self.last_decision_steps),
                "conflicts": int(self.last_conflicts),
                "conflict_rate": float(rate),
                "explained_steps": int(self.last_explained_steps),
                "explained_rate": float(explained_rate),
                "trace": list(self.last_reasoning_trace)}

    def _reset_reasoning_state(self) -> None:
        self.last_reasoning_trace = []
        self.last_conflicts = 0
        self.last_decision_steps = 0
        self.last_explained_steps = 0


class GuidedMode(DecisionRuleGuidedMixin, Mode):
    """
    Decision-rule-guided arg-max decoding with step-wise reasoning traces.
    """

    def __init__(self,
                 model,
                 dataset,
                 decision_labeler: DecisionLabeler,
                 guidance_config: Optional[DecisionGuidanceConfig] = None,
                 decision_places_bundle_path: Optional[str] = None,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS"):
        
        super().__init__(model=model, dataset=dataset, concept_name=concept_name, eos_value=eos_value)
        self._init_decision_guidance(
            decision_labeler=decision_labeler,
            guidance_config=guidance_config,
            decision_places_bundle_path=decision_places_bundle_path,
        )

    def decode_suffix(self, prefix, suffix, prefix_len, static_inputs=None, return_reasoning=False):
        max_iteration = (self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len)

        self._reset_reasoning_state()
        current_prefix = ([t.clone() for t in prefix[0]], [t.clone() for t in prefix[1]])
        decoded: List[str] = []
        past_events = self._build_initial_past_events(current_prefix, static_inputs)

        for step_idx in range(max_iteration + 1):
            model_prefix = self._project_prefix_for_model(current_prefix)
            logits = self.model(model_prefix).squeeze(0)
            probs = F.softmax(logits, dim=-1)

            input_activity_id = int(current_prefix[0][self.concept_name_id][0, -1].item())
            input_activity = self._activity_label(input_activity_id) if input_activity_id > 0 else ""

            decision_context = None
            masked_probs = probs
            if input_activity:
                decision_context = self._get_decision_context(input_activity, past_events)
                if decision_context is not None:
                    _, z_i, c_i = decision_context
                    masked_probs = self._masked_distribution(probs, z_i, c_i, step_idx)

            activity_id = int(torch.argmax(masked_probs, dim=-1).item())
            if activity_id == self.eos_id:
                break

            selected_activity = self._activity_label(activity_id)
            decoded.append(selected_activity)
            
            self._append_reasoning_step(step_idx=step_idx,
                                        input_activity=input_activity,
                                        selected_activity=selected_activity,
                                        decision_context=decision_context,
                                        past_events=past_events,
                                        model_prob=float(masked_probs[activity_id].item()))

            current_prefix = self._roll_prefix_with_activity(prefix=current_prefix,
                                                             suffix=suffix,
                                                             step_idx=step_idx,
                                                             activity_id=activity_id)

            next_event_attrs = self._decode_event_attrs_from_prefix_last(current_prefix)
            next_event_attrs.update(self._extract_static_attrs(static_inputs))
            next_event_attrs = self.decision_labeler._filter_attributes(next_event_attrs)
            past_events.append(next_event_attrs)

        if return_reasoning:
            return decoded, self.summarize_last_reasoning()
        return decoded

    def evaluate(self, random_order=False, include_model_states=False, return_reasoning=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            import random

            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, _, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)

                if return_reasoning:
                    decoded, reasoning = self.decode_suffix(prefix=prefix,
                                                            suffix=suffix,
                                                            prefix_len=prefix_len,
                                                            static_inputs=statics,
                                                            return_reasoning=True)
                    
                    yield (case_name, prefix_len, prefix_activity, target_suffix, [decoded], reasoning)
                else:
                    decoded = self.decode_suffix(prefix=prefix, suffix=suffix, prefix_len=prefix_len, static_inputs=statics)
                    yield (case_name, prefix_len, prefix_activity, target_suffix, [decoded])


class GuidedMCSA(DecisionRuleGuidedMixin, MCSA):
    """
    Decision-rule-guided Monte-Carlo suffix sampling.
    """

    def __init__(self,
                 model,
                 dataset,
                 decision_labeler: DecisionLabeler,
                 guidance_config: Optional[DecisionGuidanceConfig] = None,
                 decision_places_bundle_path: Optional[str] = None,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS",
                 samples_per_case: int = 100,
                 sample_argmax: bool = False,
                 use_variance_cat: bool = True,
                 variational_dropout_sampling: bool = True):
        
        super().__init__(model=model,
                         dataset=dataset,
                         concept_name=concept_name,
                         eos_value=eos_value,
                         samples_per_case=samples_per_case,
                         sample_argmax=sample_argmax,
                         use_variance_cat=use_variance_cat,
                         variational_dropout_sampling=variational_dropout_sampling)
        
        self._init_decision_guidance(decision_labeler=decision_labeler,
                                     guidance_config=guidance_config,
                                     decision_places_bundle_path=decision_places_bundle_path)

    def sample_suffix(self,
                     prefix,
                     prefix_len,
                     static_inputs,
                     mask,
                     suffix=None,
                     include_model_states=False,
                     return_reasoning=False):
        
        # call model:
        prediction, (h, c), z = self.model.inference(prefix=prefix, static_inputs=static_inputs, mask=mask)

        max_iteration = (self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len)

        self._reset_reasoning_state()
        sampled_suffix: List[str] = []
        model_states = [] if include_model_states else None

        past_events = self._build_initial_past_events(prefix, static_inputs)
        current_input_activity_id = int(prefix[0][self.concept_name_id][0, -1].item())

        step_idx = 0
        while step_idx <= max_iteration:
            cat_means, cat_vars = prediction[0][0], prediction[1][0]
            activity_key = f"{self.concept_name}_mean"
            if activity_key not in cat_means:
                activity_key = [k for k in cat_means.keys() if k.endswith("_mean")][0]

            activity_logits = cat_means[activity_key]
            if self.use_variance_cat:
                var_key = f"{self.concept_name}_var"
                if var_key in cat_vars:
                    logvar = torch.clamp(cat_vars[var_key], min=-6.0, max=6.0)
                    std = torch.exp(0.5 * logvar)
                    activity_logits = torch.normal(activity_logits, std)

            base_probs = F.softmax(activity_logits, dim=-1).squeeze(0)

            input_activity = self._activity_label(current_input_activity_id) if current_input_activity_id > 0 else ""
            decision_context = None
            masked_probs = base_probs
            if input_activity:
                decision_context = self._get_decision_context(input_activity, past_events)
                if decision_context is not None:
                    _, z_i, c_i = decision_context
                    masked_probs = self._masked_distribution(base_probs, z_i, c_i, step_idx)

            if self.sample_argmax:
                sampled_activity_id = int(torch.argmax(masked_probs, dim=-1).item())
            else:
                sampled_activity_id = int(torch.multinomial(masked_probs, num_samples=1, replacement=True).item())

            if sampled_activity_id == self.eos_id:
                break

            selected_activity = self._activity_label(sampled_activity_id)
            sampled_suffix.append(selected_activity)

            self._append_reasoning_step(step_idx=step_idx,
                                        input_activity=input_activity,
                                        selected_activity=selected_activity,
                                        decision_context=decision_context,
                                        past_events=past_events,
                                        model_prob=float(masked_probs[sampled_activity_id].item()))

            cat_predictions = self._sample_categorical_predictions(cat_means, cat_vars)
            cat_predictions[activity_key] = torch.tensor([[sampled_activity_id]], device=activity_logits.device, dtype=torch.long)

            if include_model_states:
                model_states.append((h, c))

            if suffix is not None:
                next_event_attrs = self._decode_event_attrs_from_suffix_step(suffix=suffix,
                                                                             step_idx=step_idx,
                                                                             activity_id=sampled_activity_id,
                                                                             static_inputs=static_inputs)
            else:
                predicted_ids: Dict[str, int] = {}
                for key, value in cat_predictions.items():
                    feature_name = key[:-5] if key.endswith("_mean") else key
                    predicted_ids[feature_name] = int(value.item())

                next_event_attrs = self._decode_event_attrs_from_predicted_ids(predicted_ids)
                next_event_attrs.update(self._extract_static_attrs(static_inputs))
                next_event_attrs = self.decision_labeler._filter_attributes(next_event_attrs)
            past_events.append(next_event_attrs)

            next_event = (list(cat_predictions.values()), [])

            if self.variational_dropout_sampling:
                prediction, (h, c) = self.model.inference(last_event=next_event, hx=(h, c), z=z)
            else:
                prediction, (h, c) = self.model.inference(last_event=next_event, hx=(h, c), z=None)

            current_input_activity_id = sampled_activity_id
            step_idx += 1

        if return_reasoning:
            reasoning = self.summarize_last_reasoning()
            if include_model_states:
                return sampled_suffix, model_states, reasoning
            return sampled_suffix, reasoning

        if include_model_states:
            return sampled_suffix, model_states
        return sampled_suffix

    def predict_probabilistic_suffix(self, prefix, prefix_len, static_inputs, mask, suffix=None, include_model_states=False):
        suffixes = []
        for _ in range(self.samples_per_case):
            suffixes.append(
                self.sample_suffix(prefix=prefix,
                                   prefix_len=prefix_len,
                                   static_inputs=static_inputs,
                                   mask=mask,
                                   suffix=suffix,
                                   include_model_states=include_model_states))
            
        return suffixes

    def evaluate(self, random_order=False, include_model_states=False, return_reasoning=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            import random

            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, zero_mask, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)

                if not return_reasoning:
                    sampled_suffixes = self.predict_probabilistic_suffix(prefix=prefix,
                                                                         prefix_len=prefix_len,
                                                                         static_inputs=statics,
                                                                         mask=zero_mask,
                                                                         suffix=suffix,
                                                                         include_model_states=include_model_states)
                    yield (case_name, prefix_len, prefix_activity, target_suffix, sampled_suffixes)
                    continue

                sampled_suffixes: List[List[str]] = []
                reasonings: List[Dict[str, Any]] = []
                for _ in range(self.samples_per_case):
                    sampled, reasoning = self.sample_suffix(prefix=prefix,
                                                            prefix_len=prefix_len,
                                                            static_inputs=statics,
                                                            mask=zero_mask,
                                                            suffix=suffix,
                                                            include_model_states=False,
                                                            return_reasoning=True
                                                            )
                    sampled_suffixes.append(sampled)
                    reasonings.append(reasoning)

                yield (case_name, prefix_len, prefix_activity, target_suffix, sampled_suffixes, reasonings)


class GuidedBeam(DecisionRuleGuidedMixin, Beam):
    """
    Decision-rule-guided beam search with beam-local decision contexts.
    Taymouri GAN LSTM
    """
    def __init__(self,
                 model,
                 dataset,
                 decision_labeler: DecisionLabeler,
                 guidance_config: Optional[DecisionGuidanceConfig] = None,
                 decision_places_bundle_path: Optional[str] = None,
                 concept_name: str = "concept:name",
                 eos_value: str = "EOS",
                 beam_width: int = 3):
        
        super().__init__(model=model, dataset=dataset, concept_name=concept_name, eos_value=eos_value, beam_width=beam_width)
        self._init_decision_guidance(decision_labeler=decision_labeler,
                                     guidance_config=guidance_config,
                                     decision_places_bundle_path=decision_places_bundle_path)

    def _next_activity_probs_from_prefix(self, prefix) -> torch.Tensor:
        model_prefix = self._project_prefix_for_model(prefix)
        logits = self.model(model_prefix)
        if logits.dim() == 3:
            logits = logits[0, 0, :]
        elif logits.dim() == 2:
            logits = logits[0, :]
        return F.softmax(logits, dim=-1)

    def decode_suffix(self, prefix, suffix, prefix_len, static_inputs=None, return_reasoning=False):
        max_iteration = (self.dataset.encoder_decoder.window_size - self.dataset.encoder_decoder.min_suffix_size - prefix_len)

        static_attrs = self._extract_static_attrs(static_inputs)
        initial_events = self._build_initial_past_events(prefix, static_inputs)

        beams: List[Dict[str, Any]] = [{"prefix": ([t.clone() for t in prefix[0]], [t.clone() for t in prefix[1]]),
                                        "seq": [],
                                        "score": 0.0,
                                        "past_events": [dict(ev) for ev in initial_events],
                                        "reasoning": [],
                                        "conflicts": 0,
                                        "decision_steps": 0,
                                        "done": False}]

        for step_idx in range(max_iteration + 1):
            candidates: List[Dict[str, Any]] = []
            for beam in beams:
                if beam["done"]:
                    candidates.append(beam)
                    continue

                current_prefix = beam["prefix"]
                probs = self._next_activity_probs_from_prefix(current_prefix)

                input_activity_id = int(current_prefix[0][self.concept_name_id][0, -1].item())
                input_activity = self._activity_label(input_activity_id) if input_activity_id > 0 else ""

                decision_context = None
                masked_probs = probs
                if input_activity:
                    decision_context = self._get_decision_context(input_activity, beam["past_events"])
                    if decision_context is not None:
                        _, z_i, c_i = decision_context
                        masked_probs = self._masked_distribution(probs, z_i, c_i, step_idx)

                topk_logp, topk_idx = torch.topk(torch.log(masked_probs + 1e-12), k=min(self.beam_width, masked_probs.shape[-1]))
                for j in range(topk_idx.shape[0]):
                    tok = int(topk_idx[j].item())
                    tok_logp = float(topk_logp[j].item())

                    new_seq = beam["seq"] + [tok]
                    done = tok == self.eos_id

                    new_prefix = beam["prefix"]
                    new_past_events = [dict(ev) for ev in beam["past_events"]]
                    new_reasoning = [dict(r) for r in beam["reasoning"]]
                    new_conflicts = int(beam["conflicts"])
                    new_decision_steps = int(beam["decision_steps"])

                    if not done:
                        selected_activity = self._activity_label(tok)
                        if decision_context is not None:
                            place_name, z_i, c_i = decision_context
                            conflict, supported = self._is_conflict(selected_activity, z_i)
                            sorted_decisions = sorted(z_i.items(), key=lambda kv: float(kv[1]), reverse=True)
                            top_decision_event = str(sorted_decisions[0][0]) if len(sorted_decisions) > 0 else None
                            top_decision_prob = float(sorted_decisions[0][1]) if len(sorted_decisions) > 0 else None
                            new_decision_steps += 1
                            if conflict:
                                new_conflicts += 1

                            guard = None
                            attribute_checks: List[Dict[str, Any]] = []
                            explanation_status = "conflict_not_supported" if conflict else "no_matching_rule"
                            explained = False
                            if not conflict:
                                guard = self._best_matching_guard(place_name, selected_activity, new_past_events)
                                if guard is not None:
                                    feature_row = self.decision_labeler._build_feature_row(new_past_events)
                                    attribute_checks = self._build_attribute_checks(feature_row, guard)
                                    explained = True
                                    explanation_status = "explained"

                            new_reasoning.append({"step": int(step_idx),
                                                  "place": place_name,
                                                  "input_event": str(input_activity),
                                                  "next_event": str(selected_activity),
                                                  "confidence": float(c_i),
                                                  "decision_top_event": top_decision_event,
                                                  "decision_top_prob": top_decision_prob,
                                                  "supported_set": [str(a) for a in supported],
                                                  "conflict": bool(conflict),
                                                  "explained": bool(explained),
                                                  "explanation_status": str(explanation_status),
                                                  "decision_distribution": {k: float(v) for k, v in z_i.items()},
                                                  "attribute_checks": attribute_checks,
                                                  "matched_rule": None if guard is None else {"rule": guard.get("rule", ""),
                                                                                              "raw_rule": guard.get("raw_rule", ""),
                                                                                              "prob_model": float(guard.get("prob_model", 0.0)),
                                                                                              "support": int(guard.get("support", 0)),
                                                                                              "score": float(guard.get("score", 0.0))},
                                                  })

                        new_prefix = self._roll_prefix_with_activity_from_suffix(
                            prefix=beam["prefix"],
                            suffix=suffix,
                            step_idx=step_idx,
                            activity_id=tok,
                        )
                        next_attrs = self._decode_event_attrs_from_prefix_last(new_prefix)
                        next_attrs.update(static_attrs)
                        next_attrs = self.decision_labeler._filter_attributes(next_attrs)
                        new_past_events.append(next_attrs)

                    candidates.append({"prefix": new_prefix,
                                       "seq": new_seq,
                                       "score": float(beam["score"] + tok_logp),
                                       "past_events": new_past_events,
                                       "reasoning": new_reasoning,
                                       "conflicts": new_conflicts,
                                       "decision_steps": new_decision_steps,
                                       "done": done
                                       })

            candidates.sort(key=lambda b: b["score"], reverse=True)
            beams = candidates[: self.beam_width]
            if all(b["done"] for b in beams):
                break

        decoded_beams: List[List[str]] = []
        reasoning_beams: List[Dict[str, Any]] = []

        for b in beams:
            decoded = []
            for token in b["seq"]:
                if token == 0 or token == self.eos_id:
                    break
                decoded.append(self._activity_label(int(token)))
            decoded_beams.append(decoded)

            rate = 0.0
            if b["decision_steps"] > 0:
                rate = float(b["conflicts"]) / float(b["decision_steps"])
            explained_steps = sum(1 for r in b["reasoning"] if bool(r.get("explained", False)))
            explained_rate = 0.0
            if b["decision_steps"] > 0:
                explained_rate = float(explained_steps) / float(b["decision_steps"])
            
            reasoning_beams.append({"decision_steps": int(b["decision_steps"]),
                                    "conflicts": int(b["conflicts"]),
                                    "conflict_rate": float(rate),
                                    "explained_steps": int(explained_steps),
                                    "explained_rate": float(explained_rate),
                                    "trace": b["reasoning"],
                                    "beam_logprob": float(b["score"])
                                    })

        if return_reasoning:
            return decoded_beams, reasoning_beams
        return decoded_beams

    def evaluate(self, random_order=False, include_model_states=False, return_reasoning=False):
        self._ensure_eval_mode()
        case_items = list(self.cases.items())
        if random_order:
            import random

            case_items = random.sample(case_items, len(case_items))

        for _, (case_name, full_case) in tqdm(enumerate(case_items), total=len(self.cases)):
            for _, (prefix_len, prefix, _, statics, suffix) in enumerate(self._iterate_case(full_case)):
                prefix_activity = self._decode_activity_prefix(prefix)
                target_suffix = self._decode_activity_suffix(suffix)
                if return_reasoning:
                    
                    decoded_suffixes, reasonings = self.decode_suffix(prefix=prefix,
                                                                      suffix=suffix,
                                                                      prefix_len=prefix_len,
                                                                      static_inputs=statics,
                                                                      return_reasoning=True)
                    
                    yield (case_name, prefix_len, prefix_activity, target_suffix, decoded_suffixes, reasonings)
                else:
                    decoded_suffixes = self.decode_suffix(prefix=prefix, suffix=suffix, prefix_len=prefix_len, static_inputs=statics)
                    yield (case_name, prefix_len, prefix_activity, target_suffix, decoded_suffixes)


# call the right mode method:
def get_decision_guided_evaluator(kind: str,
                                  model,
                                  dataset,
                                  decision_labeler: DecisionLabeler,
                                  guidance_config: Optional[DecisionGuidanceConfig] = None,
                                  decision_places_bundle_path: Optional[str] = None,
                                  **kwargs):
    """
    Factory for decision-rule-guided evaluators.
    Supported kinds:
    - "mcsa": guided Monte-Carlo suffix sampling
    - "mode": guided arg-max decoding
    - "beam": guided beam-search decoding
    """
    kind_normalized = kind.strip().lower()

    common_kwargs = {"model": model,
                     "dataset": dataset,
                     "decision_labeler": decision_labeler,
                     "guidance_config": guidance_config,
                     "decision_places_bundle_path": decision_places_bundle_path}
    common_kwargs.update(kwargs)

    if kind_normalized == "mcsa":
        return GuidedMCSA(**common_kwargs)
    if kind_normalized == "mode":
        return GuidedMode(**common_kwargs)
    if kind_normalized == "beam":
        return GuidedBeam(**common_kwargs)

    raise ValueError(f"Unknown guided evaluator kind '{kind}'. Supported: 'mcsa', 'mode', 'beam'.")

