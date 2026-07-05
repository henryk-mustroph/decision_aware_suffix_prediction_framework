"""
Trainers for the three suffix-prediction architectures:
- UEDTrainer: dropout-uncertainty encoder-decoder LSTM (U-ED-LSTM)
- CTraining:  Camargo-style next-event LSTM (FS-LSTM)
- TTraining:  Taymouri-style GAN encoder-decoder LSTM (GAN-LSTM)

Each trainer supports clean and decision-aware training. Decision-aware
training adds a semantic set-membership loss (loss.Loss.semantic_loss)
weighted by lambda_sem.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm

try:
    from .loss import Loss
except ImportError:
    from loss import Loss


class Trainer:
    """
    General base trainer for shared training setup and utilities.
    """
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 optimize_values,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = "model.pkl"):

        self.device = device
        self.model = model.to(device)
        self.data_train = data_train
        self.data_val = data_val

        self.optimize_values = optimize_values
        self.optimizer = optimize_values.get("optimizer", None)
        self.scheduler = optimize_values.get("scheduler", None)
        self.epochs = optimize_values.get("epochs", 1)
        self.mini_batches = optimize_values.get("mini_batches", 1)
        self.shuffle = optimize_values.get("shuffle", True)
        # Semantic-loss support threshold tau in (0, 1].
        # Default 0.5 ("majority support"); set per-notebook via optimize_values["tau"].
        self.tau = float(optimize_values.get("tau", 0.5))

        # Decision-label denoising for the semantic loss: only constrain steps
        # whose ground-truth next activity is inside the decision model's
        # tau-support (i.e. the soft constraint agrees with the observed
        # outcome).
        self.sem_gate_gt_in_support = bool(optimize_values.get("sem_gate_gt_in_support", True))

        # Teacher forcing policy shared by autoregressive trainers.
        self.teacher_forcing_mode = str(optimize_values.get("teacher_forcing_mode", "scheduled")).lower()
        if self.teacher_forcing_mode not in {"scheduled", "fixed"}:
            raise ValueError("teacher_forcing_mode must be either 'scheduled' or 'fixed'")

        fixed_ratio = float(optimize_values.get("fixed_teacher_forcing_ratio", 1.0))
        self.fixed_teacher_forcing_ratio = max(0.0, min(1.0, fixed_ratio))

        self.save_model_n_th_epoch = save_model_n_th_epoch
        self.saving_path = saving_path

    def _build_dataloader(self, dataset, num_workers=8):
        return DataLoader(dataset=dataset,
                          batch_size=self.mini_batches,
                          shuffle=self.shuffle,
                          num_workers=num_workers,
                          pin_memory=True)

    def _save_model(self):
        self.model.save(self.saving_path)

    def _current_lr(self):
        if self.scheduler is None:
            return None
        return self.scheduler.optimizer.param_groups[0]["lr"]

    def _step_scheduler(self, metric):
        if self.scheduler is not None:
            self.scheduler.step(metric)

    def _save_if_due(self, epoch_index: int):
        if self.save_model_n_th_epoch > 0 and (epoch_index + 1) % self.save_model_n_th_epoch == 0:
            tqdm.write("saving model")
            self._save_model()

    def _unpack_batch_common(self, batch):
        if len(batch) == 8:
            _, cats, nums, eos_paddings, zero_paddings, cats_static, nums_static, decision_data = batch
        else:
            raise ValueError(
                f"Unsupported batch format with len={len(batch)}. " "Expected full 8-tuple: (_, cats, nums, eos, zero, cats_static, nums_static, decision_data).")

        # decision_data is a tuple (z_targets, z_mask)
        z_targets, z_mask = decision_data

        return {"cats": cats,
                "nums": nums,
                "eos_paddings": eos_paddings,
                "zero_paddings": zero_paddings,
                "cats_static": cats_static,
                "nums_static": nums_static,
                "z_targets": z_targets,
                "z_mask": z_mask}

    def _prepare_static_inputs(self, cats_static, nums_static):
        static_cat = None
        static_num = None

        if cats_static is not None and hasattr(cats_static, "numel") and cats_static.numel() > 0:
            static_cat = cats_static.to(self.device)
        if nums_static is not None and hasattr(nums_static, "numel") and nums_static.numel() > 0:
            static_num = nums_static.to(self.device)

        if static_cat is None and static_num is None:
            return None

        return (static_cat, static_num)

    # Used by U-ED-LSTM and T-GAN-LSTM (fixed-length suffix split).
    def _split_prefix_suffix(self, cats, nums, suffix_size):
        prefixes_cat = [cat[:, :-suffix_size].to(self.device) for cat in cats]
        prefixes_num = [num[:, :-suffix_size].to(self.device) for num in nums]
        suffixes_cat = [cat[:, -suffix_size:].to(self.device) for cat in cats]
        suffixes_num = [num[:, -suffix_size:].to(self.device) for num in nums]
        return [prefixes_cat, prefixes_num], [suffixes_cat, suffixes_num]

    def _build_masks(self, eos_paddings, zero_paddings, suffix_size, use_zero_padd_masking, use_eos_padd_masking):
        prefix_mask = None
        eos_paddings_suffix = None

        if use_zero_padd_masking and zero_paddings is not None:
            prefix_mask = zero_paddings[:, :-suffix_size].to(self.device)

        if use_eos_padd_masking and eos_paddings is not None:
            eos_paddings_suffix = eos_paddings[:, -suffix_size:].to(self.device)

        if use_zero_padd_masking and zero_paddings is not None:
            suffix_zero_mask = zero_paddings[:, -suffix_size:].to(self.device)
            if eos_paddings_suffix is not None:
                eos_paddings_suffix = eos_paddings_suffix * suffix_zero_mask
            else:
                eos_paddings_suffix = suffix_zero_mask

        return prefix_mask, eos_paddings_suffix

    # teacher forcing and epsilon sampling:
    def _scheduled_sampling_rates(self, step_index, epsilon_max, inverse_sigmoid_k, min_teacher_forcing=0.0):
        """
        Inverse-sigmoid scheduled sampling.

        Returns:
        - epsilon: probability of feeding model prediction.
        - teacher_forcing_ratio: probability of feeding ground truth.
        """
        k = max(1e-6, float(inverse_sigmoid_k))
        t = float(step_index)

        # Rising inverse-sigmoid from ~0 to ~1 with exact 0 at t=0.
        raw = 1.0 - (k / (k + math.exp(t / k)))
        raw0 = 1.0 - (k / (k + 1.0))
        norm = (raw - raw0) / max(1e-8, (1.0 - raw0))

        epsilon = float(epsilon_max) * max(0.0, min(1.0, norm))
        teacher_forcing_ratio = max(float(min_teacher_forcing), 1.0 - epsilon)
        teacher_forcing_ratio = min(1.0, teacher_forcing_ratio)
        epsilon = 1.0 - teacher_forcing_ratio
        return epsilon, teacher_forcing_ratio

    def _teacher_forcing_rates(self,
                               step_index,
                               *,
                               epsilon_max,
                               inverse_sigmoid_k,
                               min_teacher_forcing=0.0):
        """
        Resolve teacher forcing for this epoch according to configured mode.

        Modes:
        - scheduled: inverse-sigmoid schedule (existing behavior)
        - fixed: constant teacher forcing ratio across all epochs
        """
        if self.teacher_forcing_mode == "fixed":
            teacher_forcing_ratio = self.fixed_teacher_forcing_ratio
            epsilon = 1.0 - teacher_forcing_ratio
            return epsilon, teacher_forcing_ratio

        return self._scheduled_sampling_rates(step_index=step_index,
                                              epsilon_max=epsilon_max,
                                              inverse_sigmoid_k=inverse_sigmoid_k,
                                              min_teacher_forcing=min_teacher_forcing)

    def _extract_guard_suffix(self, z_targets, z_mask, suffix_size):
        """
        Per-event decision labels that are stored for the whole window and cut out exactly the subset that lines up with decoder steps during autoregressive suffix training
        The guard label for step s is therefore the label of event at position T-S-1+s.

        Outputs:
        - z_suffix_targets: [B, S, C] or None
        - z_suffix_mask: [B, S] or None
        """
        if z_targets.shape[-1] == 0:
            return None, None
        S = suffix_size

        z_suffix_targets = z_targets[:, -(S + 1):-1, :].to(self.device)
        z_suffix_mask = z_mask[:, -(S + 1):-1].to(self.device)
        return z_suffix_targets, z_suffix_mask


#
# Trainer for the U-ED-LSTM (with and without decision-awareness).
class UEDTrainer(Trainer):
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 loss_obj,
                 optimize_values,
                 suffix_data_split_value,
                 lambda_sem: float = 0.0,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'U_ED_LSTM_train.pkl'):

        super().__init__(device=device,
                         model=model,
                         data_train=data_train,
                         data_val=data_val,
                         optimize_values=optimize_values,
                         save_model_n_th_epoch=save_model_n_th_epoch,
                         saving_path=saving_path)

        self.loss_obj = loss_obj
        self.suffix_data_split_value = suffix_data_split_value
        self.lambda_sem = lambda_sem

        self.regularization_term = optimize_values["regularization_term"]

        # Teacher forcing (scheduled-sampling parameters override the base policy).
        self.min_teacher_forcing_value = optimize_values["min_teacher_forcing_value"]
        self.max_teacher_forcing_value = optimize_values["max_teacher_forcing_value"]
        self.scheduled_sampling_epsilon_max = optimize_values.get("scheduled_sampling_epsilon_max", self.max_teacher_forcing_value)
        self.scheduled_sampling_k = optimize_values.get("scheduled_sampling_k", max(1.0, self.epochs / 10.0))

        # Auxiliary loss weights for non-activity dynamic attribute heads.
        # Activity head dominates; auxiliary heads enable autoregressive
        # decoding of resources/timestamps without GT leakage.
        self.aux_cat_loss_weight = float(optimize_values.get("aux_cat_loss_weight", 0.25))
        self.aux_num_loss_weight = float(optimize_values.get("aux_num_loss_weight", 0.25))
        self.aux_cat_attenuation_samples = int(optimize_values.get("aux_cat_attenuation_samples", 5))

        print("Device:", device)
        print("Optimizer:", self.optimizer)
        print("Scheduler:", self.scheduler)
        print(f"Epochs: {self.epochs}, mini-batch size: {self.mini_batches}, shuffle: {self.shuffle}")
        print("Regularization:", self.regularization_term)
        print("Teacher forcing mode:", self.teacher_forcing_mode)
        if self.teacher_forcing_mode == "fixed":
            print("Fixed teacher forcing ratio:", self.fixed_teacher_forcing_ratio)
        else:
            print(f"Scheduled sampling ε: 0.0 -> {self.scheduled_sampling_epsilon_max} (inverse-sigmoid)")

    def _select_activity_feature_name(self, cat_features_indeces, predictions_cat):
        """
        Pick the categorical activity feature key from decoder outputs.
        """
        if "concept:name" in cat_features_indeces:
            return "concept:name"

        for feature_name in cat_features_indeces.keys():
            lowered = feature_name.lower()
            if "concept" in lowered and "name" in lowered:
                return feature_name
            if "activity" in lowered:
                return feature_name

        mean_feature_names = [key[:-5] for key in predictions_cat.keys() if key.endswith("_mean")]
        if len(mean_feature_names) == 0:
            raise ValueError("No categorical activity prediction head found.")
        return mean_feature_names[0]
    
    def train_model(self,
                    use_statics: Optional[bool] = False,
                    use_zero_padd_masking: Optional[bool] = False,
                    use_eos_padd_masking: Optional[bool] = False):
        """
        Run the full epoch loop.

        Returns:
        - train_attenuated_losses: per-epoch mean attenuated training loss (total)
        - val_losses: per-epoch standard CE validation loss
        - val_attenuated_losses: per-epoch attenuated CE validation loss
        - train_sem_losses: per-epoch mean raw semantic loss L_sem (0.0 when
          lambda_sem == 0, since no decision-aware term is added)
        """
        self.model.train()

        train_attenuated_losses = []
        train_sem_losses = []
        val_losses = []
        val_attenuated_losses = []

        val_dataloader = self._build_dataloader(self.data_val, num_workers=4)

        for epoch in tqdm(range(self.epochs)):
            train_dataloader = self._build_dataloader(self.data_train, num_workers=0)

            epoch_loss = 0.0
            epoch_sem_loss = 0.0
            num_batches_per_epoch = 0

            self.scheduled_sampling_epsilon, self.teacher_forcing_ratio = self._teacher_forcing_rates(
                step_index=epoch,
                epsilon_max=self.scheduled_sampling_epsilon_max,
                inverse_sigmoid_k=self.scheduled_sampling_k,
                min_teacher_forcing=self.min_teacher_forcing_value,
            )

            for train_data in train_dataloader:
                batch = self._unpack_batch_common(train_data)
                cats = batch["cats"]
                nums = batch["nums"]
                eos_paddings = batch["eos_paddings"]
                zero_paddings = batch["zero_paddings"]

                static_inputs = self._prepare_static_inputs(batch["cats_static"], batch["nums_static"]) if use_statics else None

                prefixes, suffixes = self._split_prefix_suffix(cats=cats,
                                                               nums=nums,
                                                               suffix_size=self.suffix_data_split_value)

                prefix_mask, eos_paddings_suffix = self._build_masks(eos_paddings=eos_paddings,
                                                                    zero_paddings=zero_paddings,
                                                                    suffix_size=self.suffix_data_split_value,
                                                                    use_zero_padd_masking=use_zero_padd_masking,
                                                                    use_eos_padd_masking=use_eos_padd_masking)

                z_suffix_targets, z_suffix_mask = self._extract_guard_suffix(batch["z_targets"],
                                                                             batch["z_mask"],
                                                                             suffix_size=self.suffix_data_split_value)

                all_losses, loss_value = self.train_epoch(prefixes=prefixes,
                                                          suffixes=suffixes,
                                                          eos_paddings=eos_paddings_suffix,
                                                          prefix_mask=prefix_mask,
                                                          static_inputs=static_inputs,
                                                          z_targets=z_suffix_targets,
                                                          z_mask=z_suffix_mask)

                epoch_loss += loss_value.item()
                if "semantic" in all_losses:
                    epoch_sem_loss += all_losses["semantic"].item()
                num_batches_per_epoch += 1

            epoch_loss_train = epoch_loss / max(1, num_batches_per_epoch)
            epoch_sem_loss_train = epoch_sem_loss / max(1, num_batches_per_epoch)
            train_attenuated_losses.append(epoch_loss_train)
            train_sem_losses.append(epoch_sem_loss_train)

            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {self._current_lr()}, "
                       f"Teacher forcing ratio: {self.teacher_forcing_ratio:.4f}, "
                       f"Scheduled sampling epsilon: {self.scheduled_sampling_epsilon:.4f}")
            tqdm.write(f"Training: Avg Attenuated Training Loss (total): {epoch_loss_train:.4f}")
            if self.lambda_sem > 0:
                tqdm.write(f"Training: Avg Semantic Loss L_sem (raw): {epoch_sem_loss_train:.4f}, "
                           f"weighted λ_sem·L_sem: {self.lambda_sem * epoch_sem_loss_train:.4f}")

            epoch_loss_val_std, epoch_loss_val_unc = self.validation_epoch(val_dataloader=val_dataloader,
                                                                           use_statics=use_statics,
                                                                           use_zero_padd_masking=use_zero_padd_masking,
                                                                           use_eos_padd_masking=use_eos_padd_masking)
            val_losses.append(epoch_loss_val_std)
            val_attenuated_losses.append(epoch_loss_val_unc)

            tqdm.write(f"Validation: Avg Standard Validation Loss: {epoch_loss_val_std:.4f}")
            tqdm.write(f"Validation: Avg Attenuated Validation Loss: {epoch_loss_val_unc:.4f}")

            self._step_scheduler(epoch_loss_val_std)
            self._save_if_due(epoch)

        print("Training complete.")
        self._save_model()
        tqdm.write(f"Model saved to path: {self.saving_path}")

        return train_attenuated_losses, val_losses, val_attenuated_losses, train_sem_losses

    def train_epoch(self, prefixes, suffixes, eos_paddings, prefix_mask=None, static_inputs=None,
                    z_targets=None, z_mask=None):
        """
        Single optimization step on one mini-batch.
        Returns (all_losses_dict, total_loss).
        """
        predictions, _, _, data_features_indeces_dec, tf_mask = self.model(prefixes=prefixes,
                                                                           suffixes=suffixes,
                                                                           teacher_forcing_ratio=self.teacher_forcing_ratio,
                                                                           static_inputs=static_inputs,
                                                                           prefix_mask=prefix_mask,
                                                                           return_teacher_forcing_mask=True)

        predictions_cat, predictions_num = predictions
        cat_features_indeces, num_features_indeces = data_features_indeces_dec
        cat_suffixes, num_suffixes = suffixes

        cat_suffixes_dict = {name: cat_suffixes[idx] for name, idx in cat_features_indeces.items()}
        num_suffixes_dict = {name: num_suffixes[idx] for name, idx in num_features_indeces.items()}

        # Activity head: uncertainty-attenuated cross entropy.
        activity_feature_name = self._select_activity_feature_name(cat_features_indeces, predictions_cat)
        mean_cat_pred = predictions_cat[f"{activity_feature_name}_mean"]
        var_cat_pred = predictions_cat.get(f"{activity_feature_name}_var")
        target_cat = cat_suffixes_dict[activity_feature_name]

        loss_cat = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred,
                                                                pred_logvars=var_cat_pred,
                                                                T=30,
                                                                targets=target_cat.long(),
                                                                eos_paddings=eos_paddings)
        all_losses = {activity_feature_name: loss_cat}
        loss_terms = [loss_cat]

        # Auxiliary heads (non-activity categorical): uncertainty-attenuated CE.
        for feature_name in cat_features_indeces:
            if feature_name == activity_feature_name:
                continue
            mean_pred = predictions_cat.get(f"{feature_name}_mean")
            var_pred = predictions_cat.get(f"{feature_name}_var")
            target = cat_suffixes_dict.get(feature_name)
            if mean_pred is None or target is None:
                continue
            aux_loss = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_pred,
                                                                    pred_logvars=var_pred,
                                                                    T=self.aux_cat_attenuation_samples,
                                                                    targets=target.long(),
                                                                    eos_paddings=eos_paddings) * self.aux_cat_loss_weight
            all_losses[feature_name] = aux_loss
            loss_terms.append(aux_loss)

        # Auxiliary heads (numerical): Gaussian NLL.
        for feature_name in num_features_indeces:
            mean_pred = predictions_num.get(f"{feature_name}_mean")
            var_pred = predictions_num.get(f"{feature_name}_var")
            target = num_suffixes_dict.get(feature_name)
            if mean_pred is None or target is None:
                continue
            num_loss = self.loss_obj.gaussian_nll_loss(pred_means=mean_pred,
                                                      pred_logvars=var_pred,
                                                      targets=target.float(),
                                                      eos_paddings=eos_paddings) * self.aux_num_loss_weight
            all_losses[feature_name] = num_loss
            loss_terms.append(num_loss)

        weight_reg_enc, bias_reg_enc = self.model.encoder.regularizer()
        weight_reg_dec, bias_reg_dec = self.model.decoder.regularizer()
        weight_reg = (weight_reg_enc + weight_reg_dec).to(self.device)
        bias_reg = (bias_reg_enc + bias_reg_dec).to(self.device)

        data_loss = torch.stack(loss_terms).sum()
        loss = data_loss + self.regularization_term * (weight_reg + bias_reg)

        if self.lambda_sem > 0 and z_targets is not None:
            sem_loss = self.loss_obj.semantic_loss(pred_logits=mean_cat_pred,
                                                   guard_targets=z_targets,
                                                   guard_mask=z_mask,
                                                   tau=self.tau,
                                                   eos_paddings=eos_paddings,
                                                   teacher_forcing_mask=tf_mask,
                                                   gt_targets=target_cat.long(),
                                                   gt_in_support_only=self.sem_gate_gt_in_support)
            loss = loss + self.lambda_sem * sem_loss
            # Track raw (unweighted) L_sem for logging / diagnostics.
            all_losses["semantic"] = sem_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        return all_losses, loss

    def validation_epoch(self, val_dataloader, use_statics=False, use_zero_padd_masking=False, use_eos_padd_masking=False):
        """
        Validates the model on the validation set during training.
        """
        # Set model to evaluation mode
        self.model.eval()

        loss_std_total = 0.0
        loss_unc_total = 0.0
        num_batches = 0

        with torch.no_grad():
            for val_data in val_dataloader:
                batch = self._unpack_batch_common(val_data)

                static_inputs = self._prepare_static_inputs(batch["cats_static"], batch["nums_static"]) if use_statics else None

                prefixes, suffixes = self._split_prefix_suffix(cats=batch["cats"],
                                                               nums=batch["nums"],
                                                               suffix_size=self.suffix_data_split_value)
                prefix_mask, eos_paddings_suffix = self._build_masks(eos_paddings=batch["eos_paddings"],
                                                                    zero_paddings=batch["zero_paddings"],
                                                                    suffix_size=self.suffix_data_split_value,
                                                                    use_zero_padd_masking=use_zero_padd_masking,
                                                                    use_eos_padd_masking=use_eos_padd_masking)

                predictions, _, _, data_features_indeces_dec = self.model(prefixes=prefixes,
                                                                          suffixes=suffixes,
                                                                          teacher_forcing_ratio=self.teacher_forcing_ratio,
                                                                          static_inputs=static_inputs,
                                                                          prefix_mask=prefix_mask)
                predictions_cat, _ = predictions
                cat_features_indeces, _ = data_features_indeces_dec
                cat_suffixes, _ = suffixes

                activity_feature_name = self._select_activity_feature_name(cat_features_indeces, predictions_cat)
                mean_cat_pred = predictions_cat[f"{activity_feature_name}_mean"]
                var_cat_pred = predictions_cat.get(f"{activity_feature_name}_var")
                target_cat = cat_suffixes[cat_features_indeces[activity_feature_name]].long()

                loss_std_total += self.loss_obj.standard_cross_entropy(pred_logits=mean_cat_pred,
                                                                       targets=target_cat,
                                                                       eos_paddings=eos_paddings_suffix).item()
                loss_unc_total += self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred,
                                                                               pred_logvars=var_cat_pred,
                                                                               T=30,
                                                                               targets=target_cat,
                                                                               eos_paddings=eos_paddings_suffix).item()
                num_batches += 1

        self.model.train()

        denom = max(1, num_batches)
        return loss_std_total / denom, loss_unc_total / denom


#
# Trainer for the Camargo C-LSTM (next-event prediction).
#
class CTraining(Trainer):
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 optimize_values,
                 concept_name_id,
                 eos_id,
                 loss_obj=None,
                 lambda_sem: float = 0.0,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'C_LSTM.pkl'):

        super().__init__(device=device,
                         model=model,
                         data_train=data_train,
                         data_val=data_val,
                         optimize_values=optimize_values,
                         save_model_n_th_epoch=save_model_n_th_epoch,
                         saving_path=saving_path)

        self.loss_obj = loss_obj if loss_obj is not None else Loss()
        self.concept_name_id = concept_name_id
        self.eos_id = eos_id
        self.lambda_sem = lambda_sem

        # Auxiliary loss weights for non-activity dynamic attribute heads.
        self.aux_cat_loss_weight = float(optimize_values.get("aux_cat_loss_weight", 0.25))
        self.aux_num_loss_weight = float(optimize_values.get("aux_num_loss_weight", 0.25))

        # Map model feature names to dataset tensor indices.
        self.prefix_cat_feature_indices = None
        self.prefix_num_feature_indices = None
        self._init_prefix_feature_indices()

        print("Device:", device)
        print("Optimizer:", self.optimizer)
        print("Scheduler:", self.scheduler)
        print(f"Epochs: {self.epochs}, mini-batch size: {self.mini_batches}, shuffle: {self.shuffle}")

    def _init_prefix_feature_indices(self):
        """Map configured model feature names to dataset tensor indices."""
        cat_categories, num_categories = self.data_train.all_categories
        cat_names_dataset = [cat[0] for cat in cat_categories]
        num_names_dataset = [num[0] for num in num_categories]

        model_feat = getattr(self.model, "model_feat", None)
        if model_feat is None:
            # Fallback: model does not expose feature config; use all dynamic features.
            self.prefix_cat_feature_indices = list(range(len(cat_names_dataset)))
            self.prefix_num_feature_indices = list(range(len(num_names_dataset)))
            return

        model_cat_names, model_num_names = model_feat

        missing_cat = [name for name in model_cat_names if name not in cat_names_dataset]
        missing_num = [name for name in model_num_names if name not in num_names_dataset]
        if missing_cat or missing_num:
            raise ValueError(f"Configured model features are missing in dataset categories. "
                             f"Missing categorical: {missing_cat}, missing numerical: {missing_num}.")

        self.prefix_cat_feature_indices = [cat_names_dataset.index(name) for name in model_cat_names]
        self.prefix_num_feature_indices = [num_names_dataset.index(name) for name in model_num_names]

    def _preprocess_batch(self, cats, nums, eos_paddings=None):
        """
        Build prefixes + next-activity targets (supervision length S=1).
        Filters out traces with more than two EOS tokens.

        Returns:
        - prefixes:   [[prefix_cat], [prefix_num]]
        - target_act: next-activity labels (V,)
        - eos_next:   EOS mask for the next step (V, 1) or None
        - V:          number of valid traces in the batch
        - valid_mask: original-batch boolean mask used for filtering
        """
        if len(cats) == 0:
            return None, None, None, 0, None

        valid_mask = torch.ones(cats[0].shape[0], dtype=torch.bool, device=cats[0].device)
        if self.eos_id is not None:
            eos_counts = (cats[self.concept_name_id] == self.eos_id).sum(dim=1)
            valid_mask = eos_counts <= 2

        V = int(valid_mask.sum().item())
        if V == 0:
            return None, None, None, 0, None

        batch_cats = [cat[valid_mask] for cat in cats]
        batch_nums = [num[valid_mask] for num in nums]

        selected_cats = [batch_cats[i] for i in self.prefix_cat_feature_indices]
        selected_nums = [batch_nums[i] for i in self.prefix_num_feature_indices]

        prefixes_cat = [cat[:, :-1].to(self.device) for cat in selected_cats]
        prefixes_num = [num[:, :-1].to(self.device) for num in selected_nums]
        prefixes = [prefixes_cat, prefixes_num]

        target_act = batch_cats[self.concept_name_id][:, -1].to(self.device).long()

        eos_next = None
        if eos_paddings is not None:
            eos_next = eos_paddings[valid_mask][:, -1:].to(self.device)

        return prefixes, target_act, eos_next, V, valid_mask

    def train(self):
        """Run full training and validation loops.

        Returns (train_losses, val_losses, train_sem_losses), where
        train_sem_losses is the per-epoch mean raw semantic loss L_sem (0.0 when
        lambda_sem == 0, since no decision-aware term is added).
        """
        train_losses = []
        val_losses = []
        train_sem_losses = []

        val_dataloader = self._build_dataloader(self.data_val, num_workers=4)

        # Precompute name->index maps for auxiliary losses (avoid per-batch recomputation).
        cat_categories_dataset, num_categories_dataset = self.data_train.all_categories
        cat_name_to_dataset_idx = {cat[0]: i for i, cat in enumerate(cat_categories_dataset)}
        num_name_to_dataset_idx = {num[0]: i for i, num in enumerate(num_categories_dataset)}

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_dataloader = self._build_dataloader(self.data_train, num_workers=4)

            total = 0.0
            total_sem = 0.0
            num_batches = 0

            for train_cases in train_dataloader:
                batch = self._unpack_batch_common(train_cases)
                z_targets_full = batch["z_targets"]
                z_mask_full = batch["z_mask"]

                prefixes, target_act, eos_next, V, valid_mask = self._preprocess_batch(cats=batch["cats"],
                                                                                       nums=batch["nums"],
                                                                                       eos_paddings=batch["eos_paddings"])
                if V == 0:
                    continue

                model_out = self.model(prefixes, return_dict=True)
                a_logits = model_out["activity_logits"]
                other_cat_logits = model_out["other_cat_logits"]
                num_means = model_out["num_means"]

                # Activity: standard CE (single-step sequence => shape [1, V, C]).
                pred_logits = a_logits.unsqueeze(0)
                loss = self.loss_obj.standard_cross_entropy(pred_logits=pred_logits,
                                                            targets=target_act.unsqueeze(1),
                                                            eos_paddings=eos_next)

                # Auxiliary categorical heads.
                for feat_name, feat_logits in other_cat_logits.items():
                    dataset_idx = cat_name_to_dataset_idx.get(feat_name)
                    if dataset_idx is None:
                        continue
                    target_other = batch["cats"][dataset_idx][valid_mask][:, -1].to(self.device).long()
                    aux_loss = self.loss_obj.standard_cross_entropy(pred_logits=feat_logits.unsqueeze(0),
                                                                    targets=target_other.unsqueeze(1),
                                                                    eos_paddings=eos_next)
                    loss = loss + self.aux_cat_loss_weight * aux_loss

                # Auxiliary numerical heads (simple MSE; no variance head learned).
                for feat_name, feat_means in num_means.items():
                    dataset_idx = num_name_to_dataset_idx.get(feat_name)
                    if dataset_idx is None:
                        continue
                    target_num = batch["nums"][dataset_idx][valid_mask][:, -1].to(self.device).float()
                    loss = loss + self.aux_num_loss_weight * torch.mean((feat_means - target_num) ** 2)

                # Decision-aware semantic loss on the last prefix event (position -2).
                # C-LSTM is next-event prediction so every step is teacher-forced;
                # tf_mask is therefore omitted.
                if self.lambda_sem > 0 and z_targets_full.shape[-1] > 0:
                    gt = z_targets_full[valid_mask][:, -2, :].unsqueeze(1).to(self.device)  # [V, 1, C]
                    gm = z_mask_full[valid_mask][:, -2].unsqueeze(1).to(self.device)        # [V, 1]
                    sem_loss = self.loss_obj.semantic_loss(pred_logits=pred_logits,
                                                           guard_targets=gt,
                                                           guard_mask=gm,
                                                           tau=self.tau,
                                                           eos_paddings=eos_next,
                                                           gt_targets=target_act.unsqueeze(1),
                                                           gt_in_support_only=self.sem_gate_gt_in_support)
                    loss = loss + self.lambda_sem * sem_loss
                    total_sem += sem_loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total += loss.item()
                num_batches += 1

            epoch_loss = total / max(1, num_batches)
            epoch_sem_loss = total_sem / max(1, num_batches)
            train_losses.append(epoch_loss)
            train_sem_losses.append(epoch_sem_loss)

            val_loss = self._validate(loader=val_dataloader)
            val_losses.append(val_loss)

            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {self._current_lr()}")
            tqdm.write(f"Training: Avg Training Loss: {epoch_loss:.4f}")
            if self.lambda_sem > 0:
                tqdm.write(f"Training: Avg Semantic Loss L_sem (raw): {epoch_sem_loss:.4f}, "
                           f"weighted λ_sem·L_sem: {self.lambda_sem * epoch_sem_loss:.4f}")
            tqdm.write(f"Validation: Avg Validation Loss: {val_loss:.4f}")

            self._step_scheduler(val_loss)
            self._save_if_due(epoch)

        print("Training complete.")
        self._save_model()
        tqdm.write(f"Model saved to path: {self.saving_path}")

        return train_losses, val_losses, train_sem_losses

    def _validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for val_batch in loader:
                batch = self._unpack_batch_common(val_batch)
                prefixes, target_act, eos_next, V, _ = self._preprocess_batch(cats=batch["cats"],
                                                                              nums=batch["nums"],
                                                                              eos_paddings=batch["eos_paddings"])
                if V == 0:
                    continue

                a_logits = self.model(input=prefixes)
                act_loss = self.loss_obj.standard_cross_entropy(pred_logits=a_logits.unsqueeze(0),
                                                                targets=target_act.unsqueeze(1),
                                                                eos_paddings=eos_next)
                total_loss += act_loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)


#
#
# trainings class for Taymouri et.al. GAN based LSTM for suffix prediction
#
#
class TTraining(Trainer):
    """
    Trainer for Taymouri's GAN encoder-decoder LSTM.
    Implements adversarial training with Gumbel-softmax for differentiable categorical suffix generation:
    """

    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 optimize_values,
                 suffix_data_split_value,
                 concept_name_id,
                 eos_id,
                 loss_obj=None,
                 lambda_sem: float = 0.0,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'taymouri_model.pkl'):

        super().__init__(device=device,
                         model=model,
                         data_train=data_train,
                         data_val=data_val,
                         optimize_values=optimize_values,
                         save_model_n_th_epoch=save_model_n_th_epoch,
                         saving_path=saving_path)

        self.loss_obj = loss_obj if loss_obj is not None else Loss()
        self.suffix_data_split_value = suffix_data_split_value
        self.concept_name_id = concept_name_id
        self.eos_id = eos_id
        self.lambda_sem = lambda_sem

        # Auxiliary loss weights for non-activity dynamic attribute heads.
        self.aux_cat_loss_weight = float(optimize_values.get("aux_cat_loss_weight", 0.25))
        self.aux_num_loss_weight = float(optimize_values.get("aux_num_loss_weight", 0.25))

        # Teacher forcing (scheduled-sampling parameters override the base policy).
        self.min_teacher_forcing_value = optimize_values.get("min_teacher_forcing_value", 0.0)
        self.max_teacher_forcing_value = optimize_values.get("max_teacher_forcing_value", 0.5)
        self.scheduled_sampling_epsilon_max = optimize_values.get("scheduled_sampling_epsilon_max", self.max_teacher_forcing_value)
        self.scheduled_sampling_k = optimize_values.get("scheduled_sampling_k", max(1.0, self.epochs / 10.0))

        # Gumbel-softmax temperature annealing (distinct from semantic-loss tau).
        self.tau_start = optimize_values.get("tau_start", 0.9)
        self.tau_min = optimize_values.get("tau_min", 0.01)

        # GAN vs MLE-only switch.
        self.use_gan = optimize_values.get("use_gan", True)

        # Optimizers: G (encoder-decoder) and D (discriminator).
        self.generator_optimizer = optimize_values.get("generator_optimizer", optimize_values.get("optimizer", None))
        if self.generator_optimizer is None:
            raise ValueError("Provide `generator_optimizer` (or `optimizer`) in optimize_values.")

        self.discriminator_optimizer = optimize_values.get("discriminator_optimizer", None)
        if self.use_gan and self.discriminator_optimizer is None:
            raise ValueError("Provide `discriminator_optimizer` in optimize_values for GAN training.")

        self.generator_scheduler = optimize_values.get("generator_scheduler", optimize_values.get("scheduler", None))
        self.discriminator_scheduler = optimize_values.get("discriminator_scheduler", None) if self.use_gan else None

        # Ensure the generator optimizer also updates prefix embeddings.
        self._register_missing_generator_parameters()

        # Feature projection: model_feat -> dataset tensor indices.
        self.prefix_cat_feature_indices = None
        self.prefix_num_feature_indices = None
        self._init_prefix_feature_indices()

        print("Device:", device)
        print("Mode:", "GAN (Algorithm 1: MLMME)" if self.use_gan else "MLE-only")
        print(f"Epochs (iterations): {self.epochs}")
        print(f"Gumbel-softmax τ: {self.tau_start} → {self.tau_min} (exponential anneal)")
        print("Teacher forcing mode:", self.teacher_forcing_mode)
        if self.teacher_forcing_mode == "fixed":
            print("Fixed teacher forcing ratio:", self.fixed_teacher_forcing_ratio)
        else:
            print(f"Scheduled sampling ε: 0.0 -> {self.scheduled_sampling_epsilon_max} (inverse-sigmoid)")

    def _init_prefix_feature_indices(self):
        """Map configured model feature names to dataset tensor indices."""
        cat_categories, num_categories = self.data_train.all_categories
        cat_names_dataset = [cat[0] for cat in cat_categories]
        num_names_dataset = [num[0] for num in num_categories]

        model_feat = getattr(self.model, "model_feat", None)
        if model_feat is None:
            self.prefix_cat_feature_indices = list(range(len(cat_names_dataset)))
            self.prefix_num_feature_indices = list(range(len(num_names_dataset)))
            return

        model_cat_names, model_num_names = model_feat

        missing_cat = [name for name in model_cat_names if name not in cat_names_dataset]
        missing_num = [name for name in model_num_names if name not in num_names_dataset]
        if missing_cat or missing_num:
            raise ValueError(f"Configured model features are missing in dataset categories. "
                             f"Missing categorical: {missing_cat}, missing numerical: {missing_num}.")

        self.prefix_cat_feature_indices = [cat_names_dataset.index(name) for name in model_cat_names]
        self.prefix_num_feature_indices = [num_names_dataset.index(name) for name in model_num_names]

    def _unpack_batch(self, batch):
        """
        Split batch into model-projected prefixes, activity targets, EOS mask, guard
        data, and auxiliary-head targets (non-activity categorical + numerical suffix
        tensors keyed by feature name).
        """
        unpacked = self._unpack_batch_common(batch)
        eos_paddings = unpacked["eos_paddings"]

        prefixes, suffixes = self._split_prefix_suffix(cats=unpacked["cats"],
                                                       nums=unpacked["nums"],
                                                       suffix_size=self.suffix_data_split_value)

        # Project prefix features to model_feat subset.
        prefix_cats, prefix_nums = prefixes
        prefixes = [[prefix_cats[i] for i in self.prefix_cat_feature_indices],
                    [prefix_nums[i] for i in self.prefix_num_feature_indices]]

        act_targets = suffixes[0][self.concept_name_id].long()

        # Auxiliary suffix targets keyed by feature name (used to supervise aux heads).
        cat_categories_dataset, num_categories_dataset = self.data_train.all_categories
        other_cat_names = getattr(self.model, "_other_cat_feature_names", [])
        num_names = getattr(self.model, "_num_feature_names", [])

        aux_cat_targets = {}
        for i, cat in enumerate(cat_categories_dataset):
            feat_name = cat[0]
            if i == self.concept_name_id or feat_name not in other_cat_names:
                continue
            aux_cat_targets[feat_name] = suffixes[0][i].long().to(self.device)

        aux_num_targets = {}
        for i, num in enumerate(num_categories_dataset):
            feat_name = num[0]
            if feat_name not in num_names:
                continue
            aux_num_targets[feat_name] = suffixes[1][i].float().to(self.device)

        eos_suffix = None if eos_paddings is None else eos_paddings[:, -self.suffix_data_split_value:].to(self.device)

        z_suffix_targets, z_suffix_mask = self._extract_guard_suffix(unpacked["z_targets"],
                                                                     unpacked["z_mask"],
                                                                     self.suffix_data_split_value)

        return prefixes, act_targets, eos_suffix, z_suffix_targets, z_suffix_mask, aux_cat_targets, aux_num_targets

    def _generator_parameters(self):
        generator_params = list(self.model.seq2seq.parameters())
        if hasattr(self.model, "embeddings"):
            generator_params.extend(list(self.model.embeddings.parameters()))
        return generator_params

    def _register_missing_generator_parameters(self):
        optimizer_param_ids = {id(param) for group in self.generator_optimizer.param_groups for param in group["params"]}
        missing_params = [param for param in self._generator_parameters() if id(param) not in optimizer_param_ids]
        if missing_params:
            self.generator_optimizer.add_param_group({"params": missing_params})

    def _sequence_discriminator_logits(self, suffix_probabilities):
        """
        Use the last discriminator state as a sequence-level real/fake logit.
        """
        return self.model.discriminator(suffix_probabilities)[:, -1, 0]

    def _masked_activity_loss(self, logits, targets, eos_mask=None):
        """
        Cross-entropy loss. logits: [S, B, C], targets: [B, S].
        """
        return self.loss_obj.standard_cross_entropy(pred_logits=logits,
                                                    targets=targets,
                                                    eos_paddings=eos_mask)

    def train(self):
        """Adversarial training (Algorithm 1 / MLE-only when use_gan=False).

        Returns (train_gen_losses, train_disc_losses, val_losses, train_sem_losses),
        where train_sem_losses is the per-epoch mean raw semantic loss L_sem (0.0
        when lambda_sem == 0, since no decision-aware term is added).
        """
        train_gen_losses = []
        train_disc_losses = []
        val_losses = []
        train_sem_losses = []

        val_loader = self._build_dataloader(self.data_val, num_workers=4)

        # Exponential annealing rate: τ_t = max(tau_min, tau_start * exp(-rate * t))
        anneal_rate = math.log(self.tau_start / self.tau_min) / (self.epochs - 1) if self.epochs > 1 else 0.0

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loader = self._build_dataloader(self.data_train, num_workers=4)

            tau = max(self.tau_min, self.tau_start * math.exp(-anneal_rate * epoch))
            self.scheduled_sampling_epsilon, self.teacher_forcing_ratio = self._teacher_forcing_rates(
                step_index=epoch,
                epsilon_max=self.scheduled_sampling_epsilon_max,
                inverse_sigmoid_k=self.scheduled_sampling_k,
                min_teacher_forcing=self.min_teacher_forcing_value,
            )

            gen_loss_total = 0.0
            disc_loss_total = 0.0
            sem_loss_total = 0.0
            n_batches = 0

            for batch in train_loader:
                prefixes, target_suffix_act, eos_suffix, z_suffix_targets, z_suffix_mask, aux_cat_targets, aux_num_targets = self._unpack_batch(batch)

                # Discriminator step (Algorithm 1, line 4).
                if self.use_gan:
                    real_onehot = F.one_hot(target_suffix_act, self.model.output_size_act).float()
                    with torch.no_grad():
                        logits_d = self.model(prefixes=prefixes,
                                              target_suffix=target_suffix_act,
                                              teacher_forcing_ratio=self.teacher_forcing_ratio)  # [S, B, C]
                    fake_gumbel_d = F.gumbel_softmax(logits_d.permute(1, 0, 2).detach(), tau=tau, hard=False, dim=-1)

                    d_real = self._sequence_discriminator_logits(real_onehot)
                    d_fake = self._sequence_discriminator_logits(fake_gumbel_d)
                    disc_loss = (F.binary_cross_entropy_with_logits(d_real, torch.ones_like(d_real))
                                 + F.binary_cross_entropy_with_logits(d_fake, torch.zeros_like(d_fake)))

                    self.discriminator_optimizer.zero_grad()
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                    self.discriminator_optimizer.step()
                else:
                    disc_loss = torch.tensor(0.0, device=self.device)

                # Generator step (Algorithm 1, line 5): minimize L(G; D) + L_supervised + aux + L_sem.
                logits_g, tf_mask, other_cat_logits, num_means = self.model(prefixes=prefixes,
                                                                            target_suffix=target_suffix_act,
                                                                            teacher_forcing_ratio=self.teacher_forcing_ratio,
                                                                            return_teacher_forcing_mask=True,
                                                                            return_aux_predictions=True)

                loss_supervised = self._masked_activity_loss(logits_g, target_suffix_act, eos_mask=eos_suffix)

                if self.use_gan:
                    fake_gumbel_g = F.gumbel_softmax(logits_g.permute(1, 0, 2), tau=tau, hard=False, dim=-1)
                    d_fake_g = self._sequence_discriminator_logits(fake_gumbel_g)
                    adv_loss = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))
                    gen_loss = adv_loss + loss_supervised
                else:
                    gen_loss = loss_supervised

                # Auxiliary categorical heads.
                for feat_name, feat_logits in other_cat_logits.items():
                    target = aux_cat_targets.get(feat_name)
                    if target is None:
                        continue
                    aux_loss = self.loss_obj.standard_cross_entropy(pred_logits=feat_logits,
                                                                    targets=target,
                                                                    eos_paddings=eos_suffix)
                    gen_loss = gen_loss + self.aux_cat_loss_weight * aux_loss

                # Auxiliary numerical heads (MSE, EOS-weighted).
                for feat_name, feat_means in num_means.items():
                    target = aux_num_targets.get(feat_name)
                    if target is None:
                        continue
                    # feat_means: [S, B]; target: [B, S]
                    pred_aligned = feat_means.permute(1, 0)
                    if eos_suffix is not None:
                        weight = eos_suffix.float()
                        num_loss = torch.sum(weight * (pred_aligned - target) ** 2) / weight.sum().clamp(min=1e-8)
                    else:
                        num_loss = torch.mean((pred_aligned - target) ** 2)
                    gen_loss = gen_loss + self.aux_num_loss_weight * num_loss

                # Decision-aware semantic loss (set-membership over tau-support)
                if self.lambda_sem > 0 and z_suffix_targets is not None:
                    sem_loss = self.loss_obj.semantic_loss(pred_logits=logits_g,
                                                           guard_targets=z_suffix_targets,
                                                           guard_mask=z_suffix_mask,
                                                           tau=self.tau,
                                                           eos_paddings=eos_suffix,
                                                           teacher_forcing_mask=tf_mask,
                                                           gt_targets=target_suffix_act,
                                                           gt_in_support_only=self.sem_gate_gt_in_support)
                    gen_loss = gen_loss + self.lambda_sem * sem_loss
                    sem_loss_total += sem_loss.item()

                self.generator_optimizer.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._generator_parameters(), max_norm=1.0)
                self.generator_optimizer.step()

                gen_loss_total += gen_loss.item()
                disc_loss_total += disc_loss.item()
                n_batches += 1

            epoch_gen_loss = gen_loss_total / max(1, n_batches)
            epoch_disc_loss = disc_loss_total / max(1, n_batches)
            epoch_sem_loss = sem_loss_total / max(1, n_batches)
            train_gen_losses.append(epoch_gen_loss)
            train_disc_losses.append(epoch_disc_loss)
            train_sem_losses.append(epoch_sem_loss)

            val_loss = self._validate(val_loader)
            val_losses.append(val_loss)

            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], LR: {self._current_lr()}, "
                       f"τ: {tau:.4f}, TF: {self.teacher_forcing_ratio:.4f}, ε: {self.scheduled_sampling_epsilon:.4f}, "
                       f"Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}")
            if self.lambda_sem > 0:
                tqdm.write(f"Training: Avg Semantic Loss L_sem (raw): {epoch_sem_loss:.4f}, "
                           f"weighted λ_sem·L_sem: {self.lambda_sem * epoch_sem_loss:.4f}")

            if self.generator_scheduler is not None:
                self.generator_scheduler.step(val_loss)
            if self.use_gan and self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step(val_loss)

            self._save_if_due(epoch)

        print("Training complete.")
        self._save_model()
        tqdm.write(f"Model saved to path: {self.saving_path}")

        return train_gen_losses, train_disc_losses, val_losses, train_sem_losses

    def _validate(self, loader):
        """Validation CE loss on the activity-suffix head."""
        self.model.eval()
        val_loss_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                prefixes, target_suffix_act, eos_suffix, _, _, _, _ = self._unpack_batch(batch)
                logits = self.model(prefixes=prefixes, target_suffix=target_suffix_act, teacher_forcing_ratio=0.0)
                loss = self._masked_activity_loss(logits, target_suffix_act, eos_mask=eos_suffix)
                val_loss_total += loss.item()
                n_batches += 1

        self.model.train()
        return val_loss_total / max(1, n_batches)