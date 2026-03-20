"""
Training approaches from 3 different methods:
1) Mode (Arg-Max): LSTM,
2) Beam-Search (fixed n beam size): LSTM
3) Monte Carlo Suffix sampling: LSTM.

- clean training (standard)
- decision-aware training
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import math
from tqdm.notebook import tqdm
from typing import Optional

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
        self.guard_support_threshold = optimize_values.get("guard_support_threshold", 0.0)

        # Teacher forcing policy shared by autoregressive trainers.
        self.teacher_forcing_mode = str(optimize_values.get("teacher_forcing_mode", "scheduled")).lower()
        
        # fix not possible: makes no sense, to train only on target last events
        
        if self.teacher_forcing_mode not in {"scheduled", "fixed"}:
            raise ValueError("teacher_forcing_mode must be either 'scheduled' or 'fixed'")
        
        self.fixed_teacher_forcing_ratio = float(optimize_values.get("fixed_teacher_forcing_ratio", 1.0))
        
        self.fixed_teacher_forcing_ratio = max(0.0, min(1.0, self.fixed_teacher_forcing_ratio))

        self.save_model_n_th_epoch = save_model_n_th_epoch
        self.saving_path = saving_path

    def _build_dataloader(self, dataset, num_workers=8):
            return DataLoader(dataset=dataset,
                            batch_size=self.mini_batches,
                            shuffle=self.shuffle,
                            # turn off when debugging:
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

        # decision_data is a tuple (z_targets, z_mask[, c_values])
        if len(decision_data) == 3:
            z_targets, z_mask, c_values = decision_data
        else:
            z_targets, z_mask = decision_data
            c_values = None

        return {"cats": cats,
                "nums": nums,
                "eos_paddings": eos_paddings,
                "zero_paddings": zero_paddings,
                "cats_static": cats_static,
                "nums_static": nums_static,
                "z_targets": z_targets,
                "z_mask": z_mask,
                "c_values": c_values}

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

    # for U-ED-LSTM and T-GAN-LSTM
    def _split_prefix_suffix(self, cats, nums, suffix_size):
        prefixes_cat = [cat[:, :-suffix_size].to(self.device) for cat in cats]
        prefixes_num = [num[:, :-suffix_size].to(self.device) for num in nums]
        suffixes_cat = [cat[:, -suffix_size:].to(self.device) for cat in cats]
        suffixes_num = [num[:, -suffix_size:].to(self.device) for num in nums]
        return [prefixes_cat, prefixes_num], [suffixes_cat, suffixes_num]

    # for C-LSTM
    def _split_prefix_and_next_activity(self, cats, nums, concept_name_id):
        prefixes_cat = [cat[:, :-1].to(self.device) for cat in cats]
        prefixes_num = [num[:, :-1].to(self.device) for num in nums]
        target_act = cats[concept_name_id][:, -1].to(self.device).long()
        return [prefixes_cat, prefixes_num], target_act

    # zero padding and eos padding masks (Optional)
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

    def _extract_guard_suffix(self, z_targets, z_mask, suffix_size, c_values=None):
        """
        Per-event decision labels that are stored for the whole window and cut out exactly the subset that lines up with decoder steps during autoregressive suffix training
        The guard label for step s is therefore the label of event at position T-S-1+s.

        Outputs:
        - z_suffix_targets: [B, S, C] or None
        - z_suffix_mask: [B, S] or None
        - c_suffix_values: [B, S] or None
        """
        if z_targets.shape[-1] == 0:
            return None, None, None
        S = suffix_size
        
        z_suffix_targets = z_targets[:, -(S + 1):-1, :].to(self.device)
        z_suffix_mask = z_mask[:, -(S + 1):-1].to(self.device)
        c_suffix_values = None
        
        if c_values is not None:
            c_suffix_values = c_values[:, -(S + 1):-1].to(self.device)
        return z_suffix_targets, z_suffix_mask, c_suffix_values


#
# 
# Trainer for the U-ED-LSTM (with and without decision-awareness): 
#
#
class UEDTrainer(Trainer):
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 loss_obj,
                 log_normal_loss_num_feature,
                 optimize_values,
                 suffix_data_split_value,
                 lambda_g: float = 0.0,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'U_ED_LSTM_train.pkl'):
        
        # Standard Training parameters
        super().__init__(device=device,
                         model=model,
                         data_train=data_train,
                         data_val=data_val,
                         optimize_values=optimize_values,
                         save_model_n_th_epoch=save_model_n_th_epoch,
                         saving_path=saving_path)

        print("Device: ", device)
        print("Model: ", model)

        print("Train Dataset: ", data_train)
        print("Validation Dataset: ", data_val)
        
        self.loss_obj = loss_obj
        print("Loss object for method calling: ", loss_obj)
        self.log_normal_loss_num_feature = log_normal_loss_num_feature
        print("Num. feautures that follow log-normal PDF: ", log_normal_loss_num_feature)
        
        # Standard Optimization parameters
        self.regularization_term = optimize_values["regularization_term"]
        print("regularization: ", self.regularization_term)
        print("Optimizer: ", self.optimizer)
        print("Scheduler: ", self.scheduler)
        print("Epochs: ", self.epochs)
        print("Mini baches: ", self.mini_batches)
        print("Shuffle batched dataset: ", self.shuffle)
        
        # Teacher forcing
        self.min_teacher_forcing_value = optimize_values["min_teacher_forcing_value"]
        self.max_teacher_forcing_value = optimize_values["max_teacher_forcing_value"]
        self.scheduled_sampling_epsilon_max = optimize_values.get("scheduled_sampling_epsilon_max", self.max_teacher_forcing_value)
        self.scheduled_sampling_k = optimize_values.get("scheduled_sampling_k", max(1.0, self.epochs / 10.0))

        print("Teacher forcing mode:", self.teacher_forcing_mode)
        if self.teacher_forcing_mode == "fixed":
            print("Fixed teacher forcing ratio:", self.fixed_teacher_forcing_ratio)
        else:
            print("Scheduled sampling ε:",
                  f"0.0 -> {self.scheduled_sampling_epsilon_max} (inverse-sigmoid)",)
        
        # Events in sufffix: Dependent on data set
        self.suffix_data_split_value = suffix_data_split_value

        # Decision-aware guard regularization weight
        self.lambda_g = lambda_g

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
                    use_statics:Optional[bool]=False,
                    use_zero_padd_masking:Optional[bool]=False,
                    use_eos_padd_masking:Optional[bool]=False):
        """        
        Inputs:
        - use_statics:
        - use_zero_padd_masking:
        - use_eos_padd_masking:
        
        Outputs:
        - train_attenuated_losses:
        - val_losses
        - val_attenuated_losses
        """
        # Train the model
        self.model.train()

        # Lists to store the losses
        train_attenuated_losses = []
        val_losses = []
        val_attenuated_losses = []

        # Validation dataloader
        val_dataloader = self._build_dataloader(self.data_val, num_workers=4)
                
        # Trainings/ Epoch Loop
        for epoch in tqdm(range(self.epochs)):   # range(self.epochs):
       
            
            # Train dataloader
            train_dataloader = self._build_dataloader(self.data_train, num_workers=0)
            
            epoch_cat_loss = {}
            epoch_loss = 0.0
            num_batches_per_epoch = 0.0
            
            # Select teacher forcing policy (fixed ratio or scheduled sampling).
            self.scheduled_sampling_epsilon, self.teacher_forcing_ratio = self._teacher_forcing_rates(
                step_index=epoch,
                epsilon_max=self.scheduled_sampling_epsilon_max,
                inverse_sigmoid_k=self.scheduled_sampling_k,
                min_teacher_forcing=self.min_teacher_forcing_value,
            )
            
            # Bacth Loop
            for _, train_data in enumerate(train_dataloader): 
                batch = self._unpack_batch_common(train_data)
                cats = batch["cats"]
                nums = batch["nums"]
                eos_paddings = batch["eos_paddings"]
                zero_paddings = batch["zero_paddings"]
                cats_static = batch["cats_static"]
                nums_static = batch["nums_static"]
                z_targets = batch["z_targets"]
                z_mask = batch["z_mask"]
                c_values = batch["c_values"]

                # static data (only prefix input data)
                if use_statics:
                    static_inputs = self._prepare_static_inputs(cats_static, nums_static)
                else:
                    static_inputs = None

                prefixes, suffixes = self._split_prefix_suffix(cats=cats,
                                                               nums=nums,
                                                               suffix_size=self.suffix_data_split_value)
                
                prefix_mask, eos_paddings_suffix = self._build_masks(eos_paddings=eos_paddings,
                                                                     zero_paddings=zero_paddings,
                                                                     suffix_size=self.suffix_data_split_value,
                                                                     use_zero_padd_masking=use_zero_padd_masking,
                                                                     use_eos_padd_masking=use_eos_padd_masking)

                # Guard data aligned to decoder steps
                z_suffix_targets, z_suffix_mask, c_suffix_values = self._extract_guard_suffix(z_targets,
                                                                                              z_mask,
                                                                                              c_values=c_values,
                                                                                              suffix_size=self.suffix_data_split_value)

                # Optimization (categorical only)
                cat_losses_dict, loss_value = self.train_epoch(prefixes=prefixes,
                                                               suffixes=suffixes,
                                                               eos_paddings=eos_paddings_suffix,
                                                               prefix_mask=prefix_mask,
                                                               static_inputs=static_inputs,
                                                               z_targets=z_suffix_targets,
                                                               z_mask=z_suffix_mask,
                                                               c_values=c_suffix_values)
                
                # Loss calculation and output
                # Accumulate the categorical losses
                for feature_name in cat_losses_dict.keys():  
                    if feature_name in epoch_cat_loss:
                        # Add the current batch's loss to the cumulative loss
                        epoch_cat_loss[feature_name] += cat_losses_dict[feature_name].item()
                    else:
                        # Initialize the cumulative loss with the first batch's loss
                        epoch_cat_loss[feature_name] = cat_losses_dict[feature_name].item()

                # Accumulated total loss for the entire epoch
                epoch_loss += loss_value.item()
                                
                # Increase number of trained batches:
                num_batches_per_epoch += 1
                
            # Take the mean losses over all batches
            for feature_name in epoch_cat_loss.keys():
                epoch_cat_loss[feature_name] = epoch_cat_loss[feature_name] / num_batches_per_epoch
                
            epoch_loss_train = epoch_loss / num_batches_per_epoch

            # Current learning rate
            current_lr = self._current_lr()
            
            # Prints per Epoch:
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {current_lr}, "
                       f"Teacher forcing ratio: {self.teacher_forcing_ratio:.4f}, "
                       f"Scheduled sampling epsilon: {self.scheduled_sampling_epsilon:.4f}")
            
            tqdm.write(f"Training: Avg Attenuated Training Loss: {epoch_loss_train:.4f}")
            
            train_attenuated_losses.append(epoch_loss_train)
            
            # Validation            
            epoch_loss_val_std, epoch_loss_val_unc = self.validation_epoch(val_dataloader=val_dataloader,
                                                                            use_statics=use_statics,
                                                                            use_zero_padd_masking=use_zero_padd_masking,
                                                                            use_eos_padd_masking=use_eos_padd_masking)
                        
            tqdm.write(f"Validation: Avg Standard Validation Loss: {epoch_loss_val_std:.4f}")
            tqdm.write(f"Validation: Avg Attenuated Validation Loss: {epoch_loss_val_unc:.4f}")
        
            val_losses.append(epoch_loss_val_std)
            val_attenuated_losses.append(epoch_loss_val_unc)

            # Adjust the learning rate if necessary
            tqdm.write(f"Validation Loss for Scheduler: {epoch_loss_val_std:.4f}")
            
            # Adjust learning rate
            self._step_scheduler(epoch_loss_val_std)
            self._save_if_due(epoch)
                                 
        print("Training complete.")

        self._save_model()
        tqdm.write(f'Model saved to path: {self.saving_path}')

        return train_attenuated_losses, val_losses, val_attenuated_losses

    def train_epoch(self, prefixes, suffixes, eos_paddings, prefix_mask=None, static_inputs=None,
                    z_targets=None, z_mask=None, c_values=None):
        """
        one epoch iteration
        """
        # predictions: List of two Dicts one for categorical (means and vars), one for numerical (means and vars): key: feature name + _mean or _var, value: tensor with dim: seq len x batch size x output feature size
        # data_features_indeces_dec: List of two Dicts one for categorical, one for numerical: key: feature name, value: index of tensor in data list
        predictions, _, _, data_features_indeces_dec, tf_mask = self.model(prefixes=prefixes,
                                                                           suffixes=suffixes,
                                                                           teacher_forcing_ratio=self.teacher_forcing_ratio,
                                                                           static_inputs=static_inputs,
                                                                           # prefix mask for the encoder
                                                                           prefix_mask=prefix_mask,
                                                                           return_teacher_forcing_mask=True)
        
        # Get cat and num predictions
        predictions_cat, _ = predictions
        
        # cat, num feature index dict
        cat_features_indeces, _ = data_features_indeces_dec
        
        # Get cat and num targets
        cat_suffixes, _ = suffixes
        
        cat_suffixes_dict = {}
        # For suffix: map the feature name of the decoder output to the corresponding tensor using the index
        for feature_name, index in cat_features_indeces.items():
            cat_suffixes_dict[feature_name] = cat_suffixes[index]
            
        # Calculate the loss for activity categorical feature only
        cat_loss_dict = {}
        cat_loss_list = []
        activity_feature_name = self._select_activity_feature_name(cat_features_indeces, predictions_cat)
        mean_cat_pred = predictions_cat[f"{activity_feature_name}_mean"]
        var_cat_pred = predictions_cat.get(f"{activity_feature_name}_var")
        target_cat = cat_suffixes_dict[activity_feature_name]

        loss_cat = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred,
                                                                pred_logvars=var_cat_pred,
                                                                T=30,
                                                                targets=target_cat.long(),
                                                                eos_paddings=eos_paddings)
        cat_loss_dict[activity_feature_name] = loss_cat
        cat_loss_list.append(loss_cat)
        
        # List of categorical losses for 1 batch
        all_losses = cat_loss_dict
        
        weight_reg_enc, bias_reg_enc = self.model.encoder.regularizer()
        weight_reg_dec, bias_reg_dec = self.model.decoder.regularizer()
        
        weight_reg = weight_reg_enc + weight_reg_dec
        bias_reg = bias_reg_enc + bias_reg_dec

        # Zero gradients before optimization step
        self.optimizer.zero_grad()

        # Total mean loss
        stacked_tensor_losses = torch.stack(cat_loss_list)
        data_loss = stacked_tensor_losses.sum()
        loss = data_loss + self.regularization_term * (weight_reg.to(self.device) + bias_reg.to(self.device))

        # Decision-aware guard regularization
        if self.lambda_g > 0 and z_targets is not None:
            guard_loss = self.loss_obj.guard_KL_loss(pred_logits=mean_cat_pred,
                                                     guard_targets=z_targets,
                                                     guard_mask=z_mask,
                                                     eos_paddings=eos_paddings,
                                                     next_event_targets=target_cat.long(),
                                                     guard_confidence=c_values,
                                                     teacher_forcing_mask=tf_mask,
                                                     support_threshold=self.guard_support_threshold)
            loss = loss + self.lambda_g * guard_loss

        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
        # Model Optimization
        self.optimizer.step()
        
        return all_losses, loss

    def validation_epoch(self, val_dataloader, use_statics=False, use_zero_padd_masking=False, use_eos_padd_masking=False):
        """
        Validates the model on the validation set during training.
        """
        # Set model to evaluation mode
        self.model.eval()
        
        with torch.no_grad():
        
            cat_loss_dict_std = {}
            cat_loss_dict_unc = {}
            
            num_batches_per_epoch = 0.0
            
            for _, val_data in enumerate(val_dataloader): 
                batch = self._unpack_batch_common(val_data)
                cats = batch["cats"]
                nums = batch["nums"]
                eos_paddings = batch["eos_paddings"]
                zero_paddings = batch["zero_paddings"]
                cats_static = batch["cats_static"]
                nums_static = batch["nums_static"]

                if use_statics:
                    static_inputs = self._prepare_static_inputs(cats_static, nums_static)
                else:
                    static_inputs = None
                
                prefixes, suffixes = self._split_prefix_suffix(cats=cats, nums=nums, suffix_size=self.suffix_data_split_value)
                prefix_mask, eos_paddings_suffix = self._build_masks(eos_paddings=eos_paddings,
                                                                     zero_paddings=zero_paddings,
                                                                     suffix_size=self.suffix_data_split_value,
                                                                     use_zero_padd_masking=use_zero_padd_masking,
                                                                     use_eos_padd_masking=use_eos_padd_masking)

                # Model predictions:
                predictions, _, _, data_features_indeces_dec= self.model(prefixes=prefixes,
                                                                         suffixes=suffixes,
                                                                         teacher_forcing_ratio=self.teacher_forcing_ratio,
                                                                         static_inputs=static_inputs,
                                                                         prefix_mask=prefix_mask)
                predictions_cat, _ = predictions
                
                # Targets
                cat_features_indeces, _ = data_features_indeces_dec
                
                cat_suffixes, _ = suffixes
                
                cat_suffixes_dict = {}
                for feature_name, index in cat_features_indeces.items():
                    cat_suffixes_dict[feature_name] = cat_suffixes[index]

                activity_feature_name = self._select_activity_feature_name(cat_features_indeces, predictions_cat)
                mean_cat_pred = predictions_cat[f"{activity_feature_name}_mean"]
                var_cat_pred = predictions_cat.get(f"{activity_feature_name}_var")
                target_cat = cat_suffixes_dict[activity_feature_name]

                # Standard cross entropy
                cat_loss_std = self.loss_obj.standard_cross_entropy(pred_logits=mean_cat_pred,
                                                                    targets=target_cat.long(),
                                                                    eos_paddings=eos_paddings_suffix)
                # Uncertainty cross entropy
                cat_loss_unc = self.loss_obj.loss_attenuation_cross_entropy(pred_logits=mean_cat_pred,
                                                                            pred_logvars=var_cat_pred,
                                                                            T=30,
                                                                            targets=target_cat.long(),
                                                                            eos_paddings=eos_paddings_suffix)

                if activity_feature_name in cat_loss_dict_std:
                    # Add the current batch's loss to the cumulative loss
                    cat_loss_dict_std[activity_feature_name] += cat_loss_std
                    cat_loss_dict_unc[activity_feature_name] += cat_loss_unc
                else:
                    # Initialize the cumulative loss with the first batch's loss
                    cat_loss_dict_std[activity_feature_name] = cat_loss_std.clone()
                    cat_loss_dict_unc[activity_feature_name] = cat_loss_unc.clone()
                
                # Increase number of trained batches:
                num_batches_per_epoch += 1
                
            # Average losses over batches
            for feature_name in cat_loss_dict_std.keys():
                cat_loss_dict_std[feature_name] /= num_batches_per_epoch
                cat_loss_dict_unc[feature_name] /= num_batches_per_epoch

            # Sum all feature-wise losses to get total epoch losses
            val_epoch_loss_std = sum(cat_loss_dict_std.values()).item()
            val_epoch_loss_unc = sum(cat_loss_dict_unc.values()).item()
                
        # Set model back to train for gradient caluclation and optimization.
        self.model.train()
        
        return val_epoch_loss_std, val_epoch_loss_unc


#
#
# Training for camargo LSTM: 
#
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
                 lambda_g: float = 0.0,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'C_LSTM.pkl'):
        
        super().__init__(device=device,
                         model=model,
                         data_train=data_train,
                         data_val=data_val,
                         optimize_values=optimize_values,
                         save_model_n_th_epoch=save_model_n_th_epoch,
                         saving_path=saving_path)

        print("Device: ", device)
        self.loss_obj = loss_obj if loss_obj is not None else Loss()
        
        self.concept_name_id=concept_name_id
        self.eos_id = eos_id

        # Decision-aware guard regularization weight
        self.lambda_g = lambda_g

        # Select only prefix features configured in the C-LSTM model.
        self.prefix_cat_feature_indices = None
        self.prefix_num_feature_indices = None
        self._init_prefix_feature_indices()
        
        # Standard Optimization parameters
        print("Optimizer: ", self.optimizer)
        print("Scheduler: ", self.scheduler)
        print("Epochs: ", self.epochs)
        print("Mini baches: ", self.mini_batches)
        print("Shuffle batched dataset: ", self.shuffle)

    def _init_prefix_feature_indices(self):
        """
        Map configured model feature names to dataset tensor indices.
        """
        cat_categories, num_categories = self.data_train.all_categories
        cat_names_dataset = [cat[0] for cat in cat_categories]
        num_names_dataset = [num[0] for num in num_categories]

        model_feat = getattr(self.model, "model_feat", None)
        if model_feat is None:
            # Fallback to all dynamic features if model does not expose feature config.
            self.prefix_cat_feature_indices = list(range(len(cat_names_dataset)))
            self.prefix_num_feature_indices = list(range(len(num_names_dataset)))
            return

        model_cat_names, model_num_names = model_feat

        missing_cat = [name for name in model_cat_names if name not in cat_names_dataset]
        missing_num = [name for name in model_num_names if name not in num_names_dataset]
        if missing_cat or missing_num:
            raise ValueError("Configured model features are missing in dataset categories. "f"Missing categorical: {missing_cat}, missing numerical: {missing_num}.")

        self.prefix_cat_feature_indices = [cat_names_dataset.index(name) for name in model_cat_names]
        self.prefix_num_feature_indices = [num_names_dataset.index(name) for name in model_num_names]
        
    def _preprocess_batch(self, cats, nums, eos_paddings=None):
        """
        C-training is next-event prediction, so supervision length is fixed to S=1!

        Outputs:
        - prefixes
        - target_act
        - eos_next
        - V
        - valid_mask
        """
        
        if len(cats) == 0:
            return None, None, None, 0, None

        # set all traces to True
        valid_mask = torch.ones(cats[0].shape[0], dtype=torch.bool, device=cats[0].device)
        if self.eos_id is not None:
            # Keep old filtering behavior: allow at most one EOS in prefix and one in suffix.
            eos_counts = (cats[self.concept_name_id] == self.eos_id).sum(dim=1)
            # set all traces to false that contain more than two EOS token
            valid_mask = eos_counts <= 2

        # Count traces that are TRUE (128>=)
        V = int(valid_mask.sum().item())
        if V == 0:
            return None, None, None, 0, None

        batch_cats = [cat[valid_mask] for cat in cats]
        batch_nums = [num[valid_mask] for num in nums]

        # Prefix features must match the C-LSTM configuration.
        selected_cats = [batch_cats[i] for i in self.prefix_cat_feature_indices]
        selected_nums = [batch_nums[i] for i in self.prefix_num_feature_indices]

        prefixes_cat = [cat[:, :-1].to(self.device) for cat in selected_cats]
        prefixes_num = [num[:, :-1].to(self.device) for num in selected_nums]
        prefixes = [prefixes_cat, prefixes_num]

        # Next activity target is based on the activity tensor in full dataset ordering.
        target_act = batch_cats[self.concept_name_id][:, -1].to(self.device).long()

        eos_next = None
        if eos_paddings is not None:
            eos_next = eos_paddings[valid_mask][:, -1:].to(self.device)

        return prefixes, target_act, eos_next, V, valid_mask

    def train(self):
        """
        Run full training and validation loops.
        """
        self.model.train()
        
        train_losses = []
        val_losses = []
        
        # Validation dataloader
        val_dataloader = self._build_dataloader(self.data_val, num_workers=4)

        for epoch in tqdm(range(self.epochs)):
            self.model.train()

            # Train dataloader
            train_dataloader = self._build_dataloader(self.data_train, num_workers=4)
            
            total = 0
            num_batches = 0

            for _, train_cases in enumerate(train_dataloader):
                batch = self._unpack_batch_common(train_cases)
                cats = batch["cats"]
                nums = batch["nums"]
                eos_paddings = batch["eos_paddings"]

                z_targets_full = batch["z_targets"]
                z_mask_full = batch["z_mask"]
                c_values_full = batch["c_values"]

                # Get prefixes and next-activity targets (S=1).
                prefixes, target_act, eos_next, V, valid_mask = self._preprocess_batch(cats=cats, nums=nums, eos_paddings=eos_paddings)
                
                if V == 0:
                    continue
                
                # Forward pass: raw activity logits with dim [batch, activity_classes]
                a_logits = self.model(prefixes)
                
                # Compute losses
                # Activity: standard CE from loss.py (single-step sequence)
                pred_logits = a_logits.unsqueeze(0)  # [1, V, C]
                act_loss = self.loss_obj.standard_cross_entropy(pred_logits=pred_logits,
                                                                targets=target_act.unsqueeze(1),
                                                                eos_paddings=eos_next)

                loss = act_loss

                # Decision-aware guard regularization (last prefix event = position -2)
                if self.lambda_g > 0 and z_targets_full.shape[-1] > 0:
                    # Apply same valid_mask used for prefix/target filtering
                    gt = z_targets_full[valid_mask][:, -2, :].unsqueeze(1).to(self.device)  # [V, 1, C]
                    gm = z_mask_full[valid_mask][:, -2].unsqueeze(1).to(self.device)        # [V, 1]
                    gc = None
                    if c_values_full is not None:
                        gc = c_values_full[valid_mask][:, -2].unsqueeze(1).to(self.device)    # [V, 1]
                    
                    guard_loss = self.loss_obj.guard_KL_loss(pred_logits=pred_logits,
                                                             guard_targets=gt,
                                                             guard_mask=gm,
                                                             eos_paddings=eos_next,
                                                             next_event_targets=target_act.unsqueeze(1),
                                                             guard_confidence=gc,
                                                             support_threshold=self.guard_support_threshold)
                    loss = loss + self.lambda_g * guard_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
                # Mean loss over all samples in batch of size V 
                total += loss.item()
                num_batches += 1

            # Current learning rate
            current_lr = self._current_lr()
            
            # epoch averages train loss:
            epoch_loss = total / max(1, num_batches)            
            # Prints per Epoch:
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {current_lr}")
            tqdm.write(f"Training: Avg Attenuated Training Loss: {epoch_loss:.4f}")
            train_losses.append(epoch_loss)
            
            val_loss = self._validate(loader=val_dataloader)
            tqdm.write(f"Validation: Avg Validation Loss: {val_loss:.4f}")
            val_losses.append(val_loss)
            # Adjust the learning rate if necessary
            tqdm.write(f"Validation Loss for Scheduler: {val_loss:.4f}")
            
            # Adjust learning rate
            self._step_scheduler(val_loss)
            self._save_if_due(epoch)
                                 
        print("Training complete.")

        self._save_model()
        tqdm.write(f'Model saved to path: {self.saving_path}')

        return train_losses, val_losses

    def _validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for val_batch in loader:
                batch = self._unpack_batch_common(val_batch)
                cats = batch["cats"]
                nums = batch["nums"]
                eos_paddings = batch["eos_paddings"]

                prefixes, target_act, eos_next, V, _ = self._preprocess_batch(cats=cats,
                                                                              nums=nums,
                                                                              eos_paddings=eos_paddings)
                if V == 0:
                    continue

                a_logits = self.model(input=prefixes)

                pred_logits = a_logits.unsqueeze(0)  # [1, B, C]
                act_loss = self.loss_obj.standard_cross_entropy(pred_logits=pred_logits,
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
                 lambda_g: float = 0.0,
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

        # Decision-aware guard regularization weight
        self.lambda_g = lambda_g

        # Teacher forcing
        self.min_teacher_forcing_value = optimize_values.get("min_teacher_forcing_value", 0.0)
        self.max_teacher_forcing_value = optimize_values.get("max_teacher_forcing_value", 0.5)
        self.scheduled_sampling_epsilon_max = optimize_values.get("scheduled_sampling_epsilon_max", self.max_teacher_forcing_value)
        self.scheduled_sampling_k = optimize_values.get("scheduled_sampling_k", max(1.0, self.epochs / 10.0))

        # Gumbel-softmax temperature annealing (0.9 - 0, exponential)
        self.tau_start = optimize_values.get("tau_start", 0.9)
        self.tau_min = optimize_values.get("tau_min", 0.01)

        # GAN vs MLE switch
        self.use_gan = optimize_values.get("use_gan", True)
        self.beam_width = optimize_values.get("beam_width", 3)

        # Optimizers: G (encoder-decoder) and D (discriminator)
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

        # Feature projection: map dataset tensor indices -> model_feat indices (like CTraining)
        self.prefix_cat_feature_indices = None
        self.prefix_num_feature_indices = None
        self._init_prefix_feature_indices()

        print("Device: ", device)
        print("Mode: ", "GAN (Algorithm 1: MLMME)" if self.use_gan else "MLE-only")
        print("Epochs (iterations): ", self.epochs)
        print("Gumbel-softmax τ: ", f"{self.tau_start} → {self.tau_min} (exponential anneal)")
        print("Teacher forcing mode:", self.teacher_forcing_mode)
        if self.teacher_forcing_mode == "fixed":
            print("Fixed teacher forcing ratio:", self.fixed_teacher_forcing_ratio)
        else:
            print("Scheduled sampling ε:", f"0.0 -> {self.scheduled_sampling_epsilon_max} (inverse-sigmoid)")

    def _init_prefix_feature_indices(self):
        """
        Map configured model feature names to dataset tensor indices.
        """
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
            raise ValueError("Configured model features are missing in dataset categories. Missing categorical: {missing_cat}, missing numerical: {missing_num}.")

        self.prefix_cat_feature_indices = [cat_names_dataset.index(name) for name in model_cat_names]
        self.prefix_num_feature_indices = [num_names_dataset.index(name) for name in model_num_names]

    def _unpack_batch(self, batch):
        """
        Split batch into model-projected prefixes, activity targets, EOS mask, and guard data.
        """
        unpacked = self._unpack_batch_common(batch)
        cats = unpacked["cats"]
        nums = unpacked["nums"]
        eos_paddings = unpacked["eos_paddings"]
        z_targets_full = unpacked["z_targets"]
        z_mask_full = unpacked["z_mask"]
        c_values_full = unpacked["c_values"]

        # Split into prefix / suffix with fixed S (same as KTrainer)
        prefixes, suffixes = self._split_prefix_suffix(cats=cats,
                                                       nums=nums,
                                                       suffix_size=self.suffix_data_split_value)

        # Project prefix features to model_feat subset
        prefix_cats, prefix_nums = prefixes
        selected_cats = [prefix_cats[i] for i in self.prefix_cat_feature_indices]
        selected_nums = [prefix_nums[i] for i in self.prefix_num_feature_indices]
        prefixes = [selected_cats, selected_nums]

        # Activity suffix target (from full dataset concept_name_id)
        act_targets = suffixes[0][self.concept_name_id].long()

        eos_suffix = None if eos_paddings is None else eos_paddings[:, -self.suffix_data_split_value:].to(self.device)

        # Guard data aligned to decoder steps via the shared last-prefix/previous-event logic.
        z_suffix_targets, z_suffix_mask, c_suffix_values = self._extract_guard_suffix(z_targets_full,
                                                   z_mask_full,
                                                   self.suffix_data_split_value,
                                                   c_values=c_values_full)

        return prefixes, act_targets, eos_suffix, z_suffix_targets, z_suffix_mask, c_suffix_values

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
        """
        Adversarial training.
        """
        self.model.train()

        train_gen_losses = []
        train_disc_losses = []
        val_losses = []
        val_beam_token_acc = []

        val_loader = self._build_dataloader(self.data_val, num_workers=4)

        # Exponential annealing rate: τ_t = max(tau_min, tau_start * exp(-rate * t))
        if self.epochs > 1:
            anneal_rate = math.log(self.tau_start / self.tau_min) / (self.epochs - 1)
        else:
            anneal_rate = 0.0

        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            train_loader = self._build_dataloader(self.data_train, num_workers=4)

            # Exponential Gumbel-softmax temperature annealing (τ: 0.9 → ~0)
            tau = max(self.tau_min, self.tau_start * math.exp(-anneal_rate * epoch))

            # Select teacher forcing policy (fixed ratio or scheduled sampling).
            self.scheduled_sampling_epsilon, self.teacher_forcing_ratio = self._teacher_forcing_rates(
                step_index=epoch,
                epsilon_max=self.scheduled_sampling_epsilon_max,
                inverse_sigmoid_k=self.scheduled_sampling_k,
                min_teacher_forcing=self.min_teacher_forcing_value,
            )
            gen_loss_total = 0.0
            disc_loss_total = 0.0
            n_batches = 0

            for _, batch in enumerate(train_loader):
                prefixes, target_suffix_act, eos_suffix, z_suffix_targets, z_suffix_mask, c_suffix_values = self._unpack_batch(batch)

                if self.use_gan:
                    # Discriminator step (Algorithm 1, line 4): L(D;G) = -log(D(sig>k)) - log(1 - D(sig>k))
                    # Real suffix: exact one-hot activity sequence.
                    real_onehot = F.one_hot(target_suffix_act, self.model.output_size_act).float()

                    # Fake suffix: generator output with Gumbel-softmax (detached from G)
                    with torch.no_grad():
                        logits_d = self.model(prefixes=prefixes,
                                              target_suffix=target_suffix_act,
                                              teacher_forcing_ratio=self.teacher_forcing_ratio)  # [S, B, C]
                    
                    fake_gumbel_d = F.gumbel_softmax(logits_d.permute(1, 0, 2).detach(), tau=tau, hard=False, dim=-1)  # [B, S, C]

                    # Sequence-level discriminator logits from the final timestep.
                    d_real = self._sequence_discriminator_logits(real_onehot)
                    d_fake = self._sequence_discriminator_logits(fake_gumbel_d)

                    disc_loss_real = F.binary_cross_entropy_with_logits(d_real,
                                                                        torch.ones_like(d_real))
                    
                    disc_loss_fake = F.binary_cross_entropy_with_logits(d_fake,
                                                                        torch.zeros_like(d_fake))
                    disc_loss = disc_loss_real + disc_loss_fake

                    self.discriminator_optimizer.zero_grad()
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.discriminator.parameters(), max_norm=1.0)
                    self.discriminator_optimizer.step()
                else:
                    disc_loss = torch.tensor(0.0, device=self.device)

                # Generator step (Algorithm 1, line 5): Update θg by minimizing L(G;D) + L_supervised
                self.generator_optimizer.zero_grad()

                # Single forward pass for both losses
                logits_g, tf_mask = self.model(prefixes=prefixes,
                                               target_suffix=target_suffix_act,
                                               teacher_forcing_ratio=self.teacher_forcing_ratio,
                                               return_teacher_forcing_mask=True)  # [S, B, C], [B, S]

                # L_supervised: standard cross-entropy on activity suffixes
                loss_supervised = self._masked_activity_loss(logits_g, target_suffix_act, eos_mask=eos_suffix)

                if self.use_gan:
                    # Non-saturating generator loss on the discriminator's sequence-level logit.
                    fake_gumbel_g = F.gumbel_softmax(logits_g.permute(1, 0, 2), tau=tau, hard=False, dim=-1)  # [B, S, C]
                    d_fake_g = self._sequence_discriminator_logits(fake_gumbel_g)
                    adv_loss = F.binary_cross_entropy_with_logits(d_fake_g, torch.ones_like(d_fake_g))
                    gen_loss = adv_loss + loss_supervised
                else:
                    gen_loss = loss_supervised

                # Decision-aware guard regularization
                if self.lambda_g > 0 and z_suffix_targets is not None:
                    guard_loss = self.loss_obj.guard_KL_loss(pred_logits=logits_g,
                                                             guard_targets=z_suffix_targets,
                                                             guard_mask=z_suffix_mask,
                                                             eos_paddings=eos_suffix,
                                                             next_event_targets=target_suffix_act,
                                                             guard_confidence=c_suffix_values,
                                                             teacher_forcing_mask=tf_mask,
                                                             support_threshold=self.guard_support_threshold)
                    gen_loss = gen_loss + self.lambda_g * guard_loss

                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._generator_parameters(), max_norm=1.0)
                self.generator_optimizer.step()

                gen_loss_total += gen_loss.item()
                disc_loss_total += disc_loss.item()
                n_batches += 1

            epoch_gen_loss = gen_loss_total / max(1, n_batches)
            epoch_disc_loss = disc_loss_total / max(1, n_batches)
            train_gen_losses.append(epoch_gen_loss)
            train_disc_losses.append(epoch_disc_loss)

            # Validation
            val_loss= self._validate(val_loader)
            val_losses.append(val_loss)

            # Logging
            current_lr = self._current_lr()
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], LR: {current_lr}, "
                       f"τ: {tau:.4f}, TF: {self.teacher_forcing_ratio:.4f}, ε: {self.scheduled_sampling_epsilon:.4f}, "
                       f"Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}")

            # Scheduler
            if self.generator_scheduler is not None:
                self.generator_scheduler.step(val_loss)
            if self.use_gan and self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step(val_loss)

            self._save_if_due(epoch)

        print("Training complete.")
        self._save_model()
        tqdm.write(f'Model saved to path: {self.saving_path}')

        return train_gen_losses, train_disc_losses, val_losses

    def _validate(self, loader):
        """
        Validate using CE loss and beam-search token accuracy.
        """
        self.model.eval()
        val_loss_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                prefixes, target_suffix_act, eos_suffix, _, _, _ = self._unpack_batch(batch)

                logits = self.model(prefixes=prefixes, target_suffix=target_suffix_act, teacher_forcing_ratio=0.0)
                loss = self._masked_activity_loss(logits, target_suffix_act, eos_mask=eos_suffix)

                val_loss_total += loss.item()
                n_batches += 1

        self.model.train()

        mean_val_loss = val_loss_total / max(1, n_batches)
        return mean_val_loss