"""
Comprehensive efficient auto-regressive training for categorical activity-sequence prediction.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
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

        self.save_model_n_th_epoch = save_model_n_th_epoch
        self.saving_path = saving_path

    def _build_dataloader(self, dataset, num_workers=0):
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
        """
        Normalized batch unpacking.

        Required format:
        - 8-tuple full dataset format: (_, cats, nums, eos, zero, cats_static, nums_static, _)
        """
        if len(batch) == 8:
            _, cats, nums, eos_paddings, zero_paddings, cats_static, nums_static, _ = batch
        else:
            raise ValueError(
                f"Unsupported batch format with len={len(batch)}. "
                "Expected full 8-tuple: (_, cats, nums, eos, zero, cats_static, nums_static, _)."
            )

        return {
            "cats": cats,
            "nums": nums,
            "eos_paddings": eos_paddings,
            "zero_paddings": zero_paddings,
            "cats_static": cats_static,
            "nums_static": nums_static,
        }

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

    def _split_prefix_suffix(self, cats, nums, suffix_size):
        prefixes_cat = [cat[:, :-suffix_size].to(self.device) for cat in cats]
        prefixes_num = [num[:, :-suffix_size].to(self.device) for num in nums]
        suffixes_cat = [cat[:, -suffix_size:].to(self.device) for cat in cats]
        suffixes_num = [num[:, -suffix_size:].to(self.device) for num in nums]
        return [prefixes_cat, prefixes_num], [suffixes_cat, suffixes_num]

    def _split_prefix_and_next_activity(self, cats, nums, concept_name_id):
        prefixes_cat = [cat[:, :-1].to(self.device) for cat in cats]
        prefixes_num = [num[:, :-1].to(self.device) for num in nums]
        target_act = cats[concept_name_id][:, -1].to(self.device).long()
        return [prefixes_cat, prefixes_num], target_act

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


class KTrainer(Trainer):
    def __init__(self,
                 device,
                 model,
                 data_train,
                 data_val,
                 loss_obj,
                 log_normal_loss_num_feature,
                 optimize_values,
                 suffix_data_split_value,
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'model.pkl',
                 random_suffix_split: bool = False):
        """
        Trainer class constructor.
        
        ARGS:
        - device: Device (GPU or CPU).
        - model: Model that is trained and validated.
        - data_train: Training data.
        - data_val: Validation data.
        - loss_obj: object for loss functions
        - log_normal_loss_num_feature: list of strings of num feaures that follow log normal distribution.
        
        - optimize_values:
            - regularization_term: L2 regularization for weights, biases, and dropout of stochastic model. 
            - optimizer: Optimization algorithm for training.
            - epochs: Epochs the model trains the full training dataset.
            - mini_batches: Batches the model get passed at once.
            - shuffles: Shuffle batches.
            - min teacher_forcing_ratio: Value [0,1) that is used to decide if predicted or next target event is used for next prediction by model.
            - max teacher_forcing_ratio: Value [0,1) that is used to decide if predicted or next target event is used for next prediction by model.
        
        - suffix_data_split_value: Number of last values of suffix events. 
        
        - save_model_n_th_epoch: int,
        - saving_path: str, default: 'model.pkl'
        - random_suffix_split: bool, default: False. If True, randomly splits prefix/suffix per batch.
        """

        # Standard Training parameters
        super().__init__(
            device=device,
            model=model,
            data_train=data_train,
            data_val=data_val,
            optimize_values=optimize_values,
            save_model_n_th_epoch=save_model_n_th_epoch,
            saving_path=saving_path,
        )

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
        
        # Events in sufffix: Dependent on data set
        self.suffix_data_split_value = suffix_data_split_value
        self.random_suffix_split = random_suffix_split

    def _select_activity_feature_name(self, cat_features_indeces, predictions_cat):
        """Pick the categorical activity feature key from decoder outputs."""
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
    
    def train_model(self, use_statics:Optional[bool]=False, use_zero_padd_masking:Optional[bool]=False, use_eos_padd_masking:Optional[bool]=False):
        """
        Seq2Seq Multi Task Learning algorithm with uncertainties.
        
        INPUTS:
        - use_statics:
        - use_zero_padd_masking:
        - use_eos_padd_masking:
        
        OUTPUTS:
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
        for epoch in range(self.epochs):#tqdm(range(self.epochs)):
            
            # Train dataloader
            train_dataloader = self._build_dataloader(self.data_train, num_workers=0)
            
            epoch_cat_loss = {}
            epoch_loss = 0.0
            num_batches_per_epoch = 0.0
            
            # Reduce Teacher forcing ratio dynamically (scheduled sampling)
            self.teacher_forcing_ratio = max(self.min_teacher_forcing_value, self.max_teacher_forcing_value - epoch / (self.epochs * 0.5))
            
            # Bacth Loop
            for i, train_data in enumerate(train_dataloader): 
                batch = self._unpack_batch_common(train_data)
                cats = batch["cats"]
                nums = batch["nums"]
                eos_paddings = batch["eos_paddings"]
                zero_paddings = batch["zero_paddings"]
                cats_static = batch["cats_static"]
                nums_static = batch["nums_static"]

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

                # Optimization (categorical only)
                cat_losses_dict, loss_value = self.train_epoch(prefixes=prefixes,
                                                               suffixes=suffixes,
                                                               eos_paddings=eos_paddings_suffix,
                                                               prefix_mask=prefix_mask,
                                                               static_inputs=static_inputs)
                
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
            tqdm.write(f"Epoch [{epoch+1}/{self.epochs}], Learning Rate: {current_lr}, Teacher forcing ratio: {self.teacher_forcing_ratio}")
            
            tqdm.write(f"Training: Avg Attenuated Training Loss: {epoch_loss_train:.4f}")
            
            train_attenuated_losses.append(epoch_loss_train)
            
            # Validation
            epoch_cat_loss_val_std, epoch_cat_loss_val_unc,\
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

    def train_epoch(self, prefixes, suffixes, eos_paddings, prefix_mask=None, static_inputs=None):
        """
        Train the model on batches.

        INPUTS:
        - prefixes: 
        - suffixes:
        - eos_paddings: Optional EOS mask tensor matching suffix shape.
        - prefix_mask: Zero-padding mask for the encoder prefix (batch x seq_len)
        - static_inputs 

        OUTPUTS:
        - all_losses:
        - loss: 
        """
        # predictions: List of two Dicts one for categorical (means and vars), one for numerical (means and vars): key: feature name + _mean or _var, value: tensor with dim: seq len x batch size x output feature size
        # data_features_indeces_dec: List of two Dicts one for categorical, one for numerical: key: feature name, value: index of tensor in data list
        predictions, _, _, data_features_indeces_dec= self.model(prefixes=prefixes,
                                                                 suffixes=suffixes,
                                                                 teacher_forcing_ratio=self.teacher_forcing_ratio,
                                                                 static_inputs=static_inputs,
                                                                 # prefix mask for the encoder
                                                                 prefix_mask=prefix_mask)
        
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
        loss.backward()
        
        # Gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
        # Model Optimization
        self.optimizer.step()
        
        return all_losses, loss

    def validation_epoch(self, val_dataloader, use_statics=False, use_zero_padd_masking=False, use_eos_padd_masking=False):
        """
        Validates the model on the validation set during training.

        INPUTS:
        - val_dataloader: Validation data for validating the model during training.
        
        OUTPUTS:
        - cat_loss_dict_std: Cat. event attributes standard loss
        - cat_loss_dict_unc: Cat. event attributes attenuated loss
        - val_epoch_loss_std: Total categorical standard loss
        - val_epoch_loss_unc: Total categorical attenuated loss
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
                
                prefixes, suffixes = self._split_prefix_suffix(
                    cats=cats,
                    nums=nums,
                    suffix_size=self.suffix_data_split_value,
                )
                prefix_mask, eos_paddings_suffix = self._build_masks(
                    eos_paddings=eos_paddings,
                    zero_paddings=zero_paddings,
                    suffix_size=self.suffix_data_split_value,
                    use_zero_padd_masking=use_zero_padd_masking,
                    use_eos_padd_masking=use_eos_padd_masking,
                )

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
        
        return cat_loss_dict_std, cat_loss_dict_unc, val_epoch_loss_std, val_epoch_loss_unc


# Training for: 
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
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'reimpl_model.pkl'
                 ):
        
        super().__init__(
            device=device,
            model=model,
            data_train=data_train,
            data_val=data_val,
            optimize_values=optimize_values,
            save_model_n_th_epoch=save_model_n_th_epoch,
            saving_path=saving_path,
        )

        print("Device: ", device)
        self.loss_obj = loss_obj if loss_obj is not None else Loss()
        
        self.concept_name_id=concept_name_id
        self.eos_id = eos_id
        
        # Standard Optimization parameters
        print("Optimizer: ", self.optimizer)
        print("Scheduler: ", self.scheduler)
        print("Epochs: ", self.epochs)
        print("Mini baches: ", self.mini_batches)
        print("Shuffle batched dataset: ", self.shuffle)
        
    def _preprocess_batch(self, batch):
        """
        Filters each sample so only those with max one EOS in the prefix and in the suffix remain.
        """
        unpacked = self._unpack_batch_common(batch)
        cats = unpacked["cats"]
        nums = unpacked["nums"]

        if len(cats) == 0:
            return None, None, 0

        valid_mask = torch.ones(cats[0].shape[0], dtype=torch.bool, device=cats[0].device)
        if self.eos_id is not None:
            eos_counts = (cats[self.concept_name_id] == self.eos_id).sum(dim=1)
            valid_mask = eos_counts <= 2

        V = int(valid_mask.sum().item())
        if V == 0:
            return None, None, 0

        batch_cats = [cat[valid_mask] for cat in cats]
        batch_nums = [num[valid_mask] for num in nums]
        prefixes, target_act = self._split_prefix_and_next_activity(
            cats=batch_cats,
            nums=batch_nums,
            concept_name_id=self.concept_name_id,
        )
        return prefixes, target_act, V

    def train(self):
        """
        Run full training and validation loops.
        Returns:
            train_losses: list of training epoch losses
            val_losses: list of validation epoch losses
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

            for i, train_cases in enumerate(train_dataloader):

                # Get the prefixes to process, the target case elapsed time, and the new batch size as zero tensors are skipped:
                prefixes, target_act, V = self._preprocess_batch(train_cases)
                
                if V == 0:
                    continue
                
                # Forward pass: Output dim:  a_probs: batch x activity classes
                a_probs = self.model(prefixes)
                
                # Compute losses
                # Activity: standard CE from loss.py (single-step sequence)
                pred_logits = torch.log(a_probs.clamp_min(1e-8)).unsqueeze(0)  # [1, B, C]
                act_loss = self.loss_obj.standard_cross_entropy(
                    pred_logits=pred_logits,
                    targets=target_act.unsqueeze(1),
                    eos_paddings=None,
                )

                loss = act_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
            
                # Mean loss over all samples in batch of size V 
                total += loss.item()
                num_batches += 1

            # epoch averages train loss:
            epoch_loss = total / num_batches
            
            # Current learning rate
            current_lr = self._current_lr()
            
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


    def _validate(self, loader):
        self.model.eval()

        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for val_batch in loader:
                
                prefixes, target_act, V = self._preprocess_batch(val_batch)
                
                if V == 0:
                    continue

                a_probs = self.model(input=prefixes)

                pred_logits = torch.log(a_probs.clamp_min(1e-8)).unsqueeze(0)  # [1, B, C]
                act_loss = self.loss_obj.standard_cross_entropy(
                    pred_logits=pred_logits,
                    targets=target_act.unsqueeze(1),
                    eos_paddings=None,
                )

                total_loss += act_loss.item()
                num_batches += 1

        return total_loss / num_batches

class TTraining(Trainer):
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
                 save_model_n_th_epoch: int = 0,
                 saving_path: str = 'taymouri_model.pkl'):

        super().__init__(
            device=device,
            model=model,
            data_train=data_train,
            data_val=data_val,
            optimize_values=optimize_values,
            save_model_n_th_epoch=save_model_n_th_epoch,
            saving_path=saving_path,
        )
        self.loss_obj = loss_obj if loss_obj is not None else Loss()

        self.suffix_data_split_value = suffix_data_split_value
        self.concept_name_id = concept_name_id
        self.eos_id = eos_id

        self.min_teacher_forcing_value = optimize_values.get("min_teacher_forcing_value", 0.0)
        self.max_teacher_forcing_value = optimize_values.get("max_teacher_forcing_value", 0.5)
        self.use_gan = optimize_values.get("use_gan", True)
        self.lambda_adv = optimize_values.get("lambda_adv", 0.1)
        self.beam_width = optimize_values.get("beam_width", 3)

        self.generator_optimizer = optimize_values.get("generator_optimizer", optimize_values.get("optimizer", None))
        self.discriminator_optimizer = None
        if self.use_gan:
            self.discriminator_optimizer = optimize_values.get("discriminator_optimizer", None)
            if self.discriminator_optimizer is None:
                self.discriminator_optimizer = torch.optim.AdamW(
                    list(self.model.discriminator_lstm.parameters()) + list(self.model.discriminator_head.parameters()),
                    lr=1e-3,
                )
        if self.generator_optimizer is None:
            raise ValueError("Provide `generator_optimizer` (or `optimizer`) in optimize_values.")

        self.generator_scheduler = optimize_values.get("generator_scheduler", optimize_values.get("scheduler", None))
        self.discriminator_scheduler = optimize_values.get("discriminator_scheduler", None) if self.use_gan else None

    def _unpack_batch(self, batch):
        unpacked = self._unpack_batch_common(batch)
        cats = unpacked["cats"]
        nums = unpacked["nums"]
        eos_paddings = unpacked["eos_paddings"]

        prefixes, suffixes = self._split_prefix_suffix(
            cats=cats,
            nums=nums,
            suffix_size=self.suffix_data_split_value,
        )
        act_targets = suffixes[0][self.concept_name_id].long()

        eos_suffix = None if eos_paddings is None else eos_paddings[:, -self.suffix_data_split_value:].to(self.device)

        return prefixes, act_targets, eos_suffix

    def _masked_activity_loss(self, logits, targets, eos_mask=None):
        # logits: [S, B, C], targets: [B, S]
        return self.loss_obj.standard_cross_entropy(
            pred_logits=logits,
            targets=targets,
            eos_paddings=eos_mask,
        )

    def train(self):
        self.model.train()

        train_gen_losses = []
        train_disc_losses = []
        val_losses = []
        val_beam_token_acc = []

        val_loader = self._build_dataloader(self.data_val, num_workers=4)

        for epoch in range(self.epochs):
            self.model.train()
            train_loader = self._build_dataloader(self.data_train, num_workers=4)

            self.teacher_forcing_ratio = max(
                self.min_teacher_forcing_value,
                self.max_teacher_forcing_value - epoch / max(1, (self.epochs * 0.5)),
            )

            gen_loss_total = 0.0
            disc_loss_total = 0.0
            n_batches = 0

            for i, batch in enumerate(train_loader):
                prefixes, target_suffix_act, eos_suffix = self._unpack_batch(batch)
                batch_size = target_suffix_act.shape[0]
                disc_loss = torch.tensor(0.0, device=self.device)

                if self.use_gan:
                    # 1) Discriminator step
                    with torch.no_grad():
                        fake_suffix_ids = self.model.sample_activity_ids(prefixes=prefixes, max_len=target_suffix_act.shape[1])

                    real_prob = self.model.discriminate(prefixes, target_suffix_act)
                    fake_prob = self.model.discriminate(prefixes, fake_suffix_ids)

                    real_labels = torch.ones(batch_size, device=self.device)
                    fake_labels = torch.zeros(batch_size, device=self.device)

                    disc_loss = F.binary_cross_entropy(real_prob, real_labels) + F.binary_cross_entropy(fake_prob, fake_labels)

                    self.discriminator_optimizer.zero_grad()
                    disc_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.discriminator_lstm.parameters()) + list(self.model.discriminator_head.parameters()),
                        max_norm=1.0,
                    )
                    self.discriminator_optimizer.step()
                else:
                    real_labels = torch.ones(batch_size, device=self.device)

                # 2) Generator step
                logits = self.model(
                    prefixes=prefixes,
                    target_suffix=target_suffix_act,
                    teacher_forcing_ratio=self.teacher_forcing_ratio,
                )
                loss_ce = self._masked_activity_loss(logits, target_suffix_act, eos_mask=eos_suffix)

                if self.use_gan:
                    fake_probs_for_g = torch.softmax(logits, dim=-1).permute(1, 0, 2)  # [B, S, C]
                    d_fake_for_g = self.model.discriminate(prefixes, fake_probs_for_g)
                    adv_loss_g = F.binary_cross_entropy(d_fake_for_g, real_labels)
                    gen_loss = loss_ce + self.lambda_adv * adv_loss_g
                else:
                    gen_loss = loss_ce

                self.generator_optimizer.zero_grad()
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.generator_optimizer.step()

                gen_loss_total += gen_loss.item()
                disc_loss_total += disc_loss.item()
                n_batches += 1

            epoch_gen_loss = gen_loss_total / max(1, n_batches)
            epoch_disc_loss = disc_loss_total / max(1, n_batches)
            train_gen_losses.append(epoch_gen_loss)
            train_disc_losses.append(epoch_disc_loss)

            val_loss, val_acc = self._validate_with_beam(val_loader)
            val_losses.append(val_loss)
            val_beam_token_acc.append(val_acc)

            tqdm.write(
                f"Epoch [{epoch+1}/{self.epochs}] - Mode: {'GAN' if self.use_gan else 'ED'}, "
                f"Gen Loss: {epoch_gen_loss:.4f}, Disc Loss: {epoch_disc_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Beam Token Acc: {val_acc:.4f}, TF Ratio: {self.teacher_forcing_ratio:.4f}"
            )

            if self.generator_scheduler is not None:
                self.generator_scheduler.step(val_loss)
            if self.use_gan and self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step(val_loss)

            self._save_if_due(epoch)

        self._save_model()
        return train_gen_losses, train_disc_losses, val_losses, val_beam_token_acc

    def _validate_with_beam(self, loader):
        self.model.eval()
        val_loss_total = 0.0
        token_correct = 0.0
        token_total = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in loader:
                prefixes, target_suffix_act, eos_suffix = self._unpack_batch(batch)

                logits = self.model(prefixes=prefixes, target_suffix=target_suffix_act, teacher_forcing_ratio=0.0)
                loss = self._masked_activity_loss(logits, target_suffix_act, eos_mask=eos_suffix)

                decoded = self.model.beam_search(
                    prefixes=prefixes,
                    beam_width=self.beam_width,
                    max_len=target_suffix_act.shape[1],
                    eos_id=self.eos_id,
                )

                if eos_suffix is None:
                    token_correct += (decoded == target_suffix_act).sum().item()
                    token_total += target_suffix_act.numel()
                else:
                    valid = eos_suffix.bool()
                    token_correct += ((decoded == target_suffix_act) & valid).sum().item()
                    token_total += valid.sum().item()

                val_loss_total += loss.item()
                n_batches += 1

        self.model.train()

        mean_val_loss = val_loss_total / max(1, n_batches)
        beam_token_acc = token_correct / max(1.0, token_total)
        return mean_val_loss, beam_token_acc