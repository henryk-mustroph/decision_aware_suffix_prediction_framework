"""
Loss functions for categorical event label sequence training.

Includes standard and uncertainty-attenuated cross entropy variants, and a decision-aware guard cross-entropy for regularization.
"""

# performance imports for torch: torch kernel uses one core only.
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1" 
import torch
import torch.nn.functional as F

class Loss:
    def __init__(self):
        pass
    
    def _reduce_loss(self, loss_matrix, eos_paddings):
        # normal loss reduction
        if eos_paddings is None:
            return torch.mean(loss_matrix)
        
        # eos padd masking loss reduction
        else:
            if loss_matrix.shape == eos_paddings.shape:
                # Mask the loss matrix: Use torch.where to avoid NaN propagation from padded regions
                L = torch.where(eos_paddings.bool(), loss_matrix, torch.tensor(0.0, device=loss_matrix.device))
            
                # Normalize loss per active timestep
                total_valid_tokens = torch.sum(eos_paddings)
                # Sum loss over all tokens and divide by total count
                return torch.sum(L) / (total_valid_tokens + 1e-8)
            else:
                return ValueError("loss and eos paddings have wrong shape!")
    
    def standard_cross_entropy(self, pred_logits, targets, eos_paddings):
        """
        Standard Cross Entropy loss.
      
        Inputs:
        - pred_logits: Predicted logit values for N events: dim: seq len x batch x labels (logit value for each label)
        - targets: Target class indices for N events: dim: batch x seq len
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        Outputs:
        - L: Global loss value for categorical event attributes: Tensor (float)
        """
        # Cross Entropy Loss
        CEL = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Change the shape of the prediction to: shape: batch_size x num_classes x seq len
        pred_logits = pred_logits.permute(1,2,0)
        L = CEL(input=pred_logits, target=targets)
        
        L = self._reduce_loss(L, eos_paddings)
        
        return L
    
    def loss_attenuation_cross_entropy(self, pred_logits, pred_logvars, T, targets, eos_paddings):
        """
        Loss attenuation cross entropy: Combined Epistemic and Aleatoric Uncertainty.
          
        Inputs:
        - pred_logits: Predicted logit values for N events: dim: seq_len x batch x classes
        - pred_logvars: Predicted log variances per logit value for N events: dim: seq len x batch x classes
        - T: T gaussian distributed random epsilon value generations.
        - targets: Target class indices for N events: dim: batch x  seq len
        - eos_paddings: Optional EOS mask (batch x seq len). Required when EOS masking is enabled.
        
        Outputs:
        - L: Global loss value for categorical event attributes: Tensor (float)
        """
            
        # Clamp the predicted log-variance to avoid collapse/instability.
        # Keeps std in [exp(-3)=0.05, exp(3)=20] since std=exp(0.5*logvar).
        min_logvariance = torch.tensor(-6.0, device=pred_logvars.device)
        max_logvariance = torch.tensor(6.0, device=pred_logvars.device)
        pred_logvars = torch.clamp(pred_logvars, min=min_logvariance, max=max_logvariance)

        # Cross Entropy Loss
        CEL = torch.nn.CrossEntropyLoss(reduction='none')
        
        # Get standard deviation
        variance = torch.exp(pred_logvars)
        std = torch.sqrt(variance)
        
        L = 0
        # T monte carlo iterations for approx. gaussian distribution
        for _ in range(T):
            # epsilon_t: Generate a random matrix to distribute the standard deviations
            noise = torch.randn_like(pred_logits)    
            pred_logits_std_noise = pred_logits + std * noise
            # Change the shape of the prediction to: shape: batch_size x num_classes x seq len
            pred_logits_std_noise = pred_logits_std_noise.permute(1,2,0)
            # CEL of gaussian distributed unaries and target
            ce_loss = CEL(input=pred_logits_std_noise, target=targets)
            L += ce_loss
        L = (1/T) * L
        
        L = self._reduce_loss(L, eos_paddings)
          
        return L

    # correcte
    def guard_KL_loss(self, 
                      pred_logits,
                      guard_targets,
                      guard_mask,
                      eos_paddings=None,
                      next_event_targets=None,
                      guard_confidence=None,
                      teacher_forcing_mask=None,
                      support_threshold=0.0):
        """
        Decision-aware guard KL loss (L_guard).
        Computes a weighted KL divergence between the thresholded and renormalized decision-model distribution q and the predicted next-event-label distribution.

        Inputs:
        - pred_logits: Predicted logit values: dim: seq_len x batch x classes
        - guard_targets: Soft target distributions from the decision model: dim: batch x seq_len x classes.  z_i(a) for each event and label.
        - guard_mask: Binary indicator for decision-labeled events: dim: batch x seq_len.  1 where z_i != bot, 0 otherwise.
        - eos_paddings: Optional EOS mask (batch x seq_len).
        - next_event_targets: Ground-truth next-event labels (batch x seq_len). Used to compute correctness weights w = z(a*).
        - guard_confidence: Optional confidence c_i with dim (batch x seq_len).
        - support_threshold: Decision-support threshold tau.

        Outputs:
        - L_guard: Scalar guard loss, averaged over effective weight sum. Returns 0 (with grad) if no regularized steps exist.
        """
        # no soft next-event labels given:
        if guard_targets is None or guard_targets.shape[-1] == 0:
            return pred_logits.sum() * 0.0

        # log p_theta(a): [S, B, C] -> [B, S, C]
        log_probs = F.log_softmax(pred_logits, dim=-1).permute(1, 0, 2)

        # Build support set S := {a | z(a) >= tau}, then renormalize to q.
        z = guard_targets.clamp_min(0.0)
        support = (z >= support_threshold).to(z.dtype)
        q_num = z * support
        q_den = q_num.sum(dim=-1, keepdim=True)
        valid_support = (q_den.squeeze(-1) > 0).to(z.dtype)
        q = q_num / q_den.clamp(min=1e-8)

        # KL(q || p) = sum_a q(a) * (log q(a) - log p(a))
        per_step_kl = (q * (torch.log(q.clamp(min=1e-8)) - log_probs)).sum(dim=-1)

        # Base weight: decision-labeled positions and non-empty support.
        weight = guard_mask * valid_support

        # Correctness weight w = z(a*), where a* is the ground-truth next label.
        if next_event_targets is not None:
            a_star = next_event_targets.long().unsqueeze(-1)
            support_star = torch.gather(support, dim=-1, index=a_star).squeeze(-1)
            w = torch.gather(z, dim=-1, index=a_star).squeeze(-1) * support_star
            weight = weight * w

        # Confidence weight c_i (defaults to 1 when not provided).
        if guard_confidence is not None:
            weight = weight * guard_confidence

        # Optional EOS masking.
        if eos_paddings is not None:
            weight = weight * eos_paddings

        # Optional teacher-forcing masking: regularize only decoder steps
        # that consumed ground-truth next-event inputs.
        if teacher_forcing_mask is not None:
            weight = weight * teacher_forcing_mask

        weighted = per_step_kl * weight

        # Normalize by effective batch weight sum.
        W = weight.sum().clamp(min=1e-8)
        
        return weighted.sum() / W
