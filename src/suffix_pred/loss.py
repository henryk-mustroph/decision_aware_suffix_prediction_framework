"""
Loss functions for categorical event label sequence training.

Includes standard and uncertainty-attenuated cross entropy variants, and a decision-aware semantic loss (set-membership constraint over the support of a decision-model distribution).
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
        if eos_paddings is None:
            return torch.mean(loss_matrix)

        if loss_matrix.shape != eos_paddings.shape:
            raise ValueError("loss and eos paddings have wrong shape!")

        # Use torch.where to avoid NaN propagation from padded regions.
        L = torch.where(eos_paddings.bool(), loss_matrix, torch.tensor(0.0, device=loss_matrix.device))
        total_valid_tokens = torch.sum(eos_paddings)
        return torch.sum(L) / (total_valid_tokens + 1e-8)
    
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

    def gaussian_nll_loss(self,
                          pred_means,
                          pred_logvars,
                          targets,
                          eos_paddings):
        """
        Gaussian negative log-likelihood loss for autoregressive numerical event attribute prediction.

        Inputs:
        - pred_means: predicted means: dim: seq_len x batch x 1
        - pred_logvars: predicted log-variances: dim: seq_len x batch x 1
        - targets: target scalar values: dim: batch x seq_len
        - eos_paddings: Optional EOS mask (batch x seq_len).

        Outputs:
        - L: Scalar loss value.
        """
        # Numerical heads output [seq_len, batch, 1] -> squeeze last dim.
        if pred_means.dim() == 3 and pred_means.shape[-1] == 1:
            pred_means = pred_means.squeeze(-1)
        if pred_logvars.dim() == 3 and pred_logvars.shape[-1] == 1:
            pred_logvars = pred_logvars.squeeze(-1)

        # Align prediction shape [seq_len, batch] with targets [batch, seq_len].
        pred_means = pred_means.permute(1, 0)
        pred_logvars = pred_logvars.permute(1, 0)

        # Clamp log-variance for numerical stability.
        pred_logvars = torch.clamp(pred_logvars, min=-6.0, max=6.0)
        precision = torch.exp(-pred_logvars)

        # Per-step Gaussian NLL (ignoring constant log(2*pi) term).
        per_step_loss = 0.5 * (precision * (targets - pred_means) ** 2 + pred_logvars)

        return self._reduce_loss(per_step_loss, eos_paddings)

    def semantic_loss(self,
                      pred_logits,
                      guard_targets,
                      guard_mask,
                      tau,
                      eos_paddings=None,
                      teacher_forcing_mask=None,
                      gt_targets=None,
                      gt_in_support_only=False):
        """
        Decision-aware semantic loss (L_sem).
        Hard set-membership constraint at each decision-labeled event: the
        predictor must place sufficient probability mass on the tau-support
        of the decision-model distribution:
            S^(i)_{k+s} := { a in A | z^(i)_{k+s}(a) >= tau }.

        Per-step loss:
            l_sem(i, s) = 1[p != bot] * 1[S != empty]
                          * ( - log sum_{a in S} p_theta(a) )

        Aggregated over a mini-batch:
            L_sem = (1 / N_B) * sum_{i in B} sum_{s=0}^{S^(i)-1} l_sem(i, s)
        where N_B is the count of decoder steps satisfying both indicators.

        Inputs:
        - pred_logits: Predicted next-event logits: dim seq_len x batch x classes.
        - guard_targets: Soft decision-model distributions z_i: dim batch x seq_len x classes.
        - guard_mask: Indicator 1[p_i != bot]: dim batch x seq_len.
        - tau: Decision-support threshold in (0, 1].
        - eos_paddings: Optional EOS mask (batch x seq_len).
        - teacher_forcing_mask: Optional mask restricting the loss to decoder steps that consumed a ground-truth previous event. Required because the offline labeling assumption only assigns decision labels to ground-truth events.
        - gt_targets: Optional ground-truth next-activity ids: dim batch x seq_len.
          Only consulted when ``gt_in_support_only`` is True.
        - gt_in_support_only: When True (and ``gt_targets`` is given), additionally
          restrict the loss to steps whose ground-truth next activity is itself in
          the tau-support, i.e. steps where the (learned, imperfect) decision model
          AGREES with the observed outcome. The semantic-loss formulation of
          Xu et al. (2018) assumes *exact* logical constraints that the true label
          always satisfies; a mined decision model is a *soft, noisy* constraint
          whose tau-support excludes the realised next activity on a large fraction
          of events (≈56% on Helpdesk). On those steps the unfiltered loss pulls
          probability mass away from the ground truth and fights the base
          cross-entropy, degrading suffix accuracy (DLS) while barely moving
          decision conformance. This gate keeps the symbolic signal only where the
          constraint is consistent with reality, which removes that label noise.
          It uses the ground-truth label, so it is a *training-time* denoising
          step only; inference-time decoding / conformance never see it (no leak).

        Outputs:
        - L_sem: Scalar semantic loss averaged over N_B. Returns 0 (with grad) when no eligible step exists.
        """
        if guard_targets is None or guard_targets.shape[-1] == 0:
            return pred_logits.sum() * 0.0

        # log p_theta(a): [S, B, C] -> [B, S, C]
        log_probs = F.log_softmax(pred_logits, dim=-1).permute(1, 0, 2)

        # tau-support: S = { a | z(a) >= tau }
        z = guard_targets.clamp_min(0.0)
        support = (z >= tau).to(log_probs.dtype)

        # 1[S != empty]
        nonempty_support = (support.sum(dim=-1) > 0).to(log_probs.dtype)

        # log( sum_{a in S} p_theta(a) ) via masked logsumexp.
        # A large negative log-mask zeroes contributions outside the support.
        log_mask = torch.where(support > 0,
                               torch.zeros_like(support),
                               torch.full_like(support, -1e30))
        log_mass = torch.logsumexp(log_probs + log_mask, dim=-1)  # [B, S]

        # Combine the two per-step indicators.
        indicator = guard_mask * nonempty_support

        if eos_paddings is not None:
            indicator = indicator * eos_paddings

        # Decision labels are only valid for ground-truth events along the
        # trace; restrict the loss to teacher-forced decoder steps.
        if teacher_forcing_mask is not None:
            indicator = indicator * teacher_forcing_mask

        # Decision-label denoising: keep only steps where the decision model's
        # tau-support contains the true next activity (see docstring).
        if gt_in_support_only and gt_targets is not None:
            gt_idx = gt_targets.long().unsqueeze(-1).clamp(min=0, max=support.shape[-1] - 1)
            gt_in_support = torch.gather(support, dim=-1, index=gt_idx).squeeze(-1)  # [B, S]
            indicator = indicator * gt_in_support

        per_step_loss = -log_mass * indicator
        N_B = indicator.sum().clamp(min=1e-8)

        return per_step_loss.sum() / N_B
