"""
Unified training entry point.

Each architecture is trained by its OWN routine (the three models are
genuinely different): 
- UED is an attenuated-CE encoder-decoder with scheduled sampling;
- FS is a single-step next-event predictor (fully shared LSTM);
- GAN is a generative adversarial LSTM with Gumbel-softmax and separate G/D optimizers. We do NOT share UED's logic across them.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

from typing import Tuple

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .configs import ExperimentConfig, ExperimentPaths, resolve_paths

#helpers
def _load_datasets(paths: ExperimentPaths):
    train_set = torch.load(str(paths.train_dataset), weights_only=False)
    val_set = torch.load(str(paths.val_dataset), weights_only=False)
    return train_set, val_set

def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _concept_info(dataset, concept_name: str) -> Tuple[int, int, int]:
    """
    Return (concept_feature_idx, output_size_act, eos_id).
    """
    cats = dataset.all_categories[0]
    idx = next(i for i, c in enumerate(cats) if c[0] == concept_name)
    output_size_act = cats[idx][1]
    eos_id = cats[idx][2].get("EOS")
    if eos_id is None:
        raise ValueError(f"No EOS id in activity label mapping for {concept_name}")
    return idx, output_size_act, eos_id


def _select_features(cfg: ExperimentConfig):
    """
    Look up this architecture's explicit input/output feature lists for the
    current dataset and return (enc_feat, dec_feat, static_enc_feat, use_statics)
    in the [categorical, numeric] form the models expect: the model inputs map to
    enc_feat, the model outputs map to dec_feat. The concrete columns live in
    ``DatasetConfig.model_features[model_key]`` (see configs.ModelFeatures).
    UED uses all four (encoder-decoder + static path). FS (single LSTM) and GAN
    (encoder-decoder generator) take only enc_feat and derive their activity
    output size from it, so they ignore dec_feat / static_enc_feat.
    """
    spec = cfg.dataset.model_features.get(cfg.model.key)
    if spec is None:
        raise KeyError(
            f"No ModelFeatures for model '{cfg.model.key}' on dataset "
            f"'{cfg.dataset.key}' (configs.DatasetConfig.model_features).")
    enc_feat = [list(spec.input_cat), list(spec.input_num)]
    dec_feat = [list(spec.output_cat), list(spec.output_num)]
    static_enc_feat = [list(spec.static_cat), list(spec.static_num)]
    return enc_feat, dec_feat, static_enc_feat, spec.use_statics


def _build_optimizer(kind: str, params, lr: float, weight_decay: float = 0.0):
    kind = kind.lower()
    # optimizer for non-GAN model
    if kind == "adam":
        return torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    # Optimizer for GAN model
    if kind == "rmsprop":
        return torch.optim.RMSprop(params=params, lr=lr)
    raise ValueError(f"Unknown optimizer '{kind}'")

def _warm_start_from_clean(model, cfg: ExperimentConfig, root) -> bool:
    """
    Load weights from the clean checkpoint into `model` when fine-tuning the
    decision variant. Returns True if the warm-start succeeded.

    This is the primary fix for the decision-aware training conformance collapse:
    starting from a converged clean model and fine-tuning with a reduced LR
    prevents the semantic loss from destroying the learned activity distributions.
    """
    from .configs import make_experiment, resolve_paths as _rp
    clean_cfg = make_experiment(cfg.dataset.key, cfg.model.key, "clean")
    clean_paths = _rp(clean_cfg, root=root)
    ckpt = clean_paths.model_checkpoint
    if not ckpt.exists():
        print(f"  [warm-start] clean checkpoint not found at {ckpt.name}; training from scratch")
        return False
    try:
        data = torch.load(str(ckpt), weights_only=False, map_location="cpu")
        model.load_state_dict(data["model_state_dict"])
        print(f"  [warm-start] loaded clean weights from {ckpt.name}")
        return True
    except Exception as e:
        print(f"  [warm-start] load failed ({e}); training from scratch")
        return False

def _resolve_lr_epochs(cfg: ExperimentConfig, model, root) -> tuple:
    """Resolve (lr, epochs, warm_started) for this run.

    Clean variants use the ModelConfig base lr/epochs. Decision-aware variants,
    when warm-start is enabled and a clean checkpoint exists, load the clean
    weights into `model` (mutated in place) and switch to the fine-tune schedule
    defined entirely in ModelConfig (warm_start_lr_factor, fine_tune_epochs).
    ``warm_started`` reports whether the clean weights were loaded, so cold-start
    initialisation (e.g. the GAN) can be skipped on a successful warm-start.
    """
    m = cfg.model
    lr, epochs = m.learning_rate, m.epochs
    warm_started = False
    if m.warm_start and cfg.variant.uses_decision_loss and _warm_start_from_clean(model, cfg, root):
        lr = lr * float(m.warm_start_lr_factor)
        epochs = int(m.fine_tune_epochs) if m.fine_tune_epochs is not None else max(10, m.epochs // 5)
        warm_started = True
        print(f"  [warm-start] fine-tuning LR = {lr:.2e}, epochs = {epochs}")
    return lr, epochs, warm_started

# Dispatch
def train(cfg: ExperimentConfig, *, root=None):
    """Dispatch to the per-architecture trainer; returns its history tuple."""
    # Decision-aware training optimises the semantic loss against the decision
    # model, so the architecture must be able to predict the decision attributes.
    if cfg.variant.uses_decision_loss:
        from .configs import require_predicted_decision_attrs
        require_predicted_decision_attrs(cfg.dataset, cfg.model.key)

    paths = resolve_paths(cfg, root=root)
    paths.model_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    lambda_sem = cfg.model.lambda_sem if cfg.variant.uses_decision_loss else 0.0

    if cfg.model.key == "UED":
        return _train_ued(cfg, paths, lambda_sem, root=root)
    
    if cfg.model.key == "FS":
        return _train_c(cfg, paths, lambda_sem, root=root)
    
    if cfg.model.key == "GAN":
        return _train_t(cfg, paths, lambda_sem, root=root)
    raise NotImplementedError(f"Unknown model '{cfg.model.key}'")

# U-ED-LSTM: get values from config
def _train_ued(cfg: ExperimentConfig, paths: ExperimentPaths, lambda_sem: float, root=None):
    from suffix_pred.models.K_UED_LSTM import DropoutUncertaintyEncoderDecoderLSTM
    from suffix_pred.loss import Loss
    from suffix_pred.train import UEDTrainer

    m, ex = cfg.model, cfg.model.extra
    train_set, val_set = _load_datasets(paths)
    seq_len_pred = train_set.min_suffix_size

    enc_feat, dec_feat, static_enc_feat, use_statics = _select_features(cfg)

    model = DropoutUncertaintyEncoderDecoderLSTM(data_set_categories=train_set.all_categories,
                                                 # features for encoder: prefic only
                                                 enc_feat=enc_feat,
                                                 # features for decoder, last prefix, all predicted or target when teacher forcig for seq. train.
                                                 dec_feat=dec_feat,
                                                 # categories
                                                 static_data_set_categories=train_set.all_static_categories,
                                                 # static features to be encoded
                                                 static_enc_feat=static_enc_feat,
                                                 # sequence length for training
                                                 seq_len_pred=seq_len_pred,
                                                 # hyperparams
                                                 hidden_size=m.hidden_size,
                                                 num_layers=m.num_layers,
                                                 dropout=m.dropout)

    lr, epochs, _ = _resolve_lr_epochs(cfg, model, root)

    optimizer = _build_optimizer(ex.get("optimizer", "adam"), model.parameters(), lr, ex.get("weight_decay", 0.0))

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=15, min_lr=1e-8)

    optimize_values = {"regularization_term": ex.get("regularization_term", 1e-4),
                       "optimizer": optimizer,
                       "scheduler": scheduler,
                       "epochs": epochs,
                       "mini_batches": m.batch_size,
                       "shuffle": True,
                       "min_teacher_forcing_value": ex.get("min_teacher_forcing_value", 0.0),
                       "max_teacher_forcing_value": ex.get("max_teacher_forcing_value", 1.0),
                       "teacher_forcing_mode": ex.get("teacher_forcing_mode", "scheduled"),
                       "tau": m.tau,
                       "sem_gate_gt_in_support": ex.get("sem_gate_gt_in_support", True)}

    trainer = UEDTrainer(device=_device(),
                         model=model,
                         data_train=train_set,
                         data_val=val_set,
                         loss_obj=Loss(),
                         optimize_values=optimize_values,
                         suffix_data_split_value=seq_len_pred,
                         lambda_sem=lambda_sem,
                         save_model_n_th_epoch=1,
                         saving_path=str(paths.model_checkpoint))
    
    return trainer.train_model(use_statics=use_statics,
                               use_zero_padd_masking=True,
                               use_eos_padd_masking=True)

# FS-LSTM (next-event)
def _train_c(cfg: ExperimentConfig, paths: ExperimentPaths, lambda_sem: float, root=None):
    from suffix_pred.models.FS_LSTM import FullShared_Join_LSTM
    from suffix_pred.loss import Loss
    from suffix_pred.train import CTraining

    m, ex = cfg.model, cfg.model.extra
    train_set, val_set = _load_datasets(paths)

    enc_feat, _, _, _ = _select_features(cfg)
    concept_idx, output_size_act, eos_id = _concept_info(train_set, cfg.dataset.concept_name)

    model = FullShared_Join_LSTM(data_set_categories=train_set.all_categories,
                                 model_feat=enc_feat,
                                 # hyperparams
                                 hidden_size=m.hidden_size,
                                 num_layers=m.num_layers,
                                 # train and output size
                                 input_size=ex.get("input_size", 1),
                                 output_size_act=output_size_act)

    lr, epochs, _ = _resolve_lr_epochs(cfg, model, root)

    optimizer = _build_optimizer(ex.get("optimizer", "adam"), model.parameters(), lr, ex.get("weight_decay", 0.0))

    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=15, min_lr=1e-8)

    optimize_values = {"optimizer": optimizer,
                       "scheduler": scheduler,
                       "epochs": epochs,
                       "mini_batches": m.batch_size,
                       "shuffle": True,
                       "tau": m.tau,
                       "sem_gate_gt_in_support": ex.get("sem_gate_gt_in_support", True)}

    trainer = CTraining(device=_device(),
                        model=model,
                        data_train=train_set,
                        data_val=val_set,
                        optimize_values=optimize_values,
                        concept_name_id=concept_idx,
                        eos_id=eos_id,
                        loss_obj=Loss(),
                        lambda_sem=lambda_sem,
                        save_model_n_th_epoch=1,
                        saving_path=str(paths.model_checkpoint))
    
    return trainer.train()


# GAN-LSTM (adversarial)
def _train_t(cfg: ExperimentConfig, paths: ExperimentPaths, lambda_sem: float, root=None):
    from suffix_pred.models.GAN_LSTM import TaymouriAdversarialLSTM
    from suffix_pred.loss import Loss
    from suffix_pred.train import TTraining

    m, ex = cfg.model, cfg.model.extra
    train_set, val_set = _load_datasets(paths)
    seq_len_pred = train_set.min_suffix_size

    enc_feat, _, _, _ = _select_features(cfg)
    concept_idx, output_size_act, eos_id = _concept_info(train_set, cfg.dataset.concept_name)

    model = TaymouriAdversarialLSTM(data_set_categories=train_set.all_categories,
                                    model_feat=enc_feat,
                                    concept_name_id=concept_idx,
                                    hidden_size=m.hidden_size,
                                    num_layers=m.num_layers,
                                    seq_len_pred=seq_len_pred,
                                    input_size=ex.get("input_size", 1),
                                    output_size_act=output_size_act,
                                    dropout=m.dropout)

    lr, epochs, warm_started = _resolve_lr_epochs(cfg, model, root)
    if not warm_started:
        # Cold start: init G and D with standard-normal (Algorithm 1, step 1).
        model.init_weights_normal()

    generator_optimizer = _build_optimizer("rmsprop", model.seq2seq.parameters(), lr)
    discriminator_optimizer = _build_optimizer("rmsprop", model.discriminator.parameters(), lr)

    optimize_values = {"optimizer": generator_optimizer,
                       "generator_optimizer": generator_optimizer,
                       "discriminator_optimizer": discriminator_optimizer,
                       "scheduler": None,
                       "epochs": epochs,
                       "mini_batches": m.batch_size,
                       "shuffle": True,
                       "min_teacher_forcing_value": ex.get("min_teacher_forcing_value", 0.0),
                       "max_teacher_forcing_value": ex.get("max_teacher_forcing_value", 1.0),
                       "teacher_forcing_mode": ex.get("teacher_forcing_mode", "scheduled"),
                       "use_gan": ex.get("use_gan", True),
                       "tau_start": ex.get("tau_start", 0.9),
                       "tau_min": ex.get("tau_min", 0.01),
                       "beam_width": ex.get("beam_width", 3),
                       "tau": m.tau,
                       "sem_gate_gt_in_support": ex.get("sem_gate_gt_in_support", True)}

    trainer = TTraining(device=_device(), model=model,
                        data_train=train_set,
                        data_val=val_set,
                        optimize_values=optimize_values,
                        suffix_data_split_value=seq_len_pred,
                        concept_name_id=concept_idx,
                        eos_id=eos_id, loss_obj=Loss(),
                        lambda_sem=lambda_sem,
                        save_model_n_th_epoch=1,
                        saving_path=str(paths.model_checkpoint))
    
    return trainer.train()
