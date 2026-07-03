"""
Experiment orchestration for suffix-prediction notebooks.
- configs.py       -- the *only* place the per-(dataset, model, variant) differences live (paths, attribute lists, concept name, hyperparameters). All paths are derived by convention.
- data_loading.py  -- build_dataset(cfg)   (base | decision-labeled)
- training.py      -- train(cfg)            (dispatch UED / C / T)
- evaluation.py    -- evaluate(cfg)         (plain | decision-guided decode)
"""
from .configs import (
    DATASETS,
    MODELS,
    DatasetConfig,
    EventLogSpec,
    ModelConfig,
    ModelFeatures,
    ExperimentConfig,
    Variant,
    resolve_paths,
    resolve_dataset_paths,
    make_experiment,
    check_model_features,
    require_predicted_decision_attrs,
)

__all__ = [
    "DATASETS",
    "MODELS",
    "DatasetConfig",
    "EventLogSpec",
    "ModelConfig",
    "ModelFeatures",
    "ExperimentConfig",
    "Variant",
    "resolve_paths",
    "resolve_dataset_paths",
    "make_experiment",
    "check_model_features",
    "require_predicted_decision_attrs",
]
