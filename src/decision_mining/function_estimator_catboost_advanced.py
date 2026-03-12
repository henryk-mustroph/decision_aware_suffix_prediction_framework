from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re
from statistics import NormalDist

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_integer_dtype, is_float_dtype

MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"
RARE_TOKEN = "__RARE__"

class _PriorProbModel:
    """
    Fallback model when CatBoost cannot train (e.g., all features constant/ignored).
    """
    _is_prior_model = True

    def __init__(self, priors: np.ndarray):
        p = np.asarray(priors, dtype=float)
        s = float(p.sum())
        if s <= 0:
            p = np.ones_like(p, dtype=float) / float(max(1, p.size))
        else:
            p = p / s
        self.priors_ = p
        self.classes_ = np.arange(int(p.size), dtype=int)

    def fit(self, X: Any, y: Any = None, sample_weight: Any = None, **kwargs: Any) -> "_PriorProbModel":
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        try:
            n = int(len(X))
        except Exception:
            n = 1
        return np.tile(self.priors_.reshape(1, -1), (max(1, n), 1))


@dataclass
class ModelConfig:
    # Base model: "catboost" (preferred) or "sklearn_tree"
    model_type: str = "catboost"
    random_state: int = 7

    # Base model hyperparams. Keep defaults modest and rely on early stopping.
    cb_iterations: int = 600
    cb_learning_rate: float = 0.05
    cb_depth: int = 6
    cb_l2_leaf_reg: float = 3.0
    cb_loss: str = "Logloss"  # auto-switched to "MultiClass" if needed
    cb_eval_fraction: float = 0.15
    cb_early_stopping_rounds: int = 80
    cb_use_best_model: bool = True
    cb_thread_count: int = -1
    cb_allow_writing_files: bool = False

    # Imbalance handling
    use_inverse_freq_weights: bool = True

    # Encoding for surrogate tree + guard readability
    treat_int_as_categorical: bool = True
    low_card_max: int = 30
    high_card_top_k: int = 50
    min_category_freq: int = 2
    include_missing_token: bool = True

    # Surrogate tree (for guards)
    surrogate_max_depth: Optional[int] = 6
    surrogate_min_samples_leaf: int = 30
    surrogate_target: str = "true_label"  # true_label | base_argmax
    surrogate_top_k_features: Optional[int] = 20
    surrogate_tune_ccp_alpha: bool = True
    surrogate_pruning_cv: int = 3
    surrogate_pruning_max_alphas: int = 12
    surrogate_pruning_metric: str = "auto"  # "auto" | "prauc" (binary) | "f1_macro" (multiclass)

    # Probability calibration (base model). Disabled by default because
    # CatBoost is usually calibrated enough for this use case, and CV
    # calibration dominates runtime across many decision places.
    calibrate: bool = False
    calibration_method: str = "sigmoid"
    calibration_cv: int = 3

    # Guard extraction
    guard_ci: float = 0.95
    include_missing_in_guards: bool = False
    min_leaf_prob: float = 0.2
    min_leaf_lift: float = 2.0
    min_leaf_support: int = 20
    always_keep_best: bool = True
    max_rules_per_label: int = 12
    rule_sort: str = "score"  # score|prob|lift|support

def _clean_value(v: Any) -> Any:
    if v is None:
        return None
    try:
        if isinstance(v, float) and np.isnan(v):
            return v
    except Exception:
        pass
    if isinstance(v, (str, int, float, bool, np.number)):
        return v
    try:
        return str(v)
    except Exception:
        return repr(v)

def _priors_from_ints(y_int: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y_int)
    s = float(counts.sum()) if counts.size else 1.0
    return {i: float(counts[i]) / s for i in range(counts.size)}

def compute_inverse_freq_weights(y_int: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_int)
    counts = np.maximum(counts, 1)
    n = float(y_int.shape[0])
    w_per_class = n / (len(counts) * counts.astype(float))
    return w_per_class[y_int]

# helper for guards
def _wilson_interval(k: float, n: float, ci: float = 0.95) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1.0 - (1.0 - float(ci)) / 2.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
    return float(max(0.0, center - half)), float(min(1.0, center + half))

def _normal_ci(mean: float, std: float, n: int, ci: float = 0.95) -> Tuple[float, float]:
    if n <= 1:
        return max(0.0, mean), min(1.0, mean)
    z = NormalDist().inv_cdf(1.0 - (1.0 - float(ci)) / 2.0)
    se = float(std) / math.sqrt(float(n))
    lo = max(0.0, mean - z * se)
    hi = min(1.0, mean + z * se)
    return float(lo), float(hi)

def _detect_categorical_columns(X: pd.DataFrame, treat_int_as_categorical: bool) -> List[str]:
    cat_cols: List[str] = []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s.dtype):
            cat_cols.append(c)
            continue
        # pandas "category" dtype should be treated as categorical
        if isinstance(s.dtype, pd.CategoricalDtype):
            cat_cols.append(c)
            continue
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
            cat_cols.append(c)
            continue
        if treat_int_as_categorical and is_integer_dtype(s.dtype) and not is_float_dtype(s.dtype):
            cat_cols.append(c)
            continue
    return cat_cols


@dataclass
class FeatureEncoder:
    raw_columns: List[str]
    dummy_columns: List[str]
    prefix_sep: str
    cat_info: Dict[str, Dict[str, Any]]
    categorical_levels: Dict[str, List[str]]

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = X_raw.reindex(columns=self.raw_columns).copy()
        X = X.apply(lambda s: s.map(_clean_value))

        for col, info in self.cat_info.items():
            if col not in X.columns:
                continue
            s = X[col].astype("string")
            if info.get("use_missing_token", True):
                s = s.fillna(info.get("missing_token", MISSING_TOKEN))

            kept = info.get("kept", set())
            s = s.where(s.isin(kept), other=info.get("other_token", OTHER_TOKEN))

            # rare bucketing only if created at fit time
            if info.get("rare_values") is not None:
                rv = set(info["rare_values"])
                s = s.where(~s.isin(rv), other=info.get("rare_token", RARE_TOKEN))

            X[col] = s

        X_enc = pd.get_dummies(X, dummy_na=False, prefix_sep=self.prefix_sep)
        return X_enc.reindex(columns=self.dummy_columns, fill_value=0)

def fit_feature_encoder(X_raw: pd.DataFrame, cfg: ModelConfig, prefix_sep: str = "=") -> Tuple[pd.DataFrame, FeatureEncoder]:
    X = X_raw.copy()
    X = X.apply(lambda s: s.map(_clean_value))

    raw_cols = list(X.columns)
    cat_cols = _detect_categorical_columns(X, bool(cfg.treat_int_as_categorical))

    cat_info: Dict[str, Dict[str, Any]] = {}
    X_work = X.copy()

    for col in cat_cols:
        s = X_work[col].astype("string")
        use_missing = bool(cfg.include_missing_token)
        if use_missing:
            s = s.fillna(MISSING_TOKEN)

        vc = s.value_counts(dropna=False)
        nunique = int(vc.shape[0])

        info: Dict[str, Any] = {
            "use_missing_token": use_missing,
            "missing_token": MISSING_TOKEN,
            "other_token": OTHER_TOKEN,
            "rare_token": RARE_TOKEN,
            "kept": set(),
            "rare_values": None,
        }

        if nunique <= int(cfg.low_card_max):
            min_freq = int(cfg.min_category_freq)
            rare_values = vc[vc < min_freq].index.astype(str).tolist() if min_freq > 1 else []
            kept = set(vc.index.astype(str).tolist())
            info["kept"] = kept
            if rare_values:
                info["rare_values"] = rare_values
                rv = set(rare_values)
                s = s.where(~s.isin(rv), other=RARE_TOKEN)
        else:
            top_k = int(cfg.high_card_top_k)
            kept_list = vc.head(top_k).index.astype(str).tolist()
            kept = set(kept_list)
            if use_missing and MISSING_TOKEN in vc.index.astype(str).tolist():
                kept.add(MISSING_TOKEN)
            kept.add(OTHER_TOKEN)
            info["kept"] = kept
            allowed = kept_list + ([MISSING_TOKEN] if use_missing else [])
            s = s.where(s.isin(allowed), other=OTHER_TOKEN)

        X_work[col] = s
        cat_info[col] = info

    X_enc = pd.get_dummies(X_work, dummy_na=False, prefix_sep=prefix_sep)

    categorical_levels: Dict[str, List[str]] = {c: [] for c in cat_cols}
    for col in X_enc.columns:
        if prefix_sep in col:
            base, level = col.split(prefix_sep, 1)
            if base in categorical_levels:
                categorical_levels[base].append(level)

    enc = FeatureEncoder(
        raw_columns=raw_cols,
        dummy_columns=list(X_enc.columns),
        prefix_sep=prefix_sep,
        cat_info=cat_info,
        categorical_levels={k: sorted(set(v)) for k, v in categorical_levels.items()},
    )
    return X_enc, enc


def _score_metric(y_true: np.ndarray, proba_pos: Optional[np.ndarray], y_pred: np.ndarray, metric: str) -> float:
    if metric == "prauc":
        if proba_pos is None:
            return -1e9
        return float(average_precision_score(y_true, proba_pos))
    return float(f1_score(y_true, y_pred, average="macro"))


def select_ccp_alpha_cv(X: pd.DataFrame,
                        y: np.ndarray,
                        *,
                        random_state: int,
                        max_depth: Optional[int],
                        min_samples_leaf: int,
                        sample_weight: Optional[np.ndarray],
                        cv: int,
                        max_alphas: int,
                        metric: str,
                        ) -> float:

    base = DecisionTreeClassifier(
        random_state=random_state,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        ccp_alpha=0.0,
    )
    path = base.cost_complexity_pruning_path(X, y, sample_weight=sample_weight)
    alphas = np.unique(path.ccp_alphas.astype(float))
    if alphas.size == 0:
        return 0.0
    alphas = np.unique(alphas[:-1]) if alphas.size > 1 else alphas
    if alphas.size == 0:
        return 0.0
    if alphas.size > max_alphas:
        qs = np.linspace(0.0, 1.0, max_alphas)
        alphas = np.unique(np.quantile(alphas, qs))

    n = int(len(y))
    counts = np.bincount(y)
    min_class = int(counts[counts > 0].min()) if (counts > 0).any() else 0
    cv = int(min(cv, n, min_class))
    if cv < 2:
        return 0.0

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    best_a, best_s = float(alphas[0]), -1e18

    binary = (len(np.unique(y)) == 2)

    metric_cfg = str(metric).lower().strip()
    if metric_cfg == "auto":
        metric_use = "prauc" if binary else "f1_macro"
    else:
        metric_use = metric_cfg
        if metric_use == "prauc" and not binary:
            metric_use = "f1_macro"

    for a in alphas.tolist():
        scores: List[float] = []
        for tr, te in skf.split(X, y):
            clf = DecisionTreeClassifier(
                random_state=random_state,
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=float(a),
            )
            sw_tr = sample_weight[tr] if sample_weight is not None else None
            clf.fit(X.iloc[tr], y[tr], sample_weight=sw_tr)

            proba_pos = None
            if binary and metric_use == "prauc":
                proba_pos = clf.predict_proba(X.iloc[te])[:, 1]
            y_hat = clf.predict(X.iloc[te])
            scores.append(_score_metric(y[te], proba_pos, y_hat, metric_use))

        m = float(np.mean(scores)) if scores else -1e18
        if m > best_s:
            best_s, best_a = m, float(a)

    return float(best_a)


class FunctionEstimator:
    def __init__(self, model_cfg: Optional[ModelConfig] = None) -> None:
        self.cfg = model_cfg or ModelConfig()

        self.encoder: Optional[FeatureEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_int_to_label: Dict[int, str] = {}

        self.base_model: Optional[Any] = None   # CatBoost or sklearn tree
        self.base_model_cal: Optional[Any] = None
        self.surrogate_tree: Optional[DecisionTreeClassifier] = None

        # CatBoost helpers
        self._cb_cat_cols: List[str] = []
        self.base_raw_feature_names: List[str] = []

        self.feature_names: List[str] = []
        self.surrogate_raw_feature_names: List[str] = []
        self._priors: Dict[int, float] = {}

        self._train_X_raw: Optional[pd.DataFrame] = None
        self._train_X_enc: Optional[pd.DataFrame] = None
        self._train_y_int: Optional[np.ndarray] = None
        self._train_base_proba: Optional[np.ndarray] = None

    def to_artifact(self) -> Dict[str, Any]:
        return {
            "artifact_type": "decision_mining_estimator",
            "artifact_version": 1,
            "estimator_module": __name__,
            "estimator_class": self.__class__.__name__,
            "state": {
                "cfg": asdict(self.cfg),
                "encoder": asdict(self.encoder) if self.encoder is not None else None,
                "label_encoder": self.label_encoder,
                "class_int_to_label": self.class_int_to_label,
                "base_model": self.base_model,
                "base_model_cal": self.base_model_cal,
                "surrogate_tree": self.surrogate_tree,
                "_cb_cat_cols": self._cb_cat_cols,
                "base_raw_feature_names": self.base_raw_feature_names,
                "feature_names": self.feature_names,
                "surrogate_raw_feature_names": self.surrogate_raw_feature_names,
                "_priors": self._priors,
                "_train_X_raw": self._train_X_raw,
                "_train_X_enc": self._train_X_enc,
                "_train_y_int": self._train_y_int,
                "_train_base_proba": self._train_base_proba,
            },
        }

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "FunctionEstimator":
        state = dict(artifact.get("state", {}))
        cfg_dict = state.pop("cfg", None) or {}
        encoder_dict = state.pop("encoder", None)

        obj = cls(model_cfg=ModelConfig(**cfg_dict))
        if encoder_dict is not None:
            obj.encoder = FeatureEncoder(**encoder_dict)
        for key, value in state.items():
            setattr(obj, key, value)
        return obj

    def _select_surrogate_features(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """Reduce surrogate input space to the most informative raw features.

        This keeps guard trees smaller and faster while preserving the CatBoost
        base model as the predictive model.
        """
        top_k = self.cfg.surrogate_top_k_features
        if top_k is None:
            self.surrogate_raw_feature_names = list(X_raw.columns)
            return X_raw

        try:
            top_k = int(top_k)
        except Exception:
            top_k = None

        if top_k is None or top_k <= 0 or top_k >= len(X_raw.columns):
            self.surrogate_raw_feature_names = list(X_raw.columns)
            return X_raw

        model = self.base_model
        if model is None or getattr(model, "_is_prior_model", False):
            self.surrogate_raw_feature_names = list(X_raw.columns)
            return X_raw

        if not hasattr(model, "get_feature_importance"):
            self.surrogate_raw_feature_names = list(X_raw.columns)
            return X_raw

        try:
            importances = np.asarray(model.get_feature_importance(), dtype=float)
        except Exception:
            self.surrogate_raw_feature_names = list(X_raw.columns)
            return X_raw

        if importances.ndim != 1 or importances.size != len(X_raw.columns):
            self.surrogate_raw_feature_names = list(X_raw.columns)
            return X_raw

        order = np.argsort(importances)[::-1]
        selected_idx = [int(i) for i in order[:top_k] if float(importances[int(i)]) > 0.0]
        if not selected_idx:
            selected_idx = [int(i) for i in order[:top_k]]

        selected_cols = [str(X_raw.columns[i]) for i in selected_idx]
        if not selected_cols:
            selected_cols = list(X_raw.columns)

        self.surrogate_raw_feature_names = selected_cols
        return X_raw[selected_cols].copy()

    def _get_surrogate_targets(self, y_true: np.ndarray, base_proba: np.ndarray) -> np.ndarray:
        target_mode = str(self.cfg.surrogate_target).lower().strip()
        if target_mode == "base_argmax":
            return np.argmax(base_proba, axis=1).astype(int)
        return np.asarray(y_true, dtype=int)

    def _cast_cat_columns_for_catboost(self, X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Cast CatBoost categorical columns to strings and fill missing token.

        CatBoost requires categorical columns to be provided (via cat_features) and values
        must be consistently typed; we use pandas string dtype and an explicit missing token.
        """
        if not cat_cols:
            return X

        X_cb = X.copy()
        for c in cat_cols:
            if c not in X_cb.columns:
                X_cb[c] = pd.NA
            s = X_cb[c].astype("string")
            if bool(self.cfg.include_missing_token):
                s = s.fillna(MISSING_TOKEN)
            X_cb[c] = s
        return X_cb

    def _prepare_catboost_X(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[int]]:
        """Prepare data for CatBoost: detect categorical columns, cast to string, fill missing token."""
        cat_cols = _detect_categorical_columns(X, bool(self.cfg.treat_int_as_categorical))
        self._cb_cat_cols = cat_cols
        cat_idx = [X.columns.get_loc(c) for c in cat_cols]

        X_cb = self._cast_cat_columns_for_catboost(X, cat_cols)
        return X_cb, cat_idx

    @classmethod
    def fit_from_xy(cls,
                    X_raw: pd.DataFrame,
                    y: List[str],
                    feature_cols: Optional[List[str]] = None,
                    model_cfg: Optional[ModelConfig] = None):
        est = cls(model_cfg=model_cfg)
        est.fit(X_raw, y, feature_cols=feature_cols)
        return est

    def fit(self, X_raw: pd.DataFrame, y: List[str], feature_cols: Optional[List[str]] = None):
        mt = str(self.cfg.model_type).lower()
        is_catboost = (mt == "catboost")

        X_work = X_raw.copy()
        if feature_cols:
            for c in feature_cols:
                if c not in X_work.columns:
                    X_work[c] = np.nan
            X_work = X_work[feature_cols]

        X_work = X_work.apply(lambda s: s.map(_clean_value))
        self.base_raw_feature_names = list(X_work.columns)

        # encode labels
        le = LabelEncoder()
        # classes as index
        y_int_full = le.fit_transform(np.asarray(y, dtype=str))
        # dict: key: index, value: class name
        self.class_int_to_label = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
        # distribution of classes: dict: key: index, value: percentage of ocurence
        self._priors = _priors_from_ints(y_int_full)

        # Train base model on full data
        X_train = X_work
        y_train = y_int_full

        # weights for base model
        sw_train = compute_inverse_freq_weights(y_train) if bool(self.cfg.use_inverse_freq_weights) else None

        # Train base model: CatBoost model
        self.base_model = self._fit_base_model(X_train, y_train, sw_train)

        # Calibrate base model probabilities on FULL data: (so probabilities used in guards match calibration distribution)
        self.base_model_cal = self.base_model
        if bool(self.cfg.calibrate) and not getattr(self.base_model, "_is_prior_model", False):
            n = int(len(y_int_full))
            counts = np.bincount(y_int_full)
            nonzero = counts[counts > 0]
            min_class = int(nonzero.min()) if nonzero.size else 0
            cv = int(min(int(self.cfg.calibration_cv), n, min_class))
            if cv >= 2:
                # CalibratedClassifierCV will refit base model in CV folds. If that’s too slow,
                # set cfg.calibrate=False (CatBoost is often well-calibrated already).
                cal = CalibratedClassifierCV(
                    self._make_fresh_base_model(n_classes=int(len(le.classes_))),  # fresh estimator for CV refit
                    method=str(self.cfg.calibration_method),
                    cv=cv,
                )
                sw_full = compute_inverse_freq_weights(y_int_full) if bool(self.cfg.use_inverse_freq_weights) else None

                # Important for CatBoost: pass cat_features during CV refits,
                # otherwise CatBoost will treat string columns as numeric and crash.
                if is_catboost:
                    X_cb_full, cat_idx = self._prepare_catboost_X(X_work)
                    cal.fit(X_cb_full, y_int_full, sample_weight=sw_full, cat_features=cat_idx)
                else:
                    cal.fit(X_work, y_int_full, sample_weight=sw_full)
                self.base_model_cal = cal

        # probabilities on FULL data (for surrogate + guard stats)
        if is_catboost and not getattr(self.base_model_cal, "_is_prior_model", False):
            X_cb_full, _ = self._prepare_catboost_X(X_work)
            base_proba = self.base_model_cal.predict_proba(X_cb_full)
        else:
            base_proba = self.base_model_cal.predict_proba(X_work)
        self._train_base_proba = np.asarray(base_proba, dtype=float)

        # Fit surrogate encoder & surrogate tree on a reduced raw feature set.
        X_surrogate_raw = self._select_surrogate_features(X_work)
        X_enc_full, encoder = fit_feature_encoder(X_surrogate_raw, self.cfg, prefix_sep="=")
        self.encoder = encoder
        self.feature_names = list(X_enc_full.columns)
        self._train_X_enc = X_enc_full
        self._train_X_raw = X_surrogate_raw
        self._train_y_int = y_int_full

        # Surrogate target can either mimic the base model or explain the true
        # observed next activity. The latter usually yields cleaner guards.
        y_sur = self._get_surrogate_targets(y_int_full, self._train_base_proba)
        sw_sur = compute_inverse_freq_weights(y_sur) if bool(self.cfg.use_inverse_freq_weights) else None

        alpha = 0.0
        if bool(self.cfg.surrogate_tune_ccp_alpha) and int(np.unique(y_sur).size) >= 2:
            alpha = select_ccp_alpha_cv(X_enc_full,
                                        y_sur,
                                        random_state=self.cfg.random_state,
                                        max_depth=self.cfg.surrogate_max_depth,
                                        min_samples_leaf=self.cfg.surrogate_min_samples_leaf,
                                        sample_weight=sw_sur,
                                        cv=int(self.cfg.surrogate_pruning_cv),
                                        max_alphas=int(self.cfg.surrogate_pruning_max_alphas),
                                        metric=str(self.cfg.surrogate_pruning_metric))

        # return surrogate decision tree for guards
        tree = DecisionTreeClassifier(
            random_state=self.cfg.random_state,
            max_depth=self.cfg.surrogate_max_depth,
            min_samples_leaf=self.cfg.surrogate_min_samples_leaf,
            ccp_alpha=float(alpha),
        )
        tree.fit(X_enc_full, y_sur, sample_weight=sw_sur)
        self.surrogate_tree = tree

        self.label_encoder = le
        
        return self

    def _make_fresh_base_model(self, n_classes: Optional[int] = None) -> Any:
        """A fresh estimator instance (used for calibration CV refits)."""
        if str(self.cfg.model_type).lower() == "catboost":
            try:
                from catboost import CatBoostClassifier
            except Exception as e:
                raise RuntimeError("catboost is not installed. `pip install catboost`") from e

            loss = str(self.cfg.cb_loss)
            if n_classes is not None:
                loss = "Logloss" if int(n_classes) == 2 else "MultiClass"

            return CatBoostClassifier(
                iterations=int(self.cfg.cb_iterations),
                learning_rate=float(self.cfg.cb_learning_rate),
                depth=int(self.cfg.cb_depth),
                l2_leaf_reg=float(self.cfg.cb_l2_leaf_reg),
                loss_function=str(loss),
                random_seed=int(self.cfg.random_state),
                thread_count=int(self.cfg.cb_thread_count),
                allow_writing_files=bool(self.cfg.cb_allow_writing_files),
                verbose=False,
            )
        # fallback
        return DecisionTreeClassifier(random_state=self.cfg.random_state)

    def _fit_base_model(self, X: pd.DataFrame, y_int: np.ndarray, sample_weight: Optional[np.ndarray]) -> Any:
        """Train the base model, with a safe prior fallback for degenerate feature sets."""
        mt = str(self.cfg.model_type).lower()
        n_classes = int(len(np.unique(y_int)))
        n_classes_total = int(len(self.class_int_to_label)) if self.class_int_to_label else n_classes
        if mt == "catboost":
            try:
                from catboost import CatBoostClassifier
            except Exception as e:
                raise RuntimeError("catboost is not installed. `pip install catboost`") from e

            X_cb, cat_idx = self._prepare_catboost_X(X)

            def make_prior_model() -> _PriorProbModel:
                counts = np.bincount(np.asarray(y_int, dtype=int), minlength=n_classes_total).astype(float)
                priors = counts / float(max(1.0, counts.sum()))
                return _PriorProbModel(priors)

            # CatBoost errors out if *all* features are constant/ignored.
            try:
                nunique = X_cb.nunique(dropna=False)
                if int((nunique > 1).sum()) == 0:
                    return make_prior_model()
            except Exception:
                pass

            loss = "Logloss" if n_classes == 2 else "MultiClass"
            model = CatBoostClassifier(
                iterations=int(self.cfg.cb_iterations),
                learning_rate=float(self.cfg.cb_learning_rate),
                depth=int(self.cfg.cb_depth),
                l2_leaf_reg=float(self.cfg.cb_l2_leaf_reg),
                loss_function=loss,
                random_seed=int(self.cfg.random_state),
                thread_count=int(self.cfg.cb_thread_count),
                allow_writing_files=bool(self.cfg.cb_allow_writing_files),
                verbose=False,
            )
            try:
                fit_kwargs: Dict[str, Any] = {
                    "cat_features": cat_idx,
                    "sample_weight": sample_weight,
                }

                eval_fraction = float(self.cfg.cb_eval_fraction)
                early_rounds = int(self.cfg.cb_early_stopping_rounds)
                class_counts = np.bincount(y_int)
                nonzero = class_counts[class_counts > 0]
                min_class = int(nonzero.min()) if nonzero.size else 0
                can_split = (
                    eval_fraction > 0.0
                    and len(y_int) >= 50
                    and min_class >= 2
                )

                if can_split:
                    idx = np.arange(len(y_int))
                    idx_fit, idx_eval = train_test_split(
                        idx,
                        test_size=eval_fraction,
                        random_state=int(self.cfg.random_state),
                        stratify=y_int,
                    )
                    X_fit = X_cb.iloc[idx_fit]
                    y_fit = y_int[idx_fit]
                    X_eval = X_cb.iloc[idx_eval]
                    y_eval = y_int[idx_eval]
                    fit_kwargs["sample_weight"] = sample_weight[idx_fit] if sample_weight is not None else None
                    fit_kwargs["eval_set"] = (X_eval, y_eval)
                    fit_kwargs["use_best_model"] = bool(self.cfg.cb_use_best_model)
                    fit_kwargs["early_stopping_rounds"] = early_rounds
                    model.fit(X_fit, y_fit, **fit_kwargs)
                else:
                    model.fit(X_cb, y_int, **fit_kwargs)
                return model
            except Exception as e:
                if "All features are either constant or ignored" in str(e):
                    return make_prior_model()
                raise

        # fallback: plain tree (less accurate on high-card cats, but runs everywhere)
        clf = DecisionTreeClassifier(
            random_state=self.cfg.random_state,
            max_depth=self.cfg.surrogate_max_depth,
            min_samples_leaf=self.cfg.surrogate_min_samples_leaf,
        )
        clf.fit(X, y_int, sample_weight=sample_weight)
        return clf

    def predict_proba(self, assignment: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
        if self.base_model_cal is None:
            raise RuntimeError("Estimator is not fitted.")
        X_raw = pd.DataFrame([assignment]).apply(lambda s: s.map(_clean_value))

        if self.base_raw_feature_names:
            for col in self.base_raw_feature_names:
                if col not in X_raw.columns:
                    X_raw[col] = np.nan
            X_raw = X_raw[self.base_raw_feature_names]

        if getattr(self.base_model_cal, "_is_prior_model", False):
            proba = self.base_model_cal.predict_proba(X_raw)[0]
        else:
            mt = str(self.cfg.model_type).lower()
            # Keep CatBoost categorical handling consistent between fit/predict
            if mt == "catboost" and self._cb_cat_cols:
                X_cb = self._cast_cat_columns_for_catboost(X_raw, self._cb_cat_cols)
                proba = self.base_model_cal.predict_proba(X_cb)[0]
            else:
                proba = self.base_model_cal.predict_proba(X_raw)[0]

        class_ints = list(getattr(self.base_model_cal, "classes_", np.arange(len(proba))))
        labels = [self.class_int_to_label[int(c)] for c in class_ints]
        return labels, np.asarray(proba, dtype=float)

    def extract_probabilistic_guards_advanced(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Guards are extracted from surrogate tree structure, but probabilities are computed from:
          - model_proba: mean calibrated base probability within the leaf (+ normal CI)
          - emp_proba: empirical label frequency in the leaf (+ Wilson CI)

        Output per label contains:
          rule, intervals, categorical_allowed, categorical_excluded,
          prob_model, prob_model_ci_low/high,
          prob_emp, prob_emp_ci_low/high,
          support, coverage, lift, score
        """
        if self.surrogate_tree is None or self.encoder is None or self._train_X_enc is None or self._train_y_int is None or self._train_base_proba is None:
            raise RuntimeError("Estimator is not fitted or missing training caches.")

        tree = self.surrogate_tree
        X_enc = self._train_X_enc
        y_true = self._train_y_int
        P = self._train_base_proba
        ci = float(self.cfg.guard_ci)

        # leaf assignment
        leaf_ids = tree.apply(X_enc)
        leaf_to_idx: Dict[int, np.ndarray] = {}
        for leaf in np.unique(leaf_ids):
            leaf_to_idx[int(leaf)] = np.where(leaf_ids == leaf)[0]

        # compute leaf stats for all classes
        n_classes = P.shape[1]
        leaf_stats: Dict[int, Dict[str, Any]] = {}
        for leaf, idx in leaf_to_idx.items():
            support = int(idx.size)
            if support <= 0:
                continue

            probs = P[idx, :]  # model probs
            mu = probs.mean(axis=0)
            sd = probs.std(axis=0, ddof=1) if support > 1 else np.zeros(n_classes, dtype=float)

            # empirical counts
            counts = np.bincount(y_true[idx], minlength=n_classes).astype(float)
            n = float(counts.sum())

            per_class: List[Dict[str, float]] = []
            for j in range(n_classes):
                pm = float(mu[j])
                lo_m, hi_m = _normal_ci(pm, float(sd[j]), support, ci=ci)

                pe = float(counts[j] / max(1.0, n))
                lo_e, hi_e = _wilson_interval(float(counts[j]), n, ci=ci)

                per_class.append(
                    {
                        "prob_model": pm,
                        "prob_model_ci_low": float(lo_m),
                        "prob_model_ci_high": float(hi_m),
                        "prob_emp": pe,
                        "prob_emp_ci_low": float(lo_e),
                        "prob_emp_ci_high": float(hi_e),
                    }
                )

            leaf_stats[int(leaf)] = {"support": support, "per_class": per_class}

        # traverse tree to produce rules per leaf node id
        rules_by_label = _extract_rules_from_surrogate_with_stats(
            tree=tree,
            feature_names=self.feature_names,
            class_int_to_label=self.class_int_to_label,
            priors=self._priors,
            leaf_stats=leaf_stats,
            min_leaf_prob=float(self.cfg.min_leaf_prob),
            min_leaf_lift=float(self.cfg.min_leaf_lift),
            min_leaf_support=int(self.cfg.min_leaf_support),
            always_keep_best=bool(self.cfg.always_keep_best),
        )

        # parse + simplify to (sets + intervals)
        cond_re = re.compile(r"\((.*?)\s*(<=|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)")
        total_n = int(len(y_true))

        out: Dict[str, List[Dict[str, Any]]] = {}
        for label, ros in rules_by_label.items():
            guards: List[Dict[str, Any]] = []
            for ro in ros:
                raw = str(ro["rule"])
                support = int(ro["support"])
                coverage = float(support) / float(max(1, total_n))
                lift = float(ro["lift"])

                intervals: Dict[str, Dict[str, Optional[float]]] = {}
                cat_inc_exc: Dict[str, Dict[str, List[str]]] = {}

                if raw.strip() != "(true)":
                    for feat, op, num_str in cond_re.findall(raw):
                        try:
                            thresh = float(num_str)
                        except ValueError:
                            continue

                        # dummy feature?
                        if self.encoder.prefix_sep in feat:
                            base, value = feat.split(self.encoder.prefix_sep, 1)
                            if (not bool(self.cfg.include_missing_in_guards)) and (str(value) == MISSING_TOKEN):
                                continue

                            entry = cat_inc_exc.setdefault(base, {"include": [], "exclude": []})
                            if -0.1 <= thresh <= 1.1:
                                if op == ">":
                                    entry["include"].append(str(value))
                                else:
                                    entry["exclude"].append(str(value))
                            continue

                        # numeric
                        iv = intervals.setdefault(feat, {"low": None, "high": None})
                        if op == ">":
                            iv["low"] = thresh if iv["low"] is None else max(float(iv["low"]), thresh)
                        else:
                            iv["high"] = thresh if iv["high"] is None else min(float(iv["high"]), thresh)

                # convert include/exclude to allowed sets where possible
                cat_allowed: Dict[str, List[str]] = {}
                cat_excluded: Dict[str, List[str]] = {}

                for base, inc_exc in cat_inc_exc.items():
                    inc = sorted(set(inc_exc.get("include", [])))
                    exc = sorted(set(inc_exc.get("exclude", [])))
                    all_levels = self.encoder.categorical_levels.get(base, None)

                    if all_levels:
                        all_set = set(map(str, all_levels))
                        allowed = sorted(set(inc)) if inc else sorted(all_set - set(exc))
                        if not bool(self.cfg.include_missing_in_guards):
                            allowed = [v for v in allowed if v != MISSING_TOKEN]
                            exc = [v for v in exc if v != MISSING_TOKEN]
                        cat_allowed[base] = allowed
                        cat_excluded[base] = exc
                    else:
                        cat_allowed[base] = inc
                        cat_excluded[base] = exc

                rule_str = _simplify_guard_rule(intervals, cat_allowed, cat_excluded)

                # choose the primary probability to rank/display (model-based by default)
                pm = float(ro["prob_model"])
                score = pm * math.log1p(support) * max(1.0, lift)

                guards.append(
                    {
                        "rule": rule_str,
                        "raw_rule": raw,
                        "intervals": intervals,
                        "categorical_allowed": cat_allowed,
                        "categorical_excluded": cat_excluded,
                        "prob_model": float(ro["prob_model"]),
                        "prob_model_ci_low": float(ro["prob_model_ci_low"]),
                        "prob_model_ci_high": float(ro["prob_model_ci_high"]),
                        "prob_emp": float(ro["prob_emp"]),
                        "prob_emp_ci_low": float(ro["prob_emp_ci_low"]),
                        "prob_emp_ci_high": float(ro["prob_emp_ci_high"]),
                        "support": support,
                        "coverage": coverage,
                        "lift": lift,
                        "score": float(score),
                    }
                )

            # sort + cap
            key = str(self.cfg.rule_sort)
            if key == "prob":
                guards.sort(key=lambda g: (g["prob_model"], g["support"]), reverse=True)
            elif key == "lift":
                guards.sort(key=lambda g: (g["lift"], g["support"]), reverse=True)
            elif key == "support":
                guards.sort(key=lambda g: (g["support"], g["prob_model"]), reverse=True)
            else:
                guards.sort(key=lambda g: (g["score"], g["prob_model"]), reverse=True)

            k = int(self.cfg.max_rules_per_label)
            if k > 0:
                guards = guards[:k]

            out[label] = guards

        return out


def _simplify_guard_rule(
    intervals: Dict[str, Dict[str, Optional[float]]],
    cat_allowed: Dict[str, List[str]],
    cat_excluded: Dict[str, List[str]],
    *,
    max_set_show: int = 18,
) -> str:
    parts: List[str] = []

    for base in sorted(set(cat_allowed.keys()) | set(cat_excluded.keys())):
        allowed = cat_allowed.get(base, []) or []
        excluded = cat_excluded.get(base, []) or []
        if allowed:
            shown = allowed[:max_set_show]
            suffix = f" (+{len(allowed)-len(shown)} more)" if len(allowed) > len(shown) else ""
            parts.append(f"({base} in {{{', '.join(shown)}}}{suffix})")
        elif excluded:
            shown = excluded[:max_set_show]
            suffix = f" (+{len(excluded)-len(shown)} more)" if len(excluded) > len(shown) else ""
            parts.append(f"({base} not in {{{', '.join(shown)}}}{suffix})")

    for feat in sorted(intervals.keys()):
        iv = intervals[feat] or {}
        low = iv.get("low")
        high = iv.get("high")
        if low is not None:
            parts.append(f"({feat} > {float(low):.6g})")
        if high is not None:
            parts.append(f"({feat} <= {float(high):.6g})")

    return " AND ".join(parts) if parts else "(true)"


def _extract_rules_from_surrogate_with_stats(
    *,
    tree: DecisionTreeClassifier,
    feature_names: List[str],
    class_int_to_label: Dict[int, str],
    priors: Dict[int, float],
    leaf_stats: Dict[int, Dict[str, Any]],
    min_leaf_prob: float,
    min_leaf_lift: float,
    min_leaf_support: int,
    always_keep_best: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Traverse surrogate tree. At each leaf node id (= sklearn apply() id), attach probabilities from leaf_stats.
    Selection uses prob_model and lift (prob_model / prior).
    """
    t = tree.tree_
    feature_undef = -2

    # ensure output keys are true labels
    out: Dict[str, List[Dict[str, Any]]] = {class_int_to_label[i]: [] for i in range(len(class_int_to_label))}
    best: Dict[str, Dict[str, Any]] = {
        class_int_to_label[i]: {"prob_model": -1.0, "support": 0, "rule": "(false)", "lift": 0.0}
        for i in range(len(class_int_to_label))
    }

    def cond_str(feat_idx: int, thresh: float, direction: str) -> str:
        fname = feature_names[feat_idx]
        return f"({fname} <= {thresh:.6g})" if direction == "left" else f"({fname} > {thresh:.6g})"

    stack: List[Tuple[int, List[str]]] = [(0, [])]
    while stack:
        node_id, path = stack.pop()
        feat_idx = int(t.feature[node_id])

        if feat_idx == feature_undef:
            stats = leaf_stats.get(int(node_id))
            if not stats:
                continue
            support = int(stats["support"])
            if support < int(min_leaf_support):
                continue
            rule_str = " AND ".join(path) if path else "(true)"

            per_class = stats["per_class"]
            for c_int in range(len(per_class)):
                lab = class_int_to_label[int(c_int)]
                pm = float(per_class[c_int]["prob_model"])
                prior = float(priors.get(int(c_int), 1e-12))
                lift = pm / max(prior, 1e-12)

                obj = {
                    "rule": rule_str,
                    "support": support,
                    "lift": float(lift),
                    "prob_model": pm,
                    "prob_model_ci_low": float(per_class[c_int]["prob_model_ci_low"]),
                    "prob_model_ci_high": float(per_class[c_int]["prob_model_ci_high"]),
                    "prob_emp": float(per_class[c_int]["prob_emp"]),
                    "prob_emp_ci_low": float(per_class[c_int]["prob_emp_ci_low"]),
                    "prob_emp_ci_high": float(per_class[c_int]["prob_emp_ci_high"]),
                }

                if pm > float(best[lab]["prob_model"]):
                    best[lab] = obj

                if (pm >= float(min_leaf_prob)) or (lift >= float(min_leaf_lift)):
                    out[lab].append(obj)
            continue

        thresh = float(t.threshold[node_id])
        left_id = int(t.children_left[node_id])
        right_id = int(t.children_right[node_id])

        stack.append((right_id, path + [cond_str(feat_idx, thresh, "right")]))
        stack.append((left_id, path + [cond_str(feat_idx, thresh, "left")]))

    if always_keep_best:
        for lab, lst in out.items():
            if not lst and best[lab]["prob_model"] > 0 and best[lab]["support"] >= int(min_leaf_support):
                out[lab] = [best[lab]]

    return out
