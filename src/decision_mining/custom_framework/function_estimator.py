"""


"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

@dataclass
class ModelConfig:
    model_type: str = "sklearn_tree"
    random_state: int = 7
    max_depth: Optional[int] = 5
    min_samples_leaf: int = 20
    ccp_alpha: float = 0.0
    n_estimators: int = 300
    learning_rate: float = 0.05
    use_inverse_freq_weights: bool = True
    use_oss: bool = True
    oss_max_ratio: float = 10.0
    calibrate: bool = True
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5

def _clean_value(v: Any) -> Any:
    """Make values safe for pandas.get_dummies.

    Keep scalars as-is; stringify non-scalars (lists/dicts/objects).
    """
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

@dataclass
class FeatureEncoder:
    raw_columns: List[str]
    dummy_columns: List[str]
    prefix_sep: str = "="

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        # Ensure same raw columns as during fit; missing columns become NaN.
        X = X_raw.reindex(columns=self.raw_columns)

        # Clean weird values column-wise (faster/cleaner than applymap for newer pandas).
        X = X.apply(lambda s: s.map(_clean_value))

        X_enc = pd.get_dummies(X, dummy_na=True, prefix_sep=self.prefix_sep)
        return X_enc.reindex(columns=self.dummy_columns, fill_value=0)

def fit_feature_encoder(X_raw: pd.DataFrame, prefix_sep: str = "=") -> Tuple[pd.DataFrame, FeatureEncoder]:
    X = X_raw.copy()
    X = X.apply(lambda s: s.map(_clean_value))
    raw_cols = list(X.columns)
    X_enc = pd.get_dummies(X, dummy_na=True, prefix_sep=prefix_sep)
    enc = FeatureEncoder(raw_columns=raw_cols, dummy_columns=list(X_enc.columns), prefix_sep=prefix_sep)
    
    return X_enc, enc

def compute_inverse_freq_weights(y_int: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_int)
    counts = np.maximum(counts, 1)
    n = float(y_int.shape[0])
    w_per_class = n / (len(counts) * counts.astype(float))
    return w_per_class[y_int]

def _priors_from_ints(y_int: np.ndarray) -> Dict[int, float]:
    counts = np.bincount(y_int)
    s = float(counts.sum()) if counts.size else 1.0
    return {i: float(counts[i]) / s for i in range(counts.size)}

def _oss_indices(y_int: np.ndarray, max_ratio: float, random_state: int) -> np.ndarray:
    rng = np.random.default_rng(int(random_state))
    counts = np.bincount(y_int)
    nonzero = counts[counts > 0]
    if nonzero.size == 0:
        return np.arange(len(y_int))
    min_count = int(nonzero.min())
    cap = int(max(1, min_count * float(max_ratio)))

    keep_indices: List[int] = []
    for cls, cnt in enumerate(counts):
        if cnt == 0:
            continue
        cls_idx = np.where(y_int == cls)[0]
        if cnt > cap:
            cls_idx = rng.choice(cls_idx, size=cap, replace=False)
        keep_indices.extend(cls_idx.tolist())

    return np.asarray(keep_indices, dtype=int)

def extract_tree_guards_probabilistic(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    class_int_to_label: Dict[int, str],
    priors: Dict[int, float],
    min_leaf_prob: float = 0.2,
    min_leaf_lift: float = 2.0,
    min_leaf_support: int = 20,
    always_keep_best: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns per-label a list of rules with:
      - rule (string)
      - prob (leaf probability for that class, Laplace-smoothed)
      - support (number of samples in the leaf)
      - lift (prob / prior)
    """
    tree = clf.tree_
    feature_undef = -2

    def cond_str(feat_idx: int, thresh: float, direction: str) -> str:
        fname = feature_names[feat_idx]
        return f"({fname} <= {thresh:.6g})" if direction == "left" else f"({fname} > {thresh:.6g})"

    # classes in the tree are ints (LabelEncoder ints)
    class_ints = [int(c) for c in clf.classes_]
    labels = [class_int_to_label[i] for i in class_ints]

    rules_by_label: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in labels}
    best_by_label: Dict[str, Dict[str, Any]] = {lab: {"prob": -1.0, "support": 0, "rule": "(false)", "lift": 0.0} for lab in labels}

    stack: List[Tuple[int, List[str]]] = [(0, [])]
    while stack:
        node_id, path_conds = stack.pop()
        feat_idx = int(tree.feature[node_id])

        if feat_idx == feature_undef:
            # IMPORTANT: use true sample count, not weighted value sum
            support = int(tree.n_node_samples[node_id])
            if support < int(min_leaf_support):
                continue

            counts = tree.value[node_id][0].astype(float)
            # Laplace smoothing (helps with rare classes)
            probs = (counts + 1.0) / (counts.sum() + float(len(counts)))

            rule_str = " AND ".join(path_conds) if path_conds else "(true)"

            for j, c_int in enumerate(class_ints):
                lab = class_int_to_label[int(c_int)]
                p = float(probs[j])
                prior = float(priors.get(int(c_int), 1e-12))
                lift = p / max(prior, 1e-12)

                # track best leaf per label (for fallback)
                if p > float(best_by_label[lab]["prob"]):
                    best_by_label[lab] = {"prob": p, "support": support, "rule": rule_str, "lift": lift}

                # selection: absolute prob OR lift-over-prior
                if (p >= float(min_leaf_prob)) or (lift >= float(min_leaf_lift)):
                    rules_by_label[lab].append(
                        {"rule": rule_str, "prob": p, "support": support, "lift": lift}
                    )
            continue

        thresh = float(tree.threshold[node_id])
        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])

        stack.append((right_id, path_conds + [cond_str(feat_idx, thresh, "right")]))
        stack.append((left_id, path_conds + [cond_str(feat_idx, thresh, "left")]))

    # fallback: ensure at least one rule per label (if possible)
    if always_keep_best:
        for lab in labels:
            if not rules_by_label[lab]:
                best = best_by_label[lab]
                if best["prob"] > 0 and best["support"] >= int(min_leaf_support):
                    rules_by_label[lab] = [best]

    return rules_by_label

class FunctionEstimator:
    def __init__(self, model_cfg: Optional[ModelConfig] = None) -> None:
        
        self.model_cfg = model_cfg or ModelConfig()
        self.encoder: Optional[FeatureEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_int_to_label: Dict[int, str] = {}
        self.clf: Optional[Any] = None  # CalibratedClassifierCV or base estimator
        self._base_clf: Optional[Any] = None
        self.feature_names: List[str] = []

        # keep for surrogate + priors
        self._train_X_enc: Optional[pd.DataFrame] = None
        self._train_y_int: Optional[np.ndarray] = None
        self._train_y_hat: Optional[np.ndarray] = None
        self._priors: Dict[int, float] = {}

    @classmethod
    def fit_from_xy(cls,
                    X_raw: pd.DataFrame,
                    y: List[str],
                    feature_cols: Optional[List[str]] = None,
                    model_cfg: Optional[ModelConfig] = None) -> "FunctionEstimator":
        # decision mining custom fremwork: ModelConfig(model_type='sklearn_tree', random_state=7, max_depth=5, min_samples_leaf=20, n_estimators=300, learning_rate=0.05, use_inverse_freq_weights=True, use_oss=True, oss_max_ratio=10.0, calibrate=True, calibration_method='sigmoid')
        est = cls(model_cfg=model_cfg)
        est.fit(X_raw, y, feature_cols=feature_cols)
        return est

    def fit(self,
            X_raw: pd.DataFrame, y: List[str],
            feature_cols: Optional[List[str]] = None) -> "FunctionEstimator":
        # X from the miner: X_raw
        X_work = X_raw.copy()
        if feature_cols:
            for c in feature_cols:
                if c not in X_work.columns:
                    X_work[c] = np.nan
            X_work = X_work[feature_cols]

        X_enc_full, encoder = fit_feature_encoder(X_work, prefix_sep="=")

        le = LabelEncoder()
        y_int_full = le.fit_transform(np.asarray(y, dtype=str))
        class_int_to_label = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}

        self._priors = _priors_from_ints(y_int_full)

        X_enc_train = X_enc_full
        y_int_train = y_int_full
        if self.model_cfg.use_oss and self.model_cfg.oss_max_ratio > 1.0:
            keep_idx = _oss_indices(y_int_full, self.model_cfg.oss_max_ratio, self.model_cfg.random_state)
            X_enc_train = X_enc_full.iloc[keep_idx]
            y_int_train = y_int_full[keep_idx]

        sample_weight = None
        if self.model_cfg.use_inverse_freq_weights and self.model_cfg.model_type != "sklearn_tree":
            sample_weight = compute_inverse_freq_weights(y_int_train)

        if self.model_cfg.model_type == "xgboost":
            n_classes = int(len(np.unique(y_int_train)))
            base_clf = XGBClassifier(
                random_state=self.model_cfg.random_state,
                max_depth=self.model_cfg.max_depth if self.model_cfg.max_depth is not None else 6,
                n_estimators=self.model_cfg.n_estimators,
                learning_rate=self.model_cfg.learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob" if n_classes > 2 else "binary:logistic",
                eval_metric="mlogloss" if n_classes > 2 else "logloss",
            )
            base_clf.fit(X_enc_train, y_int_train, sample_weight=sample_weight)
        else:
            # Simple, robust defaults for imbalance + stable leaves
            base_clf = DecisionTreeClassifier(
                random_state=self.model_cfg.random_state,
                max_depth=self.model_cfg.max_depth,
                min_samples_leaf=self.model_cfg.min_samples_leaf,
                ccp_alpha=float(self.model_cfg.ccp_alpha),
                class_weight="balanced" if self.model_cfg.use_inverse_freq_weights else None,
            )
            base_clf.fit(X_enc_train, y_int_train)

        clf: Any = base_clf
        if self.model_cfg.calibrate:
            # CalibratedClassifierCV defaults to cv=5, which fails for small n or rare classes.
            n_samples = int(len(y_int_full))
            counts = np.bincount(y_int_full) if n_samples else np.asarray([], dtype=int)
            nonzero = counts[counts > 0]
            min_class_count = int(nonzero.min()) if nonzero.size else 0

            cv_max = int(getattr(self.model_cfg, "calibration_cv", 5))
            cv = int(min(cv_max, n_samples, min_class_count))

            # Need at least 2 folds and at least cv samples per class for stratification.
            if cv >= 2:
                clf = CalibratedClassifierCV(
                    base_clf,
                    method=self.model_cfg.calibration_method,
                    cv=cv,
                )
                clf.fit(X_enc_full, y_int_full)
            else:
                # Fall back to the uncalibrated classifier.
                clf = base_clf

        self.encoder = encoder
        self.label_encoder = le
        self.class_int_to_label = class_int_to_label
        self.clf = clf
        self._base_clf = base_clf
        self.feature_names = list(X_enc_full.columns)

        self._train_X_enc = X_enc_full
        self._train_y_int = y_int_full

        # cache predictions for surrogate tree (xgb case)
        try:
            self._train_y_hat = np.argmax(base_clf.predict_proba(X_enc_full), axis=1)
        except Exception:
            self._train_y_hat = base_clf.predict(X_enc_full)

        return self

    def predict_proba(self, assignment: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
        if self.encoder is None or self.clf is None:
            raise RuntimeError("Estimator is not fitted.")
        X_raw = pd.DataFrame([assignment])
        X_enc = self.encoder.transform(X_raw)
        proba = self.clf.predict_proba(X_enc)[0]
        class_ints = list(getattr(self._base_clf or self.clf, "classes_", np.arange(len(proba))))
        labels = [self.class_int_to_label[int(c)] for c in class_ints]
        return labels, np.asarray(proba, dtype=float)

    def extract_probabilistic_guards_simple(
        self,
        min_leaf_prob: float = 0.2,
        min_leaf_lift: float = 2.0,
        min_leaf_support: int = 20,
        surrogate_max_depth: int = 4,
        surrogate_min_leaf: int = 20,
        always_keep_best: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        
        if self.encoder is None or self.clf is None:
            raise RuntimeError("Estimator is not fitted.")

        # choose tree: either direct DT or surrogate for XGB
        base = self._base_clf or self.clf
        if isinstance(base, DecisionTreeClassifier):
            tree = base
            priors = self._priors if self._priors else {i: 1.0 for i in self.clf.classes_}
        else:
            if self._train_X_enc is None or self._train_y_hat is None:
                raise RuntimeError("Training data not available for surrogate rules.")
            tree = DecisionTreeClassifier(
                random_state=self.model_cfg.random_state,
                max_depth=surrogate_max_depth,
                min_samples_leaf=surrogate_min_leaf,
            )
            tree.fit(self._train_X_enc, self._train_y_hat)
            priors = _priors_from_ints(self._train_y_hat)

        rules_by_label = extract_tree_guards_probabilistic(
            tree,
            feature_names=self.feature_names,
            class_int_to_label=self.class_int_to_label,
            priors=priors,
            min_leaf_prob=min_leaf_prob,
            min_leaf_lift=min_leaf_lift,
            min_leaf_support=min_leaf_support,
            always_keep_best=always_keep_best,
        )

        # parse conditions
        cond_re = re.compile(
            r"\((.*?)\s*(<=|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)"
        )

        guards: Dict[str, List[Dict[str, Any]]] = {}

        for label, rule_objs in rules_by_label.items():
            label_guards: List[Dict[str, Any]] = []

            for ro in rule_objs:
                rule = ro["rule"]
                rule_prob = float(ro["prob"])
                rule_support = int(ro["support"])
                rule_lift = float(ro["lift"])

                intervals: Dict[str, Dict[str, Optional[float]]] = {}
                categorical_sets: Dict[str, Dict[str, List[Any]]] = {}

                if rule.strip() != "(true)":
                    for feat, op, num_str in cond_re.findall(rule):
                        try:
                            thresh = float(num_str)
                        except ValueError:
                            continue

                        # one-hot feature uses "=" separator, e.g., Resource_prefix_mode=Value 8
                        if "=" in feat:
                            base, value = feat.split("=", 1)
                            entry = categorical_sets.setdefault(base, {"include": [], "exclude": []})

                            # typical one-hot split is around 0.5
                            if op == ">" and thresh >= 0.5:
                                if value not in entry["include"]:
                                    entry["include"].append(value)
                                continue
                            if op == "<=" and thresh < 0.5:
                                if value not in entry["exclude"]:
                                    entry["exclude"].append(value)
                                continue
                            # if threshold is weird, just treat it numerically
                            interval = intervals.setdefault(feat, {"low": None, "high": None})
                            if op == ">":
                                interval["low"] = thresh if interval["low"] is None else max(interval["low"], thresh)
                            else:
                                interval["high"] = thresh if interval["high"] is None else min(interval["high"], thresh)
                            continue

                        # numeric feature interval
                        interval = intervals.setdefault(feat, {"low": None, "high": None})
                        if op == ">":
                            interval["low"] = thresh if interval["low"] is None else max(interval["low"], thresh)
                        else:
                            interval["high"] = thresh if interval["high"] is None else min(interval["high"], thresh)

                label_guards.append(
                    {
                        "intervals": intervals,
                        "categorical_sets": categorical_sets,
                        "prob": rule_prob,
                        "support": rule_support,
                        "lift": rule_lift,
                        "rule": rule,
                    }
                )

            guards[label] = label_guards

        return guards