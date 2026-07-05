"""
CatBoost base model + sklearn surrogate-tree estimator for one decision place.

Two models are fitted on the same labels:
- base model (CatBoost) for highest-accuracy next-event prediction (used to reweight the LSTM softmax at inference and to compute leaf probabilities)
- surrogate decision tree on one-hot-encoded categoricals (so rule strings name the actual category levels and are readable as-is)

Encoding:
- categorical features: native CatBoost handling for the base model; one-hot dummies for the surrogate so leaf paths produce readable rules.
- continuous features: passed through unchanged. DecisionDiscovery is expected to apply the same StandardScalers the LSTM uses *before* fitting, so the surrogate's numeric thresholds live in the LSTM's runtime space.
"""
from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier


MISSING_TOKEN = "__MISSING__"


@dataclass
class ModelConfig:
    # CatBoost base model - it is what we predict with at decoding time, so it
    # is tuned for *calibrated held-out* accuracy rather than raw training fit.
    # Depth and iteration count are kept modest and min_data_in_leaf is raised
    # so the model cannot memorise near-unique continuous attributes (e.g.
    # BPIC2020 case:Amount, which is almost a per-case identifier); without this
    # the base model overfits and collapses below the majority baseline on the
    # test split, destroying the decision-aware reweighting signal.
    cb_iterations: int = 300
    cb_learning_rate: float = 0.05
    cb_depth: int = 4
    cb_l2_leaf_reg: float = 8.0
    cb_min_data_in_leaf: int = 50
    cb_early_stopping_rounds: int = 80
    cb_eval_fraction: float = 0.15

    # Surrogate decision tree - explanation only, so kept small.
    # Depth 4 with ccp_alpha-style cost-complexity pruning yields short,
    # readable rules without sacrificing too much fidelity to CatBoost.
    surrogate_max_depth: int = 4
    surrogate_min_samples_leaf: int = 50
    surrogate_ccp_alpha: float = 0.0  # set > 0 to prune low-impurity splits

    # Guard selection. A leaf emits a rule for label L only when BOTH thresholds
    # are met - "leaf actually prefers L" (min_leaf_prob) AND "leaf prefers L
    # more than the prior does" (min_leaf_lift). The previous OR semantics was
    # too permissive and produced weak rules for non-dominant labels.
    min_leaf_prob: float = 0.15
    min_leaf_lift: float = 1.2
    min_leaf_support: int = 20
    max_rules_per_label: int = 8

    random_state: int = 7
    # Inverse-frequency class weighting. Kept ON for the *surrogate* tree so the
    # explanation rules still surface minority branches, but OFF for the *base*
    # predictive model: weighting the base model makes it favour rare branches
    # and predict below the majority baseline on imbalanced decision places
    # (e.g. BPIC2020 `source`: 99% majority, but a weighted model scored ~1%),
    # which is exactly the calibration the decision-guided decoder relies on.
    use_inverse_freq_weights: bool = True
    use_inverse_freq_weights_base: bool = False


class _PriorProbModel:
    """
    Fallback when CatBoost cannot train (single class, all features constant).
    """
    _is_prior_model = True

    def __init__(self, priors: np.ndarray) -> None:
        p = np.asarray(priors, dtype=float)
        s = float(p.sum())
        self.priors_ = (p / s) if s > 0 else np.ones_like(p) / max(1, p.size)
        self.classes_ = np.arange(int(self.priors_.size), dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        try:
            n = int(len(X))
        except Exception:
            n = 1
        return np.tile(self.priors_.reshape(1, -1), (max(1, n), 1))


def _clean_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return v
    if isinstance(v, (str, int, float, bool, np.number)):
        return v
    try:
        return str(v)
    except Exception:
        return repr(v)


def _inverse_freq_weights(y_int: np.ndarray) -> np.ndarray:
    counts = np.maximum(np.bincount(y_int), 1)
    n = float(y_int.shape[0])
    return (n / (len(counts) * counts.astype(float)))[y_int]


def _detect_categorical_columns(X: pd.DataFrame) -> List[str]:
    """Object/string/bool/pandas-categorical/int columns are treated as categorical."""
    cat_cols: List[str] = []
    for c in X.columns:
        dt = X[c].dtype
        if (pd.api.types.is_object_dtype(dt)
                or pd.api.types.is_string_dtype(dt)
                or pd.api.types.is_bool_dtype(dt)
                or isinstance(dt, pd.CategoricalDtype)
                or is_integer_dtype(dt)):
            cat_cols.append(c)
    return cat_cols


@dataclass
class FeatureEncoder:
    """One-hot encoder used for the surrogate tree only."""
    raw_columns: List[str]
    dummy_columns: List[str]
    categorical_columns: List[str]
    prefix_sep: str = "="

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = X_raw.reindex(columns=self.raw_columns).copy()
        X = X.apply(lambda s: s.map(_clean_value))
        for c in self.categorical_columns:
            if c in X.columns:
                X[c] = X[c].astype("string").fillna(MISSING_TOKEN)
        X_enc = pd.get_dummies(X, columns=self.categorical_columns,
                               prefix_sep=self.prefix_sep, dummy_na=False)
        # Numeric NaN -> 0 (z-scaled features have mean 0, so this is the
        # natural imputation and avoids breaking sklearn's tree.)
        X_enc = X_enc.reindex(columns=self.dummy_columns, fill_value=0)
        for c in X_enc.columns:
            if X_enc[c].dtype.kind in "fc":
                X_enc[c] = X_enc[c].astype(float).fillna(0.0)
        return X_enc


def fit_feature_encoder(X_raw: pd.DataFrame,
                         prefix_sep: str = "=") -> Tuple[pd.DataFrame, FeatureEncoder]:
    X = X_raw.copy().apply(lambda s: s.map(_clean_value))
    cat_cols = _detect_categorical_columns(X)
    for c in cat_cols:
        X[c] = X[c].astype("string").fillna(MISSING_TOKEN)
    X_enc = pd.get_dummies(X, columns=cat_cols, prefix_sep=prefix_sep, dummy_na=False)
    # Numeric NaN -> 0 (see transform()).
    for c in X_enc.columns:
        if X_enc[c].dtype.kind in "fc":
            X_enc[c] = X_enc[c].astype(float).fillna(0.0)
    enc = FeatureEncoder(raw_columns=list(X_raw.columns),
                         dummy_columns=list(X_enc.columns),
                         categorical_columns=cat_cols,
                         prefix_sep=prefix_sep)
    return X_enc, enc


class FunctionEstimator:
    def __init__(self, model_cfg: Optional[ModelConfig] = None) -> None:
        self.cfg = model_cfg or ModelConfig()

        self.encoder: Optional[FeatureEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_int_to_label: Dict[int, str] = {}

        self.base_model: Optional[Any] = None
        self.surrogate_tree: Optional[DecisionTreeClassifier] = None

        self.base_raw_feature_names: List[str] = []
        self.feature_names: List[str] = []
        self._cb_cat_cols: List[str] = []
        self._priors: Dict[int, float] = {}

        self._train_X_enc: Optional[pd.DataFrame] = None
        self._train_y_int: Optional[np.ndarray] = None
        self._train_base_proba: Optional[np.ndarray] = None

    # artifact serialization
    def to_artifact(self) -> Dict[str, Any]:
        return {"artifact_type": "decision_mining_estimator",
                "artifact_version": 2,
                "estimator_module": __name__,
                "estimator_class": self.__class__.__name__,
                "state": {"cfg": asdict(self.cfg),
                          "encoder": asdict(self.encoder) if self.encoder is not None else None,
                          "label_encoder": self.label_encoder,
                          "class_int_to_label": self.class_int_to_label,
                          "base_model": self.base_model,
                          "surrogate_tree": self.surrogate_tree,
                          "_cb_cat_cols": self._cb_cat_cols,
                          "base_raw_feature_names": self.base_raw_feature_names,
                          "feature_names": self.feature_names,
                          "_priors": self._priors,
                          "_train_X_enc": self._train_X_enc,
                          "_train_y_int": self._train_y_int,
                          "_train_base_proba": self._train_base_proba}}

    @classmethod
    def from_artifact(cls, artifact: Dict[str, Any]) -> "FunctionEstimator":
        state = dict(artifact.get("state", {}))
        cfg_dict = state.pop("cfg", None) or {}
        encoder_dict = state.pop("encoder", None)
        
        # Drop unknown keys (e.g. from a previous artifact_version) - rely on
        # `setattr` for forward-compatible ones.
        known = {"label_encoder", "class_int_to_label", "base_model",
                 "surrogate_tree", "_cb_cat_cols", "base_raw_feature_names",
                 "feature_names", "_priors", "_train_X_enc", "_train_y_int",
                 "_train_base_proba"}
        
        # Filter ModelConfig kwargs to the current dataclass field set.
        cfg_fields = {f for f in ModelConfig.__dataclass_fields__}
        cfg_dict = {k: v for k, v in cfg_dict.items() if k in cfg_fields}
        obj = cls(model_cfg=ModelConfig(**cfg_dict))
        if encoder_dict is not None:
            enc_fields = {f for f in FeatureEncoder.__dataclass_fields__}
            encoder_dict = {k: v for k, v in encoder_dict.items() if k in enc_fields}

            if "categorical_columns" not in encoder_dict:
                sep = encoder_dict.get("prefix_sep", "=")
                raw = encoder_dict.get("raw_columns") or []
                dummies = encoder_dict.get("dummy_columns") or []
                encoder_dict["categorical_columns"] = [
                    c for c in raw if any(d.startswith(f"{c}{sep}") for d in dummies)
                ]
            obj.encoder = FeatureEncoder(**encoder_dict)
        for k, v in state.items():
            if k in known:
                setattr(obj, k, v)
        return obj

    # fit / predict
    @classmethod
    def fit_from_xy(cls, X_raw: pd.DataFrame, y: List[str],
                    model_cfg: Optional[ModelConfig] = None,
                    **_ignored: Any) -> "FunctionEstimator":
        est = cls(model_cfg=model_cfg)
        est.fit(X_raw, y)
        return est

    def fit(self, X_raw: pd.DataFrame, y: List[str]) -> "FunctionEstimator":
        X = X_raw.copy().apply(lambda s: s.map(_clean_value))
        self.base_raw_feature_names = list(X.columns)

        le = LabelEncoder()
        y_int = le.fit_transform(np.asarray(y, dtype=str))
        self.label_encoder = le
        self.class_int_to_label = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
        counts = np.bincount(y_int)
        self._priors = {i: float(counts[i]) / float(max(1, counts.sum()))
                        for i in range(counts.size)}

        # Base predictive model: unweighted by default (see ModelConfig) so it
        # stays calibrated to the true branch frequencies the decoder reweights.
        sw_base = _inverse_freq_weights(y_int) if self.cfg.use_inverse_freq_weights_base else None

        # Base model
        self.base_model = self._fit_catboost(X, y_int, sw_base)

        # Base-model probabilities on training data, used for guard stats.
        X_cb = self._cast_for_catboost(X)
        proba = self.base_model.predict_proba(X_cb)
        # CatBoost may return a 1-D array for binary models in some versions.
        proba = np.atleast_2d(np.asarray(proba, dtype=float))
        if proba.shape[1] == 1 and len(self.class_int_to_label) == 2:
            proba = np.concatenate([1.0 - proba, proba], axis=1)
        self._train_base_proba = proba

        # Surrogate tree on one-hot features.
        X_enc, encoder = fit_feature_encoder(X)
        self.encoder = encoder
        self.feature_names = list(X_enc.columns)
        self._train_X_enc = X_enc
        self._train_y_int = y_int

        if int(np.unique(y_int).size) >= 2 and X_enc.shape[1] > 0:
            sw_sur = _inverse_freq_weights(y_int) if self.cfg.use_inverse_freq_weights else None
            tree = DecisionTreeClassifier(random_state=self.cfg.random_state,
                                          max_depth=self.cfg.surrogate_max_depth,
                                          min_samples_leaf=self.cfg.surrogate_min_samples_leaf,
                                          ccp_alpha=float(self.cfg.surrogate_ccp_alpha))
            tree.fit(X_enc, y_int, sample_weight=sw_sur)
            self.surrogate_tree = tree

        return self

    def _cast_for_catboost(self, X: pd.DataFrame) -> pd.DataFrame:
        cat_cols = _detect_categorical_columns(X)
        self._cb_cat_cols = cat_cols
        if not cat_cols:
            return X
        X_cb = X.copy()
        for c in cat_cols:
            X_cb[c] = X_cb[c].astype("string").fillna(MISSING_TOKEN)
        return X_cb

    def _fit_catboost(self, X: pd.DataFrame, y_int: np.ndarray,
                       sw: Optional[np.ndarray]) -> Any:
        n_classes = int(len(np.unique(y_int)))
        n_classes_total = max(n_classes, int(len(self.class_int_to_label)))

        def _prior_fallback() -> _PriorProbModel:
            cnts = np.bincount(y_int, minlength=n_classes_total).astype(float)
            return _PriorProbModel(cnts / max(1.0, cnts.sum()))

        if n_classes < 2:
            return _prior_fallback()

        try:
            from catboost import CatBoostClassifier
        except Exception as e:
            raise RuntimeError("catboost is not installed. `pip install catboost`") from e

        X_cb = self._cast_for_catboost(X)
        cat_idx = [X_cb.columns.get_loc(c) for c in self._cb_cat_cols]

        # CatBoost errors out if every feature is constant; fall back to prior.
        try:
            if int((X_cb.nunique(dropna=False) > 1).sum()) == 0:
                return _prior_fallback()
        except Exception:
            pass

        loss = "Logloss" if n_classes == 2 else "MultiClass"
        model = CatBoostClassifier(iterations=int(self.cfg.cb_iterations),
                                   learning_rate=float(self.cfg.cb_learning_rate),
                                   depth=int(self.cfg.cb_depth),
                                   l2_leaf_reg=float(self.cfg.cb_l2_leaf_reg),
                                   min_data_in_leaf=int(self.cfg.cb_min_data_in_leaf),
                                   loss_function=loss,
                                   random_seed=int(self.cfg.random_state),
                                   allow_writing_files=False,
                                   verbose=False)

        class_counts = np.bincount(y_int)
        nonzero = class_counts[class_counts > 0]
        can_split = (
            float(self.cfg.cb_eval_fraction) > 0.0
            and len(y_int) >= 50
            and (int(nonzero.min()) if nonzero.size else 0) >= 2
        )

        if can_split:
            idx = np.arange(len(y_int))
            idx_fit, idx_eval = train_test_split(
                idx, test_size=float(self.cfg.cb_eval_fraction),
                random_state=int(self.cfg.random_state), stratify=y_int,
            )
            model.fit(X_cb.iloc[idx_fit], y_int[idx_fit],
                      cat_features=cat_idx,
                      sample_weight=None if sw is None else sw[idx_fit],
                      eval_set=(X_cb.iloc[idx_eval], y_int[idx_eval]),
                      use_best_model=True,
                      early_stopping_rounds=int(self.cfg.cb_early_stopping_rounds))
        else:
            model.fit(X_cb, y_int, cat_features=cat_idx, sample_weight=sw)
        return model

    def predict_proba(self, assignment: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
        if self.base_model is None:
            raise RuntimeError("Estimator is not fitted.")
        X_raw = pd.DataFrame([assignment]).apply(lambda s: s.map(_clean_value))
        for col in self.base_raw_feature_names:
            if col not in X_raw.columns:
                X_raw[col] = np.nan
        X_raw = X_raw[self.base_raw_feature_names]

        if getattr(self.base_model, "_is_prior_model", False):
            proba = self.base_model.predict_proba(X_raw)[0]
        else:
            X_cb = X_raw.copy()
            for c in self._cb_cat_cols:
                if c in X_cb.columns:
                    X_cb[c] = X_cb[c].astype("string").fillna(MISSING_TOKEN)
            # Any leftover non-categorical column with stray strings -> NaN
            num_cols = [c for c in X_cb.columns if c not in self._cb_cat_cols]
            for c in num_cols:
                X_cb[c] = pd.to_numeric(X_cb[c], errors="coerce")
            proba = self.base_model.predict_proba(X_cb)[0]

        class_ints = list(getattr(self.base_model, "classes_", np.arange(len(proba))))
        labels = [self.class_int_to_label.get(int(c), str(c)) for c in class_ints]
        return labels, np.asarray(proba, dtype=float)

    # guard extraction
    def extract_guards(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Walk the surrogate tree and emit one guard per (label, leaf) above the
        configured thresholds. Returns ``{label: [guard, ...]}``.
        """
        if (self.encoder is None
                or self._train_X_enc is None
                or self._train_y_int is None
                or self._train_base_proba is None):
            raise RuntimeError("Estimator is not fitted.")

        all_labels = list(self.class_int_to_label.values())

        # No surrogate (single class or empty features) -> emit a single (true)
        # guard per label populated from the training distribution.
        if self.surrogate_tree is None:
            P = self._train_base_proba
            mu = P.mean(axis=0)
            counts = np.bincount(self._train_y_int, minlength=len(all_labels)).astype(float)
            total = float(counts.sum()) or 1.0
            out: Dict[str, List[Dict[str, Any]]] = {}
            for c_int, lab in self.class_int_to_label.items():
                pm = float(mu[c_int])
                if pm < float(self.cfg.min_leaf_prob):
                    out[lab] = []
                    continue
                out[lab] = [{
                    "rule": "(true)",
                    "raw_rule": "(true)",
                    "intervals": {},
                    "categorical_allowed": {},
                    "categorical_excluded": {},
                    "prob_model": pm,
                    "prob_emp": float(counts[c_int] / total),
                    "support": int(total),
                    "coverage": 1.0,
                    "lift": 1.0,
                    "score": float(pm * math.log1p(total)),
                }]
            return out

        # Leaf-level statistics from base-model probabilities.
        tree = self.surrogate_tree
        X_enc = self._train_X_enc
        P = self._train_base_proba
        leaf_ids = tree.apply(X_enc)
        n_classes = P.shape[1]
        total_n = int(len(self._train_y_int))

        leaf_stats: Dict[int, Dict[str, Any]] = {}
        for leaf in np.unique(leaf_ids):
            idx = np.where(leaf_ids == leaf)[0]
            support = int(idx.size)
            if support <= 0:
                continue
            mu = P[idx, :].mean(axis=0)
            counts = np.bincount(self._train_y_int[idx], minlength=n_classes).astype(float)
            n = float(counts.sum()) or 1.0
            leaf_stats[int(leaf)] = {"support": support,
                                     "prob_model": mu,
                                     "prob_emp": counts / n}

        rules_by_label = self._walk_tree(leaf_stats)

        cond_re = re.compile(
            r"\((.*?)\s*(<=|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)"
        )
        prefix_sep = self.encoder.prefix_sep
        out: Dict[str, List[Dict[str, Any]]] = {}

        for label, rules in rules_by_label.items():
            guards: List[Dict[str, Any]] = []
            for ro in rules:
                raw = str(ro["rule"])
                intervals: Dict[str, Dict[str, Optional[float]]] = {}
                cat_inc_exc: Dict[str, Dict[str, List[str]]] = {}

                if raw != "(true)":
                    for feat, op, num_s in cond_re.findall(raw):
                        try:
                            thresh = float(num_s)
                        except ValueError:
                            continue
                        if prefix_sep in feat:
                            base, value = feat.split(prefix_sep, 1)
                            if str(value) == MISSING_TOKEN:
                                continue
                            entry = cat_inc_exc.setdefault(base, {"include": [], "exclude": []})
                            # Dummies live in [0, 1]; allow a small slack for ties.
                            if -0.1 <= thresh <= 1.1:
                                target = entry["include"] if op == ">" else entry["exclude"]
                                target.append(str(value))
                            continue
                        iv = intervals.setdefault(feat, {"low": None, "high": None})
                        if op == ">":
                            iv["low"] = thresh if iv["low"] is None else max(float(iv["low"]), thresh)
                        else:
                            iv["high"] = thresh if iv["high"] is None else min(float(iv["high"]), thresh)

                cat_allowed = {b: sorted(set(ie["include"])) for b, ie in cat_inc_exc.items() if ie.get("include")}
                cat_excluded = {b: sorted(set(ie["exclude"])) for b, ie in cat_inc_exc.items() if ie.get("exclude")}
                rule_str = _simplify_rule(intervals, cat_allowed, cat_excluded)

                support = int(ro["support"])
                lift = float(ro["lift"])
                pm = float(ro["prob_model"])
                pe = float(ro["prob_emp"])
                score = pm * math.log1p(support) * max(1.0, lift)
                guards.append({"rule": rule_str,
                               "raw_rule": raw,
                               "intervals": intervals,
                               "categorical_allowed": cat_allowed,
                               "categorical_excluded": cat_excluded,
                               "prob_model": pm,
                               "prob_emp": pe,
                               "support": support,
                               "coverage": float(support) / float(max(1, total_n)),
                               "lift": lift,
                               "score": float(score)})

            guards.sort(key=lambda g: (g["score"], g["prob_model"]), reverse=True)
            k = int(self.cfg.max_rules_per_label)
            if k > 0:
                guards = guards[:k]
            out[label] = guards
        return out

    def _walk_tree(self,
                   leaf_stats: Dict[int, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        assert self.surrogate_tree is not None
        t = self.surrogate_tree.tree_
        FEATURE_UNDEF = -2
        out: Dict[str, List[Dict[str, Any]]] = {lbl: [] for lbl in self.class_int_to_label.values()}

        def cond(feat_idx: int, thresh: float, direction: str) -> str:
            op = "<=" if direction == "left" else ">"
            return f"({self.feature_names[feat_idx]} {op} {thresh:.6g})"

        stack: List[Tuple[int, List[str]]] = [(0, [])]
        while stack:
            node_id, path = stack.pop()
            feat_idx = int(t.feature[node_id])
            if feat_idx == FEATURE_UNDEF:
                stats = leaf_stats.get(int(node_id))
                if stats is None:
                    continue
                support = int(stats["support"])
                if support < int(self.cfg.min_leaf_support):
                    continue
                rule_str = " AND ".join(path) if path else "(true)"
                for c_int, lab in self.class_int_to_label.items():
                    pm = float(stats["prob_model"][c_int])
                    prior = float(self._priors.get(int(c_int), 1e-12))
                    lift = pm / max(prior, 1e-12)

                    if pm >= float(self.cfg.min_leaf_prob) and lift >= float(self.cfg.min_leaf_lift):
                        out[lab].append({"rule": rule_str,
                                         "support": support,
                                         "lift": float(lift),
                                         "prob_model": pm,
                                         "prob_emp": float(stats["prob_emp"][c_int])})
                continue
            thresh = float(t.threshold[node_id])
            left_id = int(t.children_left[node_id])
            right_id = int(t.children_right[node_id])
            stack.append((right_id, path + [cond(feat_idx, thresh, "right")]))
            stack.append((left_id, path + [cond(feat_idx, thresh, "left")]))
        return out


def _simplify_rule(intervals: Dict[str, Dict[str, Optional[float]]],
                    cat_allowed: Dict[str, List[str]],
                    cat_excluded: Dict[str, List[str]],
                    *, max_show: int = 5) -> str:
    parts: List[str] = []
    for base in sorted(set(cat_allowed.keys()) | set(cat_excluded.keys())):
        inc = cat_allowed.get(base) or []
        exc = cat_excluded.get(base) or []
        if inc:
            shown = inc[:max_show]
            suffix = f" (+{len(inc) - len(shown)} more)" if len(inc) > len(shown) else ""
            parts.append(f"({base} in {{{', '.join(shown)}}}{suffix})")
        elif exc:
            shown = exc[:max_show]
            suffix = f" (+{len(exc) - len(shown)} more)" if len(exc) > len(shown) else ""
            parts.append(f"({base} not in {{{', '.join(shown)}}}{suffix})")
    for feat in sorted(intervals):
        iv = intervals[feat] or {}
        if iv.get("low") is not None:
            parts.append(f"({feat} > {float(iv['low']):.6g})")
        if iv.get("high") is not None:
            parts.append(f"({feat} <= {float(iv['high']):.6g})")
    return " AND ".join(parts) if parts else "(true)"
