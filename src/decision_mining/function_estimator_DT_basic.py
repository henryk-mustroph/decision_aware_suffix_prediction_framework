"""
Decision Tree model training and decision guard mining (improved)
- Mixed numeric + categorical
- High-card categorical handled via Top-K + __OTHER__
- Optional rare bucketing
- Cost-complexity pruning tuned via CV (guard-friendly compact trees)
- Probabilistic guards with Wilson CI, lift, support
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re
import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from pandas.api.types import is_integer_dtype, is_float_dtype

@dataclass
class ModelConfig:
    # Tree shape / stability
    random_state: int = 7
    max_depth: Optional[int] = 6
    min_samples_leaf: int = 20

    # "Optimal tree" via pruning
    tune_ccp_alpha: bool = True
    pruning_cv: int = 5
    pruning_max_alphas: int = 25
    pruning_metric: str = "auto"  # "auto" | "prauc" (binary) | "f1_macro" (multiclass)

    # Imbalance handling
    use_inverse_freq_weights: bool = True

    # Encoding behavior
    treat_int_as_categorical: bool = True

    # Categorical bucketing (critical for 500+ categories)
    low_card_max: int = 30              # <= this: keep categories (optionally rare-bucket)
    high_card_top_k: int = 50           # > low_card_max: keep top-k, rest -> __OTHER__
    min_category_freq: int = 2          # bucket infrequent values -> __RARE__ (for low-card)
    include_missing_token: bool = True  # represent missing as __MISSING__ (categorical)

    # Guards output controls
    include_missing_in_guards: bool = False
    max_rules_per_label: int = 12
    rule_sort: str = "score"  # "score" | "prob" | "lift" | "support"

    # Prob output
    calibrate: bool = True
    calibration_method: str = "sigmoid"
    calibration_cv: int = 5
    guard_ci: float = 0.95


# ----------------------------
# Helpers
# ----------------------------
MISSING_TOKEN = "__MISSING__"
OTHER_TOKEN = "__OTHER__"
RARE_TOKEN = "__RARE__"


def _clean_value(v: Any) -> Any:
    """Keep scalars; stringify non-scalars. Safe for pandas + dummy encoding."""
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


def _wilson_interval(k: float, n: float, ci: float = 0.95) -> Tuple[float, float]:
    """
    Wilson score interval for a Bernoulli proportion, no SciPy needed.
    Interpreted as one-vs-rest for multiclass leaf counts.
    """
    if n <= 0:
        return 0.0, 1.0
    z = NormalDist().inv_cdf(1.0 - (1.0 - float(ci)) / 2.0)
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2.0 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) + (z * z) / (4.0 * n)) / n)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return float(lo), float(hi)


# ----------------------------
# Encoding
# ----------------------------
@dataclass
class FeatureEncoder:
    raw_columns: List[str]
    dummy_columns: List[str]
    prefix_sep: str
    cat_info: Dict[str, Dict[str, Any]]        # per categorical col: kept values, tokens
    categorical_levels: Dict[str, List[str]]   # levels that appear as dummies (after bucketing)

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X = X_raw.reindex(columns=self.raw_columns).copy()
        X = X.apply(lambda s: s.map(_clean_value))

        # apply same bucketing as fit
        for col, info in self.cat_info.items():
            if col not in X.columns:
                continue
            s = X[col].astype("string")
            if info.get("use_missing_token", True):
                s = s.fillna(info.get("missing_token", MISSING_TOKEN))
            kept = info.get("kept", set())
            other_token = info.get("other_token", OTHER_TOKEN)
            rare_token = info.get("rare_token", RARE_TOKEN)

            # First, map unseen to OTHER
            s = s.where(s.isin(list(kept)), other=other_token)

            # Rare token is only used when we explicitly created it (low-card rare-bucketing)
            if info.get("rare_values") is not None:
                rare_values = set(info["rare_values"])
                s = s.where(~s.isin(list(rare_values)), other=rare_token)

            X[col] = s

        X_enc = pd.get_dummies(X, dummy_na=False, prefix_sep=self.prefix_sep)
        return X_enc.reindex(columns=self.dummy_columns, fill_value=0)


def _detect_categorical_columns(X: pd.DataFrame, treat_int_as_categorical: bool) -> List[str]:
    cat_cols: List[str] = []
    for c in X.columns:
        s = X[c]
        if pd.api.types.is_bool_dtype(s.dtype):
            cat_cols.append(c)
            continue
        if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
            cat_cols.append(c)
            continue
        if treat_int_as_categorical and is_integer_dtype(s.dtype) and not is_float_dtype(s.dtype):
            cat_cols.append(c)
            continue
    return cat_cols


def fit_feature_encoder(
    X_raw: pd.DataFrame,
    model_cfg: ModelConfig,
    prefix_sep: str = "=",
) -> Tuple[pd.DataFrame, FeatureEncoder]:
    X = X_raw.copy()
    X = X.apply(lambda s: s.map(_clean_value))

    raw_cols = list(X.columns)
    cat_cols = _detect_categorical_columns(X, bool(model_cfg.treat_int_as_categorical))

    cat_info: Dict[str, Dict[str, Any]] = {}
    X_work = X.copy()

    for col in cat_cols:
        s = X_work[col].astype("string")
        use_missing = bool(model_cfg.include_missing_token)
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

        if nunique <= int(model_cfg.low_card_max):
            # low-card: keep categories, but optionally bucket very rare values to __RARE__
            min_freq = int(model_cfg.min_category_freq)
            if min_freq > 1:
                rare_values = vc[vc < min_freq].index.astype(str).tolist()
                if rare_values:
                    info["rare_values"] = rare_values
                    # kept includes everything (we still one-hot __RARE__)
                    kept = set(vc.index.astype(str).tolist())
                else:
                    kept = set(vc.index.astype(str).tolist())
            else:
                kept = set(vc.index.astype(str).tolist())

            info["kept"] = kept

            # Apply rare bucketing if needed
            if info["rare_values"] is not None:
                rv = set(info["rare_values"])
                s = s.where(~s.isin(list(rv)), other=RARE_TOKEN)

        else:
            # high-card: keep top-k, everything else -> __OTHER__
            top_k = int(model_cfg.high_card_top_k)
            kept_list = vc.head(top_k).index.astype(str).tolist()
            kept = set(kept_list)
            # keep also missing token explicitly if enabled and present
            if use_missing and MISSING_TOKEN in vc.index.astype(str).tolist():
                kept.add(MISSING_TOKEN)
            kept.add(OTHER_TOKEN)
            info["kept"] = kept

            s = s.where(s.isin(list(kept_list + ([MISSING_TOKEN] if use_missing else []))), other=OTHER_TOKEN)

        X_work[col] = s
        cat_info[col] = info

    X_enc = pd.get_dummies(X_work, dummy_na=False, prefix_sep=prefix_sep)

    # infer categorical levels present as dummies for guard reconstruction
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


# ----------------------------
# Tree pruning selection ("optimal tree")
# ----------------------------
def _score_for_pruning(
    y_true: np.ndarray,
    proba_pos: Optional[np.ndarray],
    y_pred: np.ndarray,
    metric: str,
) -> float:
    if metric == "prauc":
        # binary only
        if proba_pos is None:
            return -1e9
        return float(average_precision_score(y_true, proba_pos))
    elif metric == "f1_macro":
        return float(f1_score(y_true, y_pred, average="macro"))
    else:
        # fallback
        return float(f1_score(y_true, y_pred, average="macro"))


def select_ccp_alpha_cv(
    X: pd.DataFrame,
    y_int: np.ndarray,
    model_cfg: ModelConfig,
    sample_weight: Optional[np.ndarray],
) -> float:
    """
    Choose ccp_alpha by CV, to get compact + accurate tree.
    Uses pruning path alphas, subsampled to pruning_max_alphas.
    """
    # base tree for pruning path
    base = DecisionTreeClassifier(
        random_state=model_cfg.random_state,
        max_depth=model_cfg.max_depth,
        min_samples_leaf=model_cfg.min_samples_leaf,
        ccp_alpha=0.0,
    )
    path = base.cost_complexity_pruning_path(X, y_int, sample_weight=sample_weight)
    alphas = np.unique(path.ccp_alphas.astype(float))
    if alphas.size == 0:
        return 0.0

    # reduce candidates (ignore last alpha often collapses to root)
    alphas = np.unique(alphas[:-1]) if alphas.size > 1 else alphas
    if alphas.size == 0:
        return 0.0

    max_cand = int(model_cfg.pruning_max_alphas)
    if alphas.size > max_cand:
        qs = np.linspace(0.0, 1.0, max_cand)
        alphas = np.unique(np.quantile(alphas, qs))

    # CV
    n = int(len(y_int))
    counts = np.bincount(y_int)
    min_class = int(counts[counts > 0].min()) if (counts > 0).any() else 0
    cv = int(min(int(model_cfg.pruning_cv), n, min_class))
    if cv < 2:
        return 0.0

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=model_cfg.random_state)

    best_alpha = float(alphas[0])
    best_score = -1e18

    binary = (len(np.unique(y_int)) == 2)

    metric_cfg = str(getattr(model_cfg, "pruning_metric", "auto")).lower().strip()
    if metric_cfg == "auto":
        metric = "prauc" if binary else "f1_macro"
    else:
        metric = metric_cfg
        # safety: PR-AUC is only meaningful for binary
        if metric == "prauc" and not binary:
            metric = "f1_macro"

    for a in alphas.tolist():
        scores: List[float] = []
        for tr, te in skf.split(X, y_int):
            clf = DecisionTreeClassifier(
                random_state=model_cfg.random_state,
                max_depth=model_cfg.max_depth,
                min_samples_leaf=model_cfg.min_samples_leaf,
                ccp_alpha=float(a),
            )
            sw_tr = sample_weight[tr] if sample_weight is not None else None
            clf.fit(X.iloc[tr], y_int[tr], sample_weight=sw_tr)

            proba = None
            if binary and metric == "prauc":
                try:
                    proba = clf.predict_proba(X.iloc[te])[:, 1]
                except Exception:
                    proba = None
            y_hat = clf.predict(X.iloc[te])
            sc = _score_for_pruning(y_int[te], proba, y_hat, metric)
            scores.append(sc)

        mean_sc = float(np.mean(scores)) if scores else -1e18
        if mean_sc > best_score:
            best_score = mean_sc
            best_alpha = float(a)

    return float(best_alpha)


# ----------------------------
# Estimator
# ----------------------------
class FunctionEstimator:
    def __init__(self, model_cfg: Optional[ModelConfig] = None) -> None:
        self.model_cfg = model_cfg or ModelConfig()

        self.encoder: Optional[FeatureEncoder] = None
        self.label_encoder: Optional[LabelEncoder] = None
        self.class_int_to_label: Dict[int, str] = {}
        self.clf: Optional[Any] = None
        self._base_clf: Optional[Any] = None
        self.feature_names: List[str] = []

        self._train_X_enc: Optional[pd.DataFrame] = None
        self._train_y_int: Optional[np.ndarray] = None
        self._priors: Dict[int, float] = {}

    @classmethod
    def fit_from_xy(
        cls,
        X_raw: pd.DataFrame,
        y: List[str],
        feature_cols: Optional[List[str]] = None,
        model_cfg: Optional[ModelConfig] = None,
    ) -> "FunctionEstimator":
        est = cls(model_cfg=model_cfg)
        est.fit(X_raw, y, feature_cols=feature_cols)
        return est

    def fit(self, X_raw: pd.DataFrame, y: List[str], feature_cols: Optional[List[str]] = None) -> "FunctionEstimator":
        X_work = X_raw.copy()

        if feature_cols:
            for c in feature_cols:
                if c not in X_work.columns:
                    X_work[c] = np.nan
            X_work = X_work[feature_cols]

        # Encode labels
        le = LabelEncoder()
        y_int_full = le.fit_transform(np.asarray(y, dtype=str))
        class_int_to_label = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}
        n_classes_total = int(len(le.classes_))

        self._priors = _priors_from_ints(y_int_full)

        # Fit encoder on full training rows
        X_enc_train, encoder = fit_feature_encoder(X_work, self.model_cfg, prefix_sep="=")

        # Also encode full for calibration & priors / guard stats
        X_enc_full = encoder.transform(X_work)

        # Sample weights (preferred imbalance handling, stays consistent for CV + pruning)
        sample_weight_train = None
        if bool(self.model_cfg.use_inverse_freq_weights):
            sample_weight_train = compute_inverse_freq_weights(y_int_full)

        # Choose pruning alpha by CV (optional)
        chosen_alpha = float(getattr(self.model_cfg, "ccp_alpha", 0.0))
        if bool(self.model_cfg.tune_ccp_alpha):
            try:
                chosen_alpha = select_ccp_alpha_cv(
                    X_enc_train,
                    y_int_full,
                    self.model_cfg,
                    sample_weight_train,
                )
            except Exception:
                chosen_alpha = float(getattr(self.model_cfg, "ccp_alpha", 0.0))

        # Train tree
        base_clf = DecisionTreeClassifier(
            random_state=self.model_cfg.random_state,
            max_depth=self.model_cfg.max_depth,
            min_samples_leaf=self.model_cfg.min_samples_leaf,
            ccp_alpha=float(chosen_alpha),
        )
        base_clf.fit(X_enc_train, y_int_full, sample_weight=sample_weight_train)

        # Calibrate (on FULL data for stable probabilities; uses automatic cv fallback)
        clf: Any = base_clf
        if bool(self.model_cfg.calibrate) and n_classes_total >= 2:
            n_samples = int(len(y_int_full))
            counts = np.bincount(y_int_full) if n_samples else np.asarray([], dtype=int)
            nonzero = counts[counts > 0]
            min_class_count = int(nonzero.min()) if nonzero.size else 0
            cv_max = int(getattr(self.model_cfg, "calibration_cv", 5))
            cv = int(min(cv_max, n_samples, min_class_count))
            if cv >= 2:
                cal = CalibratedClassifierCV(
                    base_clf,
                    method=str(self.model_cfg.calibration_method),
                    cv=cv,
                )
                # weights for calibration: use inverse-freq on FULL distribution
                sw_full = compute_inverse_freq_weights(y_int_full) if bool(self.model_cfg.use_inverse_freq_weights) else None
                try:
                    cal.fit(X_enc_full, y_int_full, sample_weight=sw_full)
                    clf = cal
                except ValueError:
                    # Degenerate calibration folds (or single-class behavior in
                    # cloned estimators) can fail with predict_proba shape
                    # mismatches. Fall back to the already fitted base tree.
                    clf = base_clf
            else:
                clf = base_clf

        # Store
        self.encoder = encoder
        self.label_encoder = le
        self.class_int_to_label = class_int_to_label
        self._base_clf = base_clf
        self.clf = clf
        self.feature_names = list(X_enc_full.columns)
        self._train_X_enc = X_enc_full
        self._train_y_int = y_int_full

        return self

    def predict_proba(self, assignment: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
        if self.encoder is None or self.clf is None:
            raise RuntimeError("Estimator is not fitted.")
        X_raw = pd.DataFrame([assignment])
        X_enc = self.encoder.transform(X_raw)
        proba = self.clf.predict_proba(X_enc)[0]

        base = self._base_clf or self.clf
        class_ints = list(getattr(base, "classes_", np.arange(len(proba))))
        labels = [self.class_int_to_label[int(c)] for c in class_ints]
        return labels, np.asarray(proba, dtype=float)

    # ----------------------------
    # Guards (probabilistic + CI)
    # ----------------------------
    def extract_probabilistic_guards(
        self,
        *,
        min_leaf_prob: float = 0.2,
        min_leaf_lift: float = 2.0,
        min_leaf_support: int = 20,
        always_keep_best: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Returns per label a list of guard objects:
          - rule (human string, set/range)
          - intervals (numeric ranges)
          - categorical (allowed/excluded sets)
          - prob_mean, prob_ci_low, prob_ci_high (one-vs-rest Wilson)
          - support, coverage, lift
          - raw_rule (tree path, dummy-level)
        """
        if self.encoder is None or self._base_clf is None:
            raise RuntimeError("Estimator is not fitted.")
        tree = self._base_clf

        rules_by_label = extract_tree_guards_probabilistic(
            tree,
            feature_names=self.feature_names,
            class_int_to_label=self.class_int_to_label,
            priors=self._priors,
            min_leaf_prob=min_leaf_prob,
            min_leaf_lift=min_leaf_lift,
            min_leaf_support=min_leaf_support,
            always_keep_best=always_keep_best,
            ci=float(self.model_cfg.guard_ci),
        )

        # Parse and simplify:
        cond_re = re.compile(
            r"\((.*?)\s*(<=|>)\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\)"
        )

        total_n = int(self._train_y_int.shape[0]) if self._train_y_int is not None else 1

        out: Dict[str, List[Dict[str, Any]]] = {}
        for label, rule_objs in rules_by_label.items():
            guards: List[Dict[str, Any]] = []
            for ro in rule_objs:
                raw = str(ro["rule"])
                support = int(ro["support"])
                coverage = float(support) / float(max(1, total_n))

                # one-vs-rest prob stats
                p = float(ro["prob"])
                lo = float(ro["ci_low"])
                hi = float(ro["ci_high"])
                lift = float(ro["lift"])

                intervals: Dict[str, Dict[str, Optional[float]]] = {}
                cat_inc_exc: Dict[str, Dict[str, List[str]]] = {}

                if raw.strip() != "(true)":
                    for feat, op, num_str in cond_re.findall(raw):
                        try:
                            thresh = float(num_str)
                        except ValueError:
                            continue

                        # categorical dummy?
                        if self.encoder.prefix_sep in feat:
                            base, value = feat.split(self.encoder.prefix_sep, 1)

                            # optionally drop missing token from guard output
                            if (not bool(self.model_cfg.include_missing_in_guards)) and (str(value) == MISSING_TOKEN):
                                continue

                            entry = cat_inc_exc.setdefault(base, {"include": [], "exclude": []})

                            # Interpret dummy split. For 0/1 dummies, op alone is sufficient.
                            # We also guard against weird thresholds:
                            if -0.1 <= thresh <= 1.1:
                                if op == ">":
                                    if value not in entry["include"]:
                                        entry["include"].append(str(value))
                                else:
                                    if value not in entry["exclude"]:
                                        entry["exclude"].append(str(value))
                            continue

                        # numeric interval
                        iv = intervals.setdefault(feat, {"low": None, "high": None})
                        if op == ">":
                            iv["low"] = thresh if iv["low"] is None else max(float(iv["low"]), thresh)
                        else:
                            iv["high"] = thresh if iv["high"] is None else min(float(iv["high"]), thresh)

                # Convert include/exclude dummies into allowed sets when we can
                cat_allowed: Dict[str, List[str]] = {}
                cat_excluded: Dict[str, List[str]] = {}

                for base, inc_exc in cat_inc_exc.items():
                    inc = sorted(set(inc_exc.get("include", [])))
                    exc = sorted(set(inc_exc.get("exclude", [])))

                    # all levels known?
                    all_levels = self.encoder.categorical_levels.get(base, None)
                    if all_levels is not None and len(all_levels) > 0:
                        all_set = set(map(str, all_levels))
                        if inc:
                            allowed = sorted(set(inc))
                        else:
                            allowed = sorted(all_set - set(exc))
                        # drop missing token if requested
                        if not bool(self.model_cfg.include_missing_in_guards):
                            allowed = [v for v in allowed if v != MISSING_TOKEN]
                            exc = [v for v in exc if v != MISSING_TOKEN]
                        cat_allowed[base] = allowed
                        cat_excluded[base] = sorted(set(exc))
                    else:
                        # fallback: keep raw include/exclude
                        cat_allowed[base] = inc
                        cat_excluded[base] = exc

                guard_str = _simplify_guard_rule(intervals, cat_allowed, cat_excluded)

                guards.append(
                    {
                        "rule": guard_str,
                        "raw_rule": raw,
                        "intervals": intervals,
                        "categorical_allowed": cat_allowed,
                        "categorical_excluded": cat_excluded,
                        "prob": p,
                        "prob_ci_low": lo,
                        "prob_ci_high": hi,
                        "support": support,
                        "coverage": coverage,
                        "lift": lift,
                        "score": (p * math.log1p(support) * max(1.0, lift)),
                    }
                )

            # sort + cap
            sort_key = str(getattr(self.model_cfg, "rule_sort", "score"))
            if sort_key == "prob":
                guards.sort(key=lambda g: (g["prob"], g["support"]), reverse=True)
            elif sort_key == "lift":
                guards.sort(key=lambda g: (g["lift"], g["support"]), reverse=True)
            elif sort_key == "support":
                guards.sort(key=lambda g: (g["support"], g["prob"]), reverse=True)
            else:
                guards.sort(key=lambda g: (g["score"], g["prob"]), reverse=True)

            k = int(getattr(self.model_cfg, "max_rules_per_label", 12))
            if k > 0:
                guards = guards[:k]

            out[label] = guards

        return out


# ----------------------------
# Guard rendering
# ----------------------------
def _simplify_guard_rule(
    intervals: Dict[str, Dict[str, Optional[float]]],
    cat_allowed: Dict[str, List[str]],
    cat_excluded: Dict[str, List[str]],
    *,
    max_set_show: int = 18,
) -> str:
    parts: List[str] = []

    # Categorical constraints (prefer allowed-set form)
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

    # Numeric intervals
    for feat in sorted(intervals.keys()):
        iv = intervals[feat] or {}
        low = iv.get("low")
        high = iv.get("high")
        if low is not None:
            parts.append(f"({feat} > {float(low):.6g})")
        if high is not None:
            parts.append(f"({feat} <= {float(high):.6g})")

    return " AND ".join(parts) if parts else "(true)"


# ----------------------------
# Core rule extraction
# ----------------------------
def extract_tree_guards_probabilistic(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    class_int_to_label: Dict[int, str],
    priors: Dict[int, float],
    *,
    min_leaf_prob: float = 0.2,
    min_leaf_lift: float = 2.0,
    min_leaf_support: int = 20,
    always_keep_best: bool = True,
    ci: float = 0.95,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Per label returns rules with:
      - rule (path string in terms of feature thresholds)
      - prob (leaf prob for that class, Laplace-smoothed)
      - ci_low/ci_high (Wilson interval one-vs-rest from raw counts)
      - support (n samples in leaf)
      - lift (prob / prior)
    """
    tree = clf.tree_
    feature_undef = -2

    def cond_str(feat_idx: int, thresh: float, direction: str) -> str:
        fname = feature_names[feat_idx]
        return f"({fname} <= {thresh:.6g})" if direction == "left" else f"({fname} > {thresh:.6g})"

    class_ints = [int(c) for c in clf.classes_]
    labels = [class_int_to_label[i] for i in class_ints]

    rules_by_label: Dict[str, List[Dict[str, Any]]] = {lab: [] for lab in labels}
    best_by_label: Dict[str, Dict[str, Any]] = {
        lab: {"prob": -1.0, "support": 0, "rule": "(false)", "lift": 0.0, "ci_low": 0.0, "ci_high": 1.0}
        for lab in labels
    }

    stack: List[Tuple[int, List[str]]] = [(0, [])]
    while stack:
        node_id, path_conds = stack.pop()
        feat_idx = int(tree.feature[node_id])

        if feat_idx == feature_undef:
            support = int(tree.n_node_samples[node_id])
            if support < int(min_leaf_support):
                continue

            counts = tree.value[node_id][0].astype(float)
            n = float(counts.sum())

            # Laplace-smoothed probabilities
            probs = (counts + 1.0) / (n + float(len(counts)))
            rule_str = " AND ".join(path_conds) if path_conds else "(true)"

            for j, c_int in enumerate(class_ints):
                lab = class_int_to_label[int(c_int)]
                p = float(probs[j])
                prior = float(priors.get(int(c_int), 1e-12))
                lift = p / max(prior, 1e-12)

                # Wilson CI from raw one-vs-rest count
                k = float(counts[j])
                ci_low, ci_high = _wilson_interval(k, n, ci=ci)

                if p > float(best_by_label[lab]["prob"]):
                    best_by_label[lab] = {
                        "prob": p,
                        "support": support,
                        "rule": rule_str,
                        "lift": lift,
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }

                if (p >= float(min_leaf_prob)) or (lift >= float(min_leaf_lift)):
                    rules_by_label[lab].append(
                        {
                            "rule": rule_str,
                            "prob": p,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "support": support,
                            "lift": lift,
                        }
                    )
            continue

        thresh = float(tree.threshold[node_id])
        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])

        stack.append((right_id, path_conds + [cond_str(feat_idx, thresh, "right")]))
        stack.append((left_id, path_conds + [cond_str(feat_idx, thresh, "left")]))

    if always_keep_best:
        for lab in labels:
            if not rules_by_label[lab]:
                best = best_by_label[lab]
                if best["prob"] > 0 and best["support"] >= int(min_leaf_support):
                    rules_by_label[lab] = [best]

    return rules_by_label
