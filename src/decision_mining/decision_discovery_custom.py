"""
Decision-aware decision mining for decision-aware suffix prediction.

Base idea adopted from:
- De Leoni, M., & Van Der Aalst, W. M. (2013). Data-aware process mining: discovering decisions in processes using alignments.

This implementation:
- Structures decision points into XOR/loop patterns
- Uses alignment-based extraction of training samples per decision point
- Trains stronger classifiers per decision point (CatBoost/LightGBM/XGBoost/sklearn fallback)
- Builds guards suitable for suffix decoding via probabilistic masking
- Optionally distills model into a small surrogate decision tree to get readable rules (DNF-ish)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from pm4py.algo.decision_mining import algorithm as dm_alg  


class DecisionPattern(str, Enum):
    XOR_1V1T = "xor_1_visible_1_tau"          # a) one visible + one tau
    XOR_NV1T = "xor_n_visible_1_tau"          # b) >=2 visible + one tau
    XOR_GENERAL = "xor_general"               # other XORs
    LOOP_OUTER = "loop_outer"                 # c) outer loop SCC
    LOOP_INNER = "loop_inner"                 # d) inner loop SCC
    LOOP_SELF = "loop_self"                   # e) self-loop SCC
    SKIP_ONLY_TAU = "skip_only_tau"           # only tau outgoing => store but do not train

@dataclass
class DecisionPointSpec:
    place_name: str
    pattern: DecisionPattern
    outgoing_transitions: List[Any]           # PM4Py Transition objects
    outgoing_branch_labels: List[str]         # visible label or tau::<t.name>
    visible_labels: List[str]                 # transition.label != None
    tau_labels: List[str]                     # tau::<t.name>

    # loop metadata (optional)
    loop_scc_id: Optional[int] = None
    loop_continue_branches: Optional[List[str]] = None
    loop_exit_branches: Optional[List[str]] = None


# Make class here: To get all relevant decision points with potntial next transitions

def _is_tau_transition(t) -> bool:
    return getattr(t, "label", None) is None

def _branch_label_for_transition(t) -> str:
    # stable ID for branch classification + decoding
    if t.label is None:
        return f"tau::{t.name}"
    return str(t.label)

def _build_transition_graph(net) -> Dict[Any, Set[Any]]:
    """
    Transition adjacency:
      t -> t2 if exists place p with t -> p -> t2
    """
    adj: Dict[Any, Set[Any]] = {t: set() for t in net.transitions}
    for t in net.transitions:
        for a in t.out_arcs:
            p = a.target
            for a2 in p.out_arcs:
                t2 = a2.target
                adj[t].add(t2)
    return adj

def _tarjan_scc(adj: Dict[Any, Set[Any]]) -> Tuple[Dict[Any, int], List[List[Any]]]:
    index = 0
    stack: List[Any] = []
    onstack: Set[Any] = set()
    idx: Dict[Any, int] = {}
    low: Dict[Any, int] = {}
    sccs: List[List[Any]] = []
    node_to_scc: Dict[Any, int] = {}

    def strongconnect(v):
        nonlocal index
        idx[v] = index
        low[v] = index
        index += 1
        stack.append(v)
        onstack.add(v)

        for w in adj.get(v, []):
            if w not in idx:
                strongconnect(w)
                low[v] = min(low[v], low[w])
            elif w in onstack:
                low[v] = min(low[v], idx[w])

        if low[v] == idx[v]:
            comp = []
            while True:
                w = stack.pop()
                onstack.remove(w)
                comp.append(w)
                if w == v:
                    break
            scc_id = len(sccs)
            for n in comp:
                node_to_scc[n] = scc_id
            sccs.append(comp)

    for v in adj.keys():
        if v not in idx:
            strongconnect(v)

    return node_to_scc, sccs

def structure_decision_points(net,
                              outer_loop_ratio: float = 0.5) -> Dict[str, DecisionPointSpec]:
    """
    Returns place_name -> DecisionPointSpec
    - decision place = place with >=2 outgoing arcs
    - classifies into XOR patterns and loop patterns
    - marks silent-only as SKIP_ONLY_TAU
    """
    decision_places = [p for p in net.places if len(p.out_arcs) > 1]

    adj = _build_transition_graph(net)
    node_to_scc, sccs = _tarjan_scc(adj)
    scc_sizes = {i: len(comp) for i, comp in enumerate(sccs)}
    total_transitions = max(1, len(list(net.transitions)))

    specs: Dict[str, DecisionPointSpec] = {}

    for p in decision_places:
        place_name = str(p.name)
        outs = [a.target for a in p.out_arcs]  # transitions

        out_branches = [_branch_label_for_transition(t) for t in outs]
        visible = [str(t.label) for t in outs if t.label is not None]
        tau = [f"tau::{t.name}" for t in outs if t.label is None]

        # skip: only tau outgoing
        if len(visible) == 0:
            specs[place_name] = DecisionPointSpec(
                place_name=place_name,
                pattern=DecisionPattern.SKIP_ONLY_TAU,
                outgoing_transitions=outs,
                outgoing_branch_labels=out_branches,
                visible_labels=[],
                tau_labels=tau,
            )
            continue

        # ---- loop detection via SCCs on transitions ----
        touched_sccs = {node_to_scc[t] for t in outs if t in node_to_scc}
        self_loop_sccs = set()
        for t in outs:
            sid = node_to_scc.get(t, None)
            if sid is None:
                continue
            if scc_sizes[sid] == 1 and t in adj.get(t, set()):
                self_loop_sccs.add(sid)

        nontrivial_sccs = {sid for sid in touched_sccs if scc_sizes.get(sid, 0) > 1} | self_loop_sccs

        if nontrivial_sccs:
            sid = max(nontrivial_sccs, key=lambda s: scc_sizes.get(s, 0))
            ratio = scc_sizes.get(sid, 0) / total_transitions

            if sid in self_loop_sccs:
                pattern = DecisionPattern.LOOP_SELF
            elif ratio >= outer_loop_ratio:
                pattern = DecisionPattern.LOOP_OUTER
            else:
                pattern = DecisionPattern.LOOP_INNER

            cont, ex = [], []
            for t in outs:
                lbl = _branch_label_for_transition(t)
                sid_t = node_to_scc.get(t, None)
                leads_inside = any(node_to_scc.get(n2, -1) == sid for n2 in adj.get(t, set()))
                stays = (sid_t == sid) or leads_inside
                (cont if stays else ex).append(lbl)

            # skip loops only tau
            if all(l.startswith("tau::") for l in cont + ex):
                specs[place_name] = DecisionPointSpec(
                    place_name=place_name,
                    pattern=DecisionPattern.SKIP_ONLY_TAU,
                    outgoing_transitions=outs,
                    outgoing_branch_labels=out_branches,
                    visible_labels=visible,
                    tau_labels=tau,
                )
            else:
                specs[place_name] = DecisionPointSpec(
                    place_name=place_name,
                    pattern=pattern,
                    outgoing_transitions=outs,
                    outgoing_branch_labels=out_branches,
                    visible_labels=visible,
                    tau_labels=tau,
                    loop_scc_id=sid,
                    loop_continue_branches=cont,
                    loop_exit_branches=ex,
                )
            continue

        # ---- XOR classification ----
        if len(visible) == 1 and len(tau) == 1:
            pattern = DecisionPattern.XOR_1V1T
        elif len(visible) >= 2 and len(tau) == 1:
            pattern = DecisionPattern.XOR_NV1T
        else:
            pattern = DecisionPattern.XOR_GENERAL

        specs[place_name] = DecisionPointSpec(
            place_name=place_name,
            pattern=pattern,
            outgoing_transitions=outs,
            outgoing_branch_labels=out_branches,
            visible_labels=visible,
            tau_labels=tau,
        )

    return specs


# ----------------------------
# Mapping tau branches to next visible activity labels
# (useful for suffix decoding masks)
# ----------------------------
def first_visible_labels_after_transition(net, t_start, max_depth: int = 50) -> Set[str]:
    """
    BFS through tau transitions to find first visible transitions reachable after t_start.
    Returns the set of visible transition labels.
    """
    visible: Set[str] = set()
    visited: Set[Any] = set([t_start])
    frontier = [t_start]
    depth = 0

    while frontier and depth < max_depth:
        nxt = []
        for t in frontier:
            for a in t.out_arcs:
                p = a.target
                for a2 in p.out_arcs:
                    t2 = a2.target
                    if t2 in visited:
                        continue
                    visited.add(t2)
                    if getattr(t2, "label", None) is not None:
                        visible.add(str(t2.label))
                    else:
                        nxt.append(t2)
        frontier = nxt
        depth += 1

    return visible


# ----------------------------
# Guard policies
# ----------------------------
class GuardMode(str, Enum):
    ARGMAX = "argmax"
    THRESHOLD = "threshold"
    TOPK = "topk"


@dataclass
class GuardPolicy:
    mode: GuardMode = GuardMode.THRESHOLD
    threshold: float = 0.15  # used for THRESHOLD
    topk: int = 2            # used for TOPK


def allowed_classes_from_proba(proba: np.ndarray, classes: List[str], policy: GuardPolicy) -> List[str]:
    proba = np.asarray(proba, dtype=float).reshape(-1)
    if policy.mode == GuardMode.ARGMAX:
        return [classes[int(np.argmax(proba))]]

    if policy.mode == GuardMode.TOPK:
        k = max(1, int(policy.topk))
        idx = np.argsort(-proba)[:k]
        return [classes[i] for i in idx]

    # THRESHOLD
    thr = float(policy.threshold)
    allowed = [classes[i] for i, p in enumerate(proba) if float(p) >= thr]
    if not allowed:
        allowed = [classes[int(np.argmax(proba))]]
    return allowed


# ----------------------------
# Simple feature encoder (pandas get_dummies)
# ----------------------------
@dataclass
class FeatureEncoder:
    dummy_columns: List[str]

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        X_enc = pd.get_dummies(X_raw, dummy_na=True)
        X_enc = X_enc.reindex(columns=self.dummy_columns, fill_value=0)
        return X_enc


def fit_feature_encoder(X_raw: pd.DataFrame) -> Tuple[pd.DataFrame, FeatureEncoder]:
    X_enc = pd.get_dummies(X_raw, dummy_na=True)
    enc = FeatureEncoder(dummy_columns=list(X_enc.columns))
    return X_enc, enc


# ----------------------------
# Model selection + imbalance handling
# ----------------------------
class ModelType(str, Enum):
    AUTO = "auto"
    CATBOOST = "catboost"
    LIGHTGBM = "lightgbm"
    XGBOOST = "xgboost"
    SKLEARN_HGB = "sklearn_hgb"
    SKLEARN_TREE = "sklearn_tree"


@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.AUTO
    random_state: int = 7
    max_depth: Optional[int] = None

    # imbalance
    use_inverse_freq_weights: bool = True

    # surrogate tree for readable rules
    train_surrogate_tree: bool = True
    surrogate_max_depth: int = 4
    surrogate_min_leaf: int = 20

    # rule extraction knobs for surrogate
    surrogate_min_leaf_prob: float = 0.2
    surrogate_min_leaf_support: int = 20


def compute_inverse_freq_weights(y_int: np.ndarray) -> np.ndarray:
    counts = np.bincount(y_int)
    counts = np.maximum(counts, 1)
    n = float(y_int.shape[0])
    w_per_class = n / (len(counts) * counts.astype(float))
    return w_per_class[y_int]


def fit_classifier(X: pd.DataFrame, y_int: np.ndarray, sample_weight: Optional[np.ndarray], cfg: ModelConfig):
    """
    Returns a fitted classifier that implements predict_proba and has classes_.
    Tries (in AUTO): LightGBM -> XGBoost -> CatBoost -> sklearn HGB -> sklearn tree
    """
    # ----- explicit model choice -----
    if cfg.model_type != ModelType.AUTO:
        order = [cfg.model_type]
    else:
        order = [
            ModelType.LIGHTGBM,
            ModelType.XGBOOST,
            ModelType.CATBOOST,
            ModelType.SKLEARN_HGB,
            ModelType.SKLEARN_TREE,
        ]
        if CatBoostClassifier is None:
            order = [m for m in order if m != ModelType.CATBOOST]

    last_err = None
    for mt in order:
        try:
            if mt == ModelType.CATBOOST:
                if CatBoostClassifier is None:
                    raise ImportError("catboost is not installed or failed to import")
                model = CatBoostClassifier(
                    depth=cfg.max_depth if cfg.max_depth is not None else 6,
                    random_seed=cfg.random_state,
                    verbose=False,
                    loss_function="MultiClass" if len(np.unique(y_int)) > 2 else "Logloss",
                )
                model.fit(X, y_int, sample_weight=sample_weight)
                return model

            if mt == ModelType.LIGHTGBM:
                model = LGBMClassifier(
                    random_state=cfg.random_state,
                    max_depth=cfg.max_depth if cfg.max_depth is not None else -1,
                    n_estimators=300,
                    learning_rate=0.05,
                )
                model.fit(X, y_int, sample_weight=sample_weight)
                return model

            if mt == ModelType.XGBOOST:
                model = XGBClassifier(
                    random_state=cfg.random_state,
                    max_depth=cfg.max_depth if cfg.max_depth is not None else 6,
                    n_estimators=400,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="multi:softprob" if len(np.unique(y_int)) > 2 else "binary:logistic",
                    eval_metric="mlogloss" if len(np.unique(y_int)) > 2 else "logloss",
                )
                model.fit(X, y_int, sample_weight=sample_weight)
                return model

            if mt == ModelType.SKLEARN_HGB and HistGradientBoostingClassifier is not None:
                model = HistGradientBoostingClassifier(
                    random_state=cfg.random_state,
                    max_depth=cfg.max_depth,
                )
                model.fit(X, y_int, sample_weight=sample_weight)
                return model

            if mt == ModelType.SKLEARN_TREE:
                model = DecisionTreeClassifier(
                    random_state=cfg.random_state,
                    max_depth=cfg.max_depth,
                )
                model.fit(X, y_int, sample_weight=sample_weight)
                return model

        except Exception as e:
            last_err = e
            continue

    raise RuntimeError(f"Could not fit any model backend. Last error: {last_err}")


# ----------------------------
# Surrogate tree rule extraction (probability-based so minority gets rules)
# ----------------------------
def extract_tree_guards_probabilistic(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    class_int_to_label: Dict[int, str],
    min_leaf_prob: float = 0.2,
    min_leaf_support: int = 20,
) -> Tuple[Dict[str, List[str]], Dict[str, str], Dict[str, float]]:
    """
    For each leaf, assign its rule to any class whose leaf prob >= min_leaf_prob
    (AND leaf support >= min_leaf_support). This avoids "(false)" for minority classes.
    """
    tree = clf.tree_
    FEATURE_UNDEF = -2

    def cond_str(feat_idx: int, thresh: float, direction: str) -> str:
        fname = feature_names[feat_idx]
        return f"({fname} <= {thresh:.6g})" if direction == "left" else f"({fname} > {thresh:.6g})"

    # init containers
    labels = [class_int_to_label[int(c)] for c in clf.classes_]
    rules_by_label: Dict[str, List[str]] = {lab: [] for lab in labels}
    best_leaf_prob: Dict[str, float] = {lab: 0.0 for lab in labels}

    stack: List[Tuple[int, List[str]]] = [(0, [])]
    while stack:
        node_id, path_conds = stack.pop()
        feat_idx = int(tree.feature[node_id])

        if feat_idx == FEATURE_UNDEF:
            counts = tree.value[node_id][0].astype(float)
            support = float(counts.sum())
            if support < float(min_leaf_support) or support <= 0:
                continue
            probs = counts / support
            rule = " AND ".join(path_conds) if path_conds else "(true)"

            for j, c_int in enumerate(clf.classes_):
                lab = class_int_to_label[int(c_int)]
                best_leaf_prob[lab] = max(best_leaf_prob[lab], float(probs[j]))
                if float(probs[j]) >= float(min_leaf_prob):
                    rules_by_label[lab].append(rule)
            continue

        thresh = float(tree.threshold[node_id])
        left_id = int(tree.children_left[node_id])
        right_id = int(tree.children_right[node_id])

        stack.append((right_id, path_conds + [cond_str(feat_idx, thresh, "right")]))
        stack.append((left_id,  path_conds + [cond_str(feat_idx, thresh, "left")]))

    guard_by_label: Dict[str, str] = {}
    for lab in labels:
        disj = rules_by_label.get(lab, [])
        if not disj:
            guard_by_label[lab] = "(false)"
        elif len(disj) == 1:
            guard_by_label[lab] = disj[0]
        else:
            guard_by_label[lab] = " OR ".join([f"({r})" for r in disj])

    return rules_by_label, guard_by_label, best_leaf_prob


# ----------------------------
# Output model per decision point
# ----------------------------
@dataclass
class DecisionPointModel:
    spec: DecisionPointSpec

    # feature encoding
    raw_feature_names: List[str]
    encoder: FeatureEncoder

    # classifier
    clf: Any
    label_encoder: LabelEncoder           # maps string branch -> int
    class_int_to_branch: Dict[int, str]   # int -> branch label (e.g., "Resolve ticket" or "tau::t_12")

    # probabilistic guard policy
    guard_policy: GuardPolicy

    # mapping branch -> allowed visible next activities (for suffix decoding)
    branch_to_allowed_activities: Dict[str, List[str]]

    # optional surrogate rules
    surrogate_tree: Optional[DecisionTreeClassifier] = None
    surrogate_rules_by_branch: Optional[Dict[str, List[str]]] = None
    surrogate_guard_by_branch: Optional[Dict[str, str]] = None
    surrogate_best_leaf_prob: Optional[Dict[str, float]] = None

    def _encode_assignment(self, assignment: Dict[str, Any]) -> pd.DataFrame:
        X_raw = pd.DataFrame([assignment], columns=self.raw_feature_names)
        return self.encoder.transform(X_raw)

    def predict_proba(self, assignment: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
        X = self._encode_assignment(assignment)
        proba = self.clf.predict_proba(X)[0]
        # clf classes_ are ints after label encoding
        class_ints = list(getattr(self.clf, "classes_", np.arange(len(proba))))
        branches = [self.class_int_to_branch[int(c)] for c in class_ints]
        return branches, np.asarray(proba, dtype=float)

    def allowed_branches(self, assignment: Dict[str, Any]) -> List[str]:
        branches, proba = self.predict_proba(assignment)
        return allowed_classes_from_proba(proba, branches, self.guard_policy)

    def allowed_next_activities(self, assignment: Dict[str, Any]) -> List[str]:
        branches = self.allowed_branches(assignment)
        allowed: Set[str] = set()
        for b in branches:
            for a in self.branch_to_allowed_activities.get(b, []):
                allowed.add(a)
        return sorted(allowed)


@dataclass
class DecisionMiningResult:
    specs: Dict[str, DecisionPointSpec]                 # place -> spec
    models: Dict[str, DecisionPointModel]               # place -> trained model (excluding SKIP_ONLY_TAU)
    skipped: Dict[str, DecisionPointSpec]               # place -> spec (SKIP_ONLY_TAU etc.)


# ----------------------------
# Main class
# ----------------------------
class DecisionDiscovery:
    def __init__(self, 
                 petri_net: Tuple,                
                 event_log_df: pd.DataFrame,
                 case_ids: Optional[List[str]],
                 case_id_key: str = "case:concept:name",
                 activity_key: str = "concept:name",
                 time_key: str = "time:timestamp"):
        self.net, self.im, self.fm = petri_net
        self.event_log_df = event_log_df
        self.case_ids = case_ids

        self.case_id_key = case_id_key
        self.activity_key = activity_key
        self.time_key = time_key

    def _filter_event_log_df(self) -> pd.DataFrame:
        df = self.event_log_df.copy()

        if self.case_ids is not None:
            df = df[df[self.case_id_key].isin(set(self.case_ids))]

        rename_map = {}
        if self.case_id_key in df.columns and self.case_id_key != "case:concept:name":
            rename_map[self.case_id_key] = "case:concept:name"
        if self.activity_key in df.columns and self.activity_key != "concept:name":
            rename_map[self.activity_key] = "concept:name"
        if self.time_key in df.columns and self.time_key != "time:timestamp":
            rename_map[self.time_key] = "time:timestamp"

        if rename_map:
            df = df.rename(columns=rename_map)

        if "time:timestamp" in df.columns:
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        sort_cols = ["case:concept:name"]
        if "time:timestamp" in df.columns:
            sort_cols.append("time:timestamp")
        df = df.sort_values(sort_cols).reset_index(drop=True)

        return df

    def _augment_with_trace_attributes(self,
                                       df: pd.DataFrame,
                                       trace_attributes: Optional[List[str]] = None,
                                       agg: str = "first") -> pd.DataFrame:
        if not trace_attributes:
            return df

        missing = [c for c in trace_attributes if c not in df.columns]
        if missing:
            raise ValueError(f"Trace attributes not found in dataframe columns: {missing}")

        def reducer(s: pd.Series):
            s2 = s.dropna()
            if s2.empty:
                return np.nan
            if agg == "first":
                return s2.iloc[0]
            if agg == "last":
                return s2.iloc[-1]
            if agg == "mode":
                m = s2.mode()
                return m.iloc[0] if not m.empty else s2.iloc[0]
            raise ValueError(f"Unknown agg='{agg}'")

        case_col = "case:concept:name"
        trace_df = df.groupby(case_col, as_index=False)[trace_attributes].agg(reducer)
        out = df.drop(columns=trace_attributes).merge(trace_df, on=case_col, how="left")
        return out

    @staticmethod
    def _uniq_preserve_order(xs: Optional[List[str]]) -> List[str]:
        if not xs:
            return []
        seen = set()
        out = []
        for x in xs:
            if x is None:
                continue
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out

    def mine_decision_models(self,
                             attributes: Optional[List[str]] = None,
                             trace_attributes: Optional[List[str]] = None,
                             trace_attr_agg: str = "first",
                             
                             k_last_activities: int = 1,
                             
                             outer_loop_ratio: float = 0.5,
                             guard_policy: GuardPolicy = GuardPolicy(),
                             model_cfg: ModelConfig = ModelConfig(),
                             map_tau_to_next_visible: bool = True,
                             max_tau_bfs_depth: int = 50) -> DecisionMiningResult:
        """
        Main entry point:
        - structures decision points into your patterns
        - builds decision table via alignments (PM4Py)
        - trains advanced classifier per decision point (skipping silent-only)
        - returns models that can produce decoding masks
        """
        df = self._filter_event_log_df()
        df = self._augment_with_trace_attributes(df, trace_attributes=trace_attributes, agg=trace_attr_agg)

        feat_cols = []
        if attributes:
            feat_cols.extend(attributes)
        if trace_attributes:
            feat_cols.extend(trace_attributes)
        feat_cols = self._uniq_preserve_order(feat_cols)

        # 1) structure decision points
        specs_all = structure_decision_points(self.net, outer_loop_ratio=outer_loop_ratio)
        skipped = {p: s for p, s in specs_all.items() if s.pattern == DecisionPattern.SKIP_ONLY_TAU}
        trainable_specs = {p: s for p, s in specs_all.items() if s.pattern != DecisionPattern.SKIP_ONLY_TAU}
        print(trainable_specs)

        # We only request decision points that we actually want to train
        pre_points = list(trainable_specs.keys())
        print(pre_points)
        
        return

        # 3) train one model per decision point
        models: Dict[str, DecisionPointModel] = {}

        for place_name, spec in trainable_specs.items():
            #samples = .get(place_name, [])
            samples = None
            if not samples:
                # no training data: skip but keep spec in skipped for debugging
                skipped[place_name] = spec
                continue

            # convert decision table rows to X_raw and y_branch
            assignments: List[Dict[str, Any]] = []
            y_branch: List[str] = []

            for (ass, chosen) in samples:
                # ass: Dict[str, Any] "current known assignment"
                # chosen: Transition or label; convert to stable branch label
                if hasattr(chosen, "label") or hasattr(chosen, "name"):
                    chosen_branch = _branch_label_for_transition(chosen)
                else:
                    chosen_branch = str(chosen)

                assignments.append(dict(ass))
                y_branch.append(chosen_branch)

            X_raw = pd.DataFrame(assignments)
            # ensure stable feature set
            if feat_cols:
                for c in feat_cols:
                    if c not in X_raw.columns:
                        X_raw[c] = np.nan
                X_raw = X_raw[feat_cols]
            else:
                # if user didn't specify, use whatever PM4Py produced
                feat_cols = list(X_raw.columns)

            # encode X (one-hot for categorical)
            X_enc, encoder = fit_feature_encoder(X_raw)

            # encode y
            le = LabelEncoder()
            y_int = le.fit_transform(np.array(y_branch, dtype=str))
            class_int_to_branch = {int(i): str(lbl) for i, lbl in enumerate(le.classes_)}

            # weights for imbalance
            sample_weight = None
            if model_cfg.use_inverse_freq_weights:
                sample_weight = compute_inverse_freq_weights(y_int)

            # fit main classifier
            clf = fit_classifier(X_enc, y_int, sample_weight, model_cfg)

            # build branch -> allowed next activities (for suffix decoding masks)
            branch_to_allowed_activities: Dict[str, List[str]] = {}
            # map outgoing transitions in spec by branch label
            out_map = { _branch_label_for_transition(t): t for t in spec.outgoing_transitions }

            for b in spec.outgoing_branch_labels:
                t = out_map.get(b, None)
                if t is None:
                    # unknown branch; do nothing
                    branch_to_allowed_activities[b] = []
                    continue

                if getattr(t, "label", None) is not None:
                    # visible: the next visible activity is usually itself
                    branch_to_allowed_activities[b] = [str(t.label)]
                else:
                    # tau branch
                    if map_tau_to_next_visible:
                        vis = first_visible_labels_after_transition(self.net, t, max_depth=max_tau_bfs_depth)
                        branch_to_allowed_activities[b] = sorted(vis)
                    else:
                        branch_to_allowed_activities[b] = []  # or keep tau as non-activity

            dpm = DecisionPointModel(
                spec=spec,
                raw_feature_names=list(feat_cols),
                encoder=encoder,
                clf=clf,
                label_encoder=le,
                class_int_to_branch=class_int_to_branch,
                guard_policy=guard_policy,
                branch_to_allowed_activities=branch_to_allowed_activities,
            )

            # optional: surrogate tree for readable rules/guards
            if model_cfg.train_surrogate_tree:
                # distill: train small tree to predict main model's argmax
                # (you can also distill on probabilities if you prefer)
                try:
                    y_hat_int = np.argmax(clf.predict_proba(X_enc), axis=1)
                except Exception:
                    y_hat_int = clf.predict(X_enc)

                surrogate = DecisionTreeClassifier(
                    random_state=model_cfg.random_state,
                    max_depth=model_cfg.surrogate_max_depth,
                    min_samples_leaf=model_cfg.surrogate_min_leaf,
                )
                surrogate.fit(X_enc, y_hat_int, sample_weight=sample_weight)

                # rules: probability-based extraction to avoid minority collapse
                class_int_to_branch_sur = {int(c): class_int_to_branch[int(c)] for c in surrogate.classes_}
                rules_by, guard_by, best_leaf = extract_tree_guards_probabilistic(
                    surrogate,
                    feature_names=list(X_enc.columns),
                    class_int_to_label=class_int_to_branch_sur,
                    min_leaf_prob=model_cfg.surrogate_min_leaf_prob,
                    min_leaf_support=model_cfg.surrogate_min_leaf_support,
                )

                dpm.surrogate_tree = surrogate
                dpm.surrogate_rules_by_branch = rules_by
                dpm.surrogate_guard_by_branch = guard_by
                dpm.surrogate_best_leaf_prob = best_leaf

            models[place_name] = dpm

        return DecisionMiningResult(specs=specs_all, models=models, skipped=skipped)
