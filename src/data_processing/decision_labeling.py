"""
Decision-aware event labeling for suffix prediction. Implements the labeling described in the paper:

Two modes:
- Offline (training): uses optimal alignments from decision mining.
- Online (inference): uses token-based replay on prefixes.
"""
from __future__ import annotations

from collections import defaultdict
import importlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.obj import Marking

_MODULE_RENAMES = {
    "decision_mining.custom_framework.function_estimator_DT_basic": "decision_mining.function_estimator_DT_basic",
    "decision_mining.custom_framework.function_estimator_catboost_advanced": "decision_mining.function_estimator_catboost_advanced",
    "decision_mining.custom_framework.decision_discovery": "decision_mining.decision_discovery",
}

class _CompatUnpickler(pickle.Unpickler):
    """Redirect old module paths so that legacy .pkl files load correctly."""

    def find_class(self, module: str, name: str):
        module = _MODULE_RENAMES.get(module, module)
        return super().find_class(module, name)


def _compat_unpickle(f):
    return _CompatUnpickler(f).load()


def _load_estimator_artifact(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    if obj.get("artifact_type") != "decision_mining_estimator":
        return obj

    module_name = str(obj.get("estimator_module", ""))
    class_name = str(obj.get("estimator_class", "FunctionEstimator"))
    if not module_name:
        return obj

    module_name = _MODULE_RENAMES.get(module_name, module_name)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    if hasattr(cls, "from_artifact"):
        return cls.from_artifact(obj)
    return obj


# Sentinel for non-decision events
BOTTOM = "⊥"


def _resolve_nu(
    transition: Any,
    net: Any,
    *,
    _visited: Optional[frozenset] = None,
) -> set:
    """Resolve ν(t): *all* first-reachable non-silent transitions after *transition*.

    If *transition* itself is non-silent (has a label), return ``{transition}``.
    Otherwise follow silent transitions through the net, branching at every
    intermediate place (including decision places).

    A frozenset copy of *_visited* is used per branch so that parallel paths
    through a shared intermediate node are all explored, while cycles are
    still detected and pruned.

    Returns a set of visible (non-silent) transitions.
    """
    if _visited is None:
        _visited = frozenset()

    if transition in _visited:
        return set()
    _visited = _visited | {transition}

    if getattr(transition, "label", None) is not None:
        return {transition}

    # transition is silent – follow its output places (branch at each)
    results: set = set()
    for out_arc in transition.out_arcs:
        place = out_arc.target
        for place_out_arc in place.out_arcs:
            next_trans = place_out_arc.target
            results |= _resolve_nu(next_trans, net, _visited=_visited)
    return results


def _build_nu_mapping(
    decision_place: Any,
    net: Any,
) -> Dict[Any, set]:
    """Build {outgoing_transition: set-of-ν-transitions} for a decision place.

    Each outgoing transition is mapped to the set of all first-reachable
    visible transitions.  For a non-silent outgoing transition the set
    contains just itself.  For a silent one the set may contain many
    visible transitions (reachable through intermediate silent/decision
    structures).
    """
    mapping: Dict[Any, set] = {}
    for arc in decision_place.out_arcs:
        t = arc.target
        mapping[t] = _resolve_nu(t, net)
    return mapping


def _build_soft_distribution(
    transition_probs: Dict[str, float],
    nu_mapping: Dict[Any, set],
    transition_by_name: Dict[str, Any],
) -> Dict[str, float]:
    """Convert decision-model transition probabilities to event-label probabilities.

    Uses the ν-mapping to distribute each outgoing transition's predicted
    probability mass uniformly across its reachable visible event labels.

    Returns a dict {event_label: probability} (normalised to sum to 1).
    """
    label_mass: Dict[str, float] = defaultdict(float)
    total_defined = 0.0

    for trans_obj, nu_set in nu_mapping.items():
        trans_name = str(trans_obj.name)
        prob = transition_probs.get(trans_name, 0.0)
        if not nu_set or prob <= 0.0:
            continue

        # Collect unique visible labels reachable from this transition
        visible_labels: set = set()
        for nu_trans in nu_set:
            lbl = getattr(nu_trans, "label", None)
            if lbl is not None:
                visible_labels.add(str(lbl))

        if not visible_labels:
            continue

        # Distribute this transition's probability uniformly across its
        # reachable visible labels
        share = prob / len(visible_labels)
        for lbl in visible_labels:
            label_mass[lbl] += share
        total_defined += prob

    if total_defined <= 0.0:
        return {}

    # normalise
    return {lbl: mass / total_defined for lbl, mass in label_mass.items()}


class DecisionLabeler:
    """Produces decision-aware event labels for prefix datasets.

    Parameters
    ----------
    petri_net : tuple
        ``(net, initial_marking, final_marking)`` from pm4py.
    decision_model_dir : str
        Path to directory containing per-place ``.pkl`` estimator files.
    decision_places_bundle_path : str
        Path to the ``decision_places_bundle.json`` produced by
        ``DecisionDiscovery.save_results``.
    dynamic_attributes, static_attributes : list[str] | None
        The same attribute lists given to ``DecisionDiscovery``
        during model training.
    """

    def __init__(
        self,
        petri_net: Tuple,
        decision_model_dir: str,
        decision_places_bundle_path: str,
        dynamic_attributes: Optional[List[str]] = None,
        static_attributes: Optional[List[str]] = None,
    ) -> None:
        import json
        from pathlib import Path

        self.net, self.im, self.fm = petri_net

        self.dynamic_attributes = list(dynamic_attributes or [])
        self.static_attributes = list(static_attributes or [])
        self.past_attr_keys = self.dynamic_attributes + self.static_attributes

        # transition lookup
        self.transition_by_name: Dict[str, Any] = {
            str(t.name): t for t in self.net.transitions
        }
        self.transition_by_label: Dict[str, Any] = {}
        for t in self.net.transitions:
            if t.label is not None:
                self.transition_by_label.setdefault(str(t.label), []).append(t)

        # decision places
        self.decision_places: List[Any] = [
            p for p in self.net.places if len(p.out_arcs) > 1
        ]
        self.decision_place_names: set = {str(p) for p in self.decision_places}
        self.decision_place_by_name: Dict[str, Any] = {
            str(p): p for p in self.decision_places
        }

        # nu-mappings per decision place
        self.nu_mappings: Dict[str, Dict[Any, set]] = {}
        for p in self.decision_places:
            self.nu_mappings[str(p)] = _build_nu_mapping(p, self.net)

        # load estimator models
        self.estimators: Dict[str, Any] = {}
        model_dir = Path(decision_model_dir)
        with open(decision_places_bundle_path, "r", encoding="utf-8") as f:
            bundle = json.load(f)

        for entry in bundle:
            place_name = entry["place_name"]
            model_path_str = entry.get("model_path", "")
            if not model_path_str:
                continue
            # Prioritise decision_model_dir (lookup by filename) because the
            # paths stored in the bundle are relative to the CWD that was
            # active when the mining notebook ran – not to the bundle file.
            model_filename = Path(model_path_str).name
            candidate = model_dir / model_filename
            candidate_sub = model_dir / "models" / model_filename
            if candidate.exists():
                model_path = candidate
            elif candidate_sub.exists():
                model_path = candidate_sub
            else:
                model_path = Path(model_path_str)
                if not model_path.is_absolute():
                    model_path = (
                        Path(decision_places_bundle_path).parent / model_path_str
                    )
            if model_path.exists():
                with open(model_path, "rb") as f:
                    loaded = _compat_unpickle(f)
                    self.estimators[place_name] = _load_estimator_artifact(loaded)

        # Build UUID remapping: model-transition-name → net-transition-name.
        # The Inductive Miner assigns random UUIDs to visible transitions on
        # every run, so models trained on one net use different UUIDs than
        # the net in *petri_net*.  Silent transitions (skip_X, init_loop_X)
        # get deterministic names and are fine.  For visible transitions we
        # match by elimination: if N model names are unmatched and N net
        # names are unmatched (all visible), pair them 1-to-1.
        self._transition_remap: Dict[str, Dict[str, str]] = {}  # place -> {model_name: net_name}
        for place_name, estimator in self.estimators.items():
            place = self.decision_place_by_name.get(place_name)
            if place is None:
                continue
            net_names = {str(a.target.name) for a in place.out_arcs}
            model_names = set(estimator.label_encoder.classes_)
            unmatched_model = sorted(model_names - net_names)
            unmatched_net = sorted(net_names - model_names)
            if len(unmatched_model) == len(unmatched_net) == 1:
                self._transition_remap[place_name] = {
                    unmatched_model[0]: unmatched_net[0]
                }
            elif len(unmatched_model) > 1 and len(unmatched_model) == len(unmatched_net):
                # Multiple visible transitions – log a warning but cannot
                # disambiguate without the original training net labels.
                import warnings
                warnings.warn(
                    f"Decision place {place_name}: {len(unmatched_model)} "
                    f"unmatched model→net UUIDs; cannot auto-remap."
                )

        # Rich labeling records for inspection/debugging.
        # Offline: case_id -> list of records (one per synchronized event)
        self.last_offline_label_records: Dict[str, List[Dict[str, Any]]] = {}
        # Online: list of records (one per prefix event)
        self.last_online_label_records: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Feature-row building (mirrors DecisionDiscovery._build_feature_row
    # and _filter_attributes, but standalone)
    # ------------------------------------------------------------------

    def _filter_attributes(
        self, attrs: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not attrs:
            return {}
        if self.past_attr_keys:
            filtered = {k: attrs.get(k, np.nan) for k in self.past_attr_keys}
        else:
            filtered = attrs
        out: Dict[str, Any] = {}
        for k, v in filtered.items():
            if isinstance(v, (int, np.integer)):
                out[k] = str(int(v))
            else:
                out[k] = v
        return out

    @staticmethod
    def _build_feature_row(past_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the feature dict consumed by the decision estimator.

        Mirrors ``DecisionDiscovery._build_feature_row`` exactly so that
        predictions are consistent with how the models were trained.
        """
        features: Dict[str, Any] = {}
        features["past_events_count"] = float(len(past_events))

        if not past_events:
            return features

        previous_event = past_events[-1]
        older_events = past_events[:-1]
        features["older_events_count"] = float(len(older_events))

        for key, value in previous_event.items():
            if isinstance(value, (int, np.integer)):
                value = str(int(value))
            features[f"{key}_prev_event"] = value

        if older_events:
            summary_keys: set = set()
            for ev in older_events:
                summary_keys.update(ev.keys())
            for key in sorted(summary_keys):
                seq = [ev.get(key, np.nan) for ev in older_events]
                valid_vals = [
                    v for v in seq
                    if not (
                        v is None
                        or (isinstance(v, (float, np.floating)) and np.isnan(v))
                    )
                ]
                features[f"{key}_older_non_null_count"] = float(len(valid_vals))
                if not valid_vals:
                    continue
                if all(isinstance(v, (float, np.floating)) for v in valid_vals):
                    arr = np.array(valid_vals, dtype=float)
                    features[f"{key}_older_mean"] = float(np.nanmean(arr))
                    features[f"{key}_older_std"] = float(np.nanstd(arr))
                    features[f"{key}_older_min"] = float(np.nanmin(arr))
                    features[f"{key}_older_max"] = float(np.nanmax(arr))
                else:
                    cat_vals = [str(v) for v in valid_vals]
                    features[f"{key}_older_last"] = cat_vals[-1]
                    features[f"{key}_older_nunique"] = float(len(set(cat_vals)))

        keys: set = set()
        for ev in past_events:
            keys.update(ev.keys())

        for key in sorted(keys):
            seq = [ev.get(key, np.nan) for ev in past_events]

            def _is_float_value(v):
                return isinstance(v, (float, np.floating)) and not np.isnan(v)

            is_continuous = any(_is_float_value(v) for v in seq)

            def _as_cat(v):
                if v is None:
                    return np.nan
                if isinstance(v, float) and np.isnan(v):
                    return np.nan
                if isinstance(v, bool):
                    return v
                return str(v)

            def _as_num(v):
                if isinstance(v, (float, np.floating)):
                    return v
                return np.nan

            getv = _as_num if is_continuous else _as_cat

            if len(seq) >= 1:
                features[f"{key}_prev1"] = getv(seq[-1])
            if len(seq) >= 2:
                features[f"{key}_prev2"] = getv(seq[-2])
            if len(seq) >= 3:
                features[f"{key}_prev3"] = getv(seq[-3])

        return features

    def _predict_at_place(
        self,
        place_name: str,
        past_events: List[Dict[str, Any]],
        _visited_places: Optional[frozenset] = None,
    ) -> Dict[str, float]:
        """Query the decision model at *place_name* and return a soft
        next-event-label distribution z_i(a).

        Uses **recursive model composition**: when a silent outgoing
        transition leads to another decision place that also has a trained
        model, that model is queried and the resulting sub-distribution is
        composed with the current transition probability.  This avoids the
        uniform-dilution problem that occurs when naïvely distributing mass
        across *all* reachable visible labels.
        """
        if _visited_places is None:
            _visited_places = frozenset()
        if place_name in _visited_places:
            return {}
        _visited_places = _visited_places | {place_name}

        estimator = self.estimators.get(place_name)
        if estimator is None:
            return {}

        feature_row = self._build_feature_row(past_events)
        transition_labels, probs = estimator.predict_proba(feature_row)

        # Remap model UUIDs to net UUIDs for visible transitions
        remap = self._transition_remap.get(place_name, {})
        transition_probs = {
            remap.get(lbl, lbl): p
            for lbl, p in zip(transition_labels, probs.tolist())
        }

        place = self.decision_place_by_name.get(place_name)
        if place is None:
            return {}

        label_mass: Dict[str, float] = defaultdict(float)
        total = 0.0

        for arc in place.out_arcs:
            trans = arc.target
            trans_name = str(trans.name)
            prob = transition_probs.get(trans_name, 0.0)
            if prob <= 0.0:
                continue

            sub_dist = self._follow_transition(
                trans, past_events, _visited_places
            )
            for lbl, sub_p in sub_dist.items():
                label_mass[lbl] += prob * sub_p
            total += prob

        if total <= 0.0:
            return {}
        return {lbl: mass / total for lbl, mass in label_mass.items()}

    def _follow_transition(
        self,
        trans: Any,
        past_events: List[Dict[str, Any]],
        visited_places: frozenset,
        _visited_trans: Optional[frozenset] = None,
    ) -> Dict[str, float]:
        """Follow *trans* and return a distribution over visible event labels.

        * Visible transition → ``{label: 1.0}``
        * Silent transition → follow output places:
          - Decision place with a model → recursively predict via the model
          - Non-decision place → follow its single outgoing transition
          - Parallel split (multiple output places) → average branches
        """
        if _visited_trans is None:
            _visited_trans = frozenset()
        if trans in _visited_trans:
            return {}
        _visited_trans = _visited_trans | {trans}

        # Visible transition – trivially maps to its label
        if trans.label is not None:
            return {str(trans.label): 1.0}

        # Silent transition – collect distributions from output places
        combined: Dict[str, float] = defaultdict(float)
        n_branches = 0

        for out_arc in trans.out_arcs:
            place = out_arc.target
            pname = str(place)

            if pname in self.decision_place_names:
                # Another decision place → use its model recursively
                sub = self._predict_at_place(
                    pname, past_events, visited_places
                )
                if sub:
                    for lbl, p in sub.items():
                        combined[lbl] += p
                    n_branches += 1
            else:
                # Non-decision place → follow its outgoing transitions
                for pa in place.out_arcs:
                    sub = self._follow_transition(
                        pa.target, past_events, visited_places, _visited_trans
                    )
                    if sub:
                        for lbl, p in sub.items():
                            combined[lbl] += p
                        n_branches += 1

        if n_branches <= 0:
            return {}
        # Average across parallel branches (tau-splits)
        return {lbl: mass / n_branches for lbl, mass in combined.items()}

    # ------------------------------------------------------------------
    # Shallow (proximate) prediction: stops at downstream decision places
    # ------------------------------------------------------------------

    def _predict_shallow(
        self,
        place_name: str,
        past_events: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """Predict next visible activity at *place_name*.

        The decision models are trained to predict next visible activity
        directly (not outgoing transitions), so the model output is
        already in the activity label space.  No ν-mapping or transition-
        following is needed.

                Returns ``activity_dist`` as ``{activity_label: probability}``.
        """
        estimator = self.estimators.get(place_name)
        if estimator is None:
                        return {}

        feature_row = self._build_feature_row(past_events)
        labels, probs = estimator.predict_proba(feature_row)

        dist = {
            lbl: float(p)
            for lbl, p in zip(labels, probs.flatten())
            if p > 0
        }
        return dist

    # ------------------------------------------------------------------
    # Offline labeling (training): uses optimal alignments
    # ------------------------------------------------------------------

    def _places_after_transition(self, trans: Any) -> List[Any]:
        """Return the decision places in t• (output places of *trans*)."""
        return [
            arc.target
            for arc in trans.out_arcs
            if arc.target in self.decision_places
        ]

    def label_traces_offline(
        self,
        event_log_df: pd.DataFrame,
        sorted_case_ids: List[str],
        alignments: List[Any],
    ) -> Dict[str, List[List[Tuple[str, List[Dict[str, Any]], Dict[str, float], float]]]]:
        """Label every event from the **first** decision place in t•.

        For each visible event e_i, we inspect the output places of its
        aligned transition t.  If any place p in t• is a decision point
        (|p•| > 1), we query the shallow decision model at p using the
        data state up to and including e_i.

        This labeling depends only on e_i and the process structure — not
        on e_{i+1} — so it is identical during training and inference
        and does not leak future information.

        The shallow prediction returns a resolved distribution z_i over
        directly reachable visible event labels.

        Returns
        -------
        dict  {case_id: list-of-event-labels}
            Each event label is
            ``(p_i, A_i, z_i, c_i)`` for a decision event, or
            ``(BOTTOM, A_i, {}, 0.0)`` for a non-decision event.
            The list has one entry per *synchronized* (visible) event.
        """
        # Pre-compute elapsed times the same way DecisionDiscovery does.
        df = event_log_df.copy()
        case_col = "case:concept:name"
        ts_col = "time:timestamp"
        if ts_col in df.columns:
            case_start = df.groupby(case_col)[ts_col].transform("min")
            df["case_elapsed_time"] = (df[ts_col] - case_start).dt.total_seconds()
            elapsed = df.groupby(case_col)[ts_col].diff().dt.total_seconds()
            df["event_elapsed_time"] = elapsed.fillna(0.0)

        result: Dict[str, List[List[Tuple[str, List[Dict[str, Any]], Dict[str, float], float]]]] = {}
        self.last_offline_label_records = {}

        for case_id, case_alignment in zip(sorted_case_ids, alignments):
            case = df[df[case_col] == case_id].reset_index(drop=True)
            case_event_cursor = 0
            past_events: List[Dict[str, Any]] = []
            event_labels: List[Tuple[str, List[Dict[str, Any]], Dict[str, float], float]] = []
            event_records: List[Dict[str, Any]] = []

            for (log_name, model_name), (log_label, model_label) in case_alignment:
                if model_name == ">>":
                    continue  # log-only move

                trans = self.transition_by_name.get(model_name)
                if trans is None:
                    continue

                is_sync = (log_name != ">>") and (model_name != ">>")

                if is_sync:
                    # Match event in the log
                    candidate_labels = [
                        lbl for lbl in [log_label, model_label] if lbl
                    ]
                    matched_event = None
                    for pos in range(case_event_cursor, len(case)):
                        if case.iloc[pos].get("concept:name", None) in candidate_labels:
                            matched_event = case.iloc[pos]
                            case_event_cursor = pos + 1
                            break

                    if matched_event is not None:
                        ev_dict = matched_event.to_dict()
                        event_attrs = self._filter_attributes(ev_dict)
                    else:
                        event_attrs = None

                    # Update data state first: decision-aware labeling for a
                    # place in t• predicts the next event after e_i, so the
                    # feature state A_i includes e_i and all prior events.
                    if event_attrs is not None:
                        past_events.append(event_attrs)

                    # Inspect t• for decision places.
                    dps = self._places_after_transition(trans)
                    if dps:
                        # In structured nets from Inductive Miner, at most
                        # one place in t• is a decision place.
                        dp = dps[0]
                        dp_name = str(dp)
                        z_i = self._predict_shallow(dp_name, past_events)
                        c_i = max(z_i.values(), default=0.0)
                        A_i = [dict(ev) for ev in past_events]
                        event_labels.append((dp_name, A_i, z_i, c_i))
                        event_records.append(
                            {
                                "p_i": dp_name,
                                "A_i": A_i,
                                "z_i": dict(z_i),
                                "c_i": float(c_i),
                            }
                        )
                    else:
                        A_i = [dict(ev) for ev in past_events]
                        event_labels.append((BOTTOM, A_i, {}, 0.0))
                        event_records.append(
                            {
                                "p_i": BOTTOM,
                                "A_i": A_i,
                                "z_i": {},
                                "c_i": 0.0,
                            }
                        )

            result[case_id] = event_labels
            self.last_offline_label_records[case_id] = event_records

        return result

    # ------------------------------------------------------------------
    # Online labeling (inference): uses token-based replay on prefixes
    # ------------------------------------------------------------------

    def label_prefix_online(
        self,
        prefix_activities: List[str],
        prefix_event_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, List[Dict[str, Any]], Dict[str, float], float]]:
        """Label a single prefix using token-based replay.

        For each event e_i we replay the prefix up to (and including) e_i
        and inspect the reached marking for decision places.  If at least
        one decision place is present, we query the shallow decision model
        at that place.

        This mirrors the offline labeling: we inspect t• (the output
        places of the transition that fired e_i) which appear as marked
        places in the marking after replaying up to e_i.

        Returns
        -------
        list of (p_i | BOTTOM, A_i, z_i | {}, c_i | 0.0)
        """
        labels: List[Tuple[str, List[Dict[str, Any]], Dict[str, float], float]] = []
        past_events: List[Dict[str, Any]] = []
        self.last_online_label_records = []

        # Pre-compute per-event markings via incremental replay
        markings: List[Marking] = []
        for event_idx in range(len(prefix_activities)):
            trace_so_far = Trace(
                [Event({"concept:name": act})
                 for act in prefix_activities[: event_idx + 1]]
            )
            replayed = token_replay.apply(
                log=[trace_so_far],
                net=self.net,
                initial_marking=self.im,
                final_marking=Marking(),
            )
            markings.append(replayed[0]["reached_marking"])

        def _decision_places_in_marking(marking: Marking) -> List[str]:
            return [
                str(p) for p in marking
                if str(p) in self.decision_place_names
            ]

        for event_idx, activity in enumerate(prefix_activities):
            # Fetch raw event attributes for this position.
            if prefix_event_data is not None and event_idx < len(prefix_event_data):
                event_attrs = self._filter_attributes(prefix_event_data[event_idx])
            else:
                event_attrs = None

            # Decision-aware labeling at t• uses A_i that includes e_i.
            if event_attrs is not None:
                past_events.append(event_attrs)

            # Use the marking after e_i — this contains t• for e_i's
            # transition, i.e. the decision places reachable right after
            # e_i fires.  No knowledge of e_{i+1} required.
            dp_list = _decision_places_in_marking(markings[event_idx])

            if dp_list:
                # In structured nets, at most one place in t• is a dp.
                dp_name = dp_list[0]
                z_i = self._predict_shallow(dp_name, past_events)
                c_i = max(z_i.values(), default=0.0)
                A_i = [dict(ev) for ev in past_events]
                labels.append((dp_name, A_i, z_i, c_i))
                self.last_online_label_records.append(
                    {
                        "p_i": dp_name,
                        "A_i": A_i,
                        "z_i": dict(z_i),
                        "c_i": float(c_i),
                    }
                )
            else:
                A_i = [dict(ev) for ev in past_events]
                labels.append((BOTTOM, A_i, {}, 0.0))
                self.last_online_label_records.append(
                    {
                        "p_i": BOTTOM,
                        "A_i": A_i,
                        "z_i": {},
                        "c_i": 0.0,
                    }
                )

        return labels

    # ------------------------------------------------------------------
    # Batch labeling for EventLogDataset
    # ------------------------------------------------------------------

    def label_dataset_offline(
        self,
        dataset: Any,
        event_log_df: pd.DataFrame,
        sorted_case_ids: List[str],
        alignments: List[Any],
    ) -> None:
        """Label all prefixes in an ``EventLogDataset`` using offline alignments.

        Calls ``dataset.set_decision_data()`` to populate decision labels
        in-place. Each position in a prefix gets a tuple
        ``(place_name, {next_event_label: probability}, c_i)``.

        For non-decision events the entry is ``(BOTTOM, {}, 0.0)``.
        For events beyond the trace length (EOS padding), the entry is
        ``(BOTTOM, {}, 0.0)``.
        """
        # 1) Label full traces
        trace_labels = self.label_traces_offline(
            event_log_df, sorted_case_ids, alignments
        )

        # 2) Map to prefix rows in the dataset
        n_samples = len(dataset)
        decision_rows: List[List[Tuple[str, Dict[str, float], float]]] = []

        for idx in range(n_samples):
            case_id = dataset.case_ids[idx]
            prefix_len = dataset._prefix_length_from_zero_mask(
                dataset.zero_padding[idx]
            )
            prefix_activities = dataset._extract_prefix_activity_labels(idx)

            # Get the full trace labels for this case
            full_trace_labels = trace_labels.get(case_id, [])

            # The prefix of length L corresponds to the first L visible events.
            # We align by matching activity labels.
            prefix_decision_labels: List[Tuple[str, Dict[str, float], float]] = []
            trace_cursor = 0
            for pos in range(prefix_len):
                act = prefix_activities[pos] if pos < len(prefix_activities) else ""
                if act == "EOS" or act == "":
                    prefix_decision_labels.append((BOTTOM, {}, 0.0))
                    continue

                # Find matching event in the trace labels
                matched = False
                while trace_cursor < len(full_trace_labels):
                    trace_entry = full_trace_labels[trace_cursor]
                    # The trace_labels list is in order of visible events,
                    # so just advance the cursor
                    trace_cursor += 1
                    if len(trace_entry) == 4:
                        place_name, _, dist, c_i = trace_entry
                        prefix_decision_labels.append((place_name, dist, c_i))
                    else:
                        prefix_decision_labels.append((BOTTOM, {}, 0.0))
                    matched = True
                    break

                if not matched:
                    prefix_decision_labels.append((BOTTOM, {}, 0.0))

            decision_rows.append(prefix_decision_labels)

        dataset.set_decision_data(decision_rows)

    def label_dataset_online(
        self,
        dataset: Any,
        event_log_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """Label all prefixes in an ``EventLogDataset`` using token-based replay.

        This is the inference-time labeling path. Each prefix is replayed
        independently through the Petri net; no alignments are needed.

        If *event_log_df* is provided, attribute data for each event is
        looked up by case ID and activity sequence position. Otherwise only
        structural features are used.
        """
        case_col = "case:concept:name"

        # Pre-index event data by case if available
        case_events: Dict[str, pd.DataFrame] = {}
        if event_log_df is not None:
            df = event_log_df.copy()
            ts_col = "time:timestamp"
            if ts_col in df.columns:
                case_start = df.groupby(case_col)[ts_col].transform("min")
                df["case_elapsed_time"] = (df[ts_col] - case_start).dt.total_seconds()
                elapsed = df.groupby(case_col)[ts_col].diff().dt.total_seconds()
                df["event_elapsed_time"] = elapsed.fillna(0.0)
            for cid, group in df.groupby(case_col, sort=False):
                case_events[cid] = group.reset_index(drop=True)

        n_samples = len(dataset)
        decision_rows: List[List[Tuple[str, Dict[str, float], float]]] = []

        for idx in range(n_samples):
            case_id = dataset.case_ids[idx]
            prefix_activities = dataset._extract_prefix_activity_labels(idx)

            # Build per-event attribute dicts if we have the source data
            prefix_event_data: Optional[List[Dict[str, Any]]] = None
            if case_id in case_events:
                case_df = case_events[case_id]
                prefix_event_data = []
                ev_cursor = 0
                for act in prefix_activities:
                    if act == "EOS" or act == "":
                        prefix_event_data.append({})
                        continue
                    matched = False
                    for pos in range(ev_cursor, len(case_df)):
                        if case_df.iloc[pos].get("concept:name", None) == act:
                            prefix_event_data.append(case_df.iloc[pos].to_dict())
                            ev_cursor = pos + 1
                            matched = True
                            break
                    if not matched:
                        prefix_event_data.append({})

            # Filter out EOS labels for replay
            replay_activities = [
                a for a in prefix_activities if a != "EOS" and a != ""
            ]
            replay_event_data: Optional[List[Dict[str, Any]]] = None
            if prefix_event_data is not None:
                replay_event_data = [
                    d
                    for a, d in zip(prefix_activities, prefix_event_data)
                    if a != "EOS" and a != ""
                ]

            # Get labels for the real (non-EOS) part of the prefix
            real_labels = self.label_prefix_online(
                replay_activities,
                replay_event_data,
            )

            # Pad back to full prefix length (EOS positions get BOTTOM)
            full_labels: List[Tuple[str, Dict[str, float], float]] = []
            real_cursor = 0
            for act in prefix_activities:
                if act == "EOS" or act == "":
                    full_labels.append((BOTTOM, {}, 0.0))
                else:
                    if real_cursor < len(real_labels):
                        place_name, _, dist, c_i = real_labels[real_cursor]
                        full_labels.append((place_name, dist, c_i))
                        real_cursor += 1
                    else:
                        full_labels.append((BOTTOM, {}, 0.0))

            decision_rows.append(full_labels)

        dataset.set_decision_data(decision_rows)


# ---------------------------------------------------------------------------
# Decision-point diagnostics
# ---------------------------------------------------------------------------

def compute_dp_diagnostics(
    decision_data: List[List[Tuple]],
    true_next_activities: List[List[str]],
    coverage_threshold: float = 0.0,
    n_calibration_bins: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Compute per-decision-place quality diagnostics.

    Parameters
    ----------
    decision_data:
        One list per prefix, one entry per visible event. Supported entry
        formats are ``(p_i, A_i, z_i, c_i)`` and compact
        ``(place_name, dist, c_i)``. BOTTOM entries are skipped.
    true_next_activities:
        Parallel structure to *decision_data*.  Each inner list contains the
        true activity label that *followed* the corresponding event in the
        prefix.  Use ``"EOS"`` for the last event and for padding positions.
    coverage_threshold:
        Minimum coverage required for a decision place to be included in the
        returned dict (0.0 = include all).
    n_calibration_bins:
        Number of equal-width bins used for Expected Calibration Error (ECE).

    Returns
    -------
    dict  {place_name: {metric_name: value}}

    Metrics per place
    -----------------
    support           : number of decision-labeled steps at this place
    entropy_mean      : mean Shannon entropy H(z_i) in nats
    top1_accuracy     : Pr(argmax z_i == true_label)
    top3_accuracy     : Pr(true_label in top-3 of z_i)
    mean_true_prob    : mean z_i(true_label)
    ece               : Expected Calibration Error (confidence vs accuracy)
    coverage          : Pr(true_label ∈ support(z_i))  support = {a: z_i(a)>0}
    """
    from collections import defaultdict
    import math

    # Accumulators keyed by place name
    acc: Dict[str, Dict[str, list]] = defaultdict(lambda: {
        "entropy": [],
        "top1_correct": [],
        "top3_correct": [],
        "true_prob": [],
        "conf": [],          # for ECE: max(z_i)
        "in_support": [],    # for coverage
    })

    for prefix_labels, prefix_true in zip(decision_data, true_next_activities):
        for entry, true_act in zip(prefix_labels, prefix_true):
            if not isinstance(entry, tuple) or len(entry) < 2:
                continue
            place_name = entry[0]
            # Supported tuple layouts:
            # 1) (p_i, A_i, z_i, c_i)
            # 2) (place_name, dist, c_i)
            second = entry[1]
            if isinstance(second, list) and len(entry) >= 3:
                dist = entry[2]
            else:
                dist = second
            if place_name == BOTTOM or not dist:
                continue
            if true_act in ("EOS", "", None):
                continue

            # --- Distribution metrics ---
            probs = list(dist.values())
            # Shannon entropy in nats
            entropy = -sum(p * math.log(p + 1e-12) for p in probs if p > 0)

            sorted_items = sorted(dist.items(), key=lambda x: -x[1])
            top1_label = sorted_items[0][0] if sorted_items else ""
            top3_labels = {itm[0] for itm in sorted_items[:3]}
            true_prob = float(dist.get(true_act, 0.0))
            conf = float(sorted_items[0][1]) if sorted_items else 0.0
            in_support = true_act in dist and dist[true_act] > 0.0

            acc[place_name]["entropy"].append(entropy)
            acc[place_name]["top1_correct"].append(int(top1_label == true_act))
            acc[place_name]["top3_correct"].append(int(true_act in top3_labels))
            acc[place_name]["true_prob"].append(true_prob)
            acc[place_name]["conf"].append(conf)
            acc[place_name]["in_support"].append(int(in_support))

    results: Dict[str, Dict[str, float]] = {}

    for place_name, data in acc.items():
        n = len(data["entropy"])
        if n == 0:
            continue

        coverage_val = float(np.mean(data["in_support"]))
        if coverage_val < coverage_threshold:
            continue

        # ECE: bin by confidence, measure |acc - conf| per bin
        confs = np.array(data["conf"])
        corrects = np.array(data["top1_correct"], dtype=float)
        bin_edges = np.linspace(0.0, 1.0, n_calibration_bins + 1)
        ece = 0.0
        for b in range(n_calibration_bins):
            lo, hi = bin_edges[b], bin_edges[b + 1]
            mask = (confs >= lo) & (confs < hi)
            if b == n_calibration_bins - 1:
                mask = (confs >= lo) & (confs <= hi)
            if mask.sum() == 0:
                continue
            acc_bin = corrects[mask].mean()
            conf_bin = confs[mask].mean()
            ece += (mask.sum() / n) * abs(acc_bin - conf_bin)

        results[place_name] = {
            "support": float(n),
            "entropy_mean": float(np.mean(data["entropy"])),
            "top1_accuracy": float(np.mean(data["top1_correct"])),
            "top3_accuracy": float(np.mean(data["top3_correct"])),
            "mean_true_prob": float(np.mean(data["true_prob"])),
            "ece": float(ece),
            "coverage": coverage_val,
        }

    return results


def extract_true_next_activities(
    event_log_df: "pd.DataFrame",
    sorted_case_ids: List[str],
    alignments: List[Any],
) -> List[List[str]]:
    """Build the per-prefix true-next-activity lists required by compute_dp_diagnostics.

    For each case in *sorted_case_ids* we walk the alignment and collect the
    label of the **next** synchronous log move after each step — mirroring the
    target label the decision model was trained to predict.

    Returns a list (one per case) of lists (one per visible event position) of
    activity label strings.  The last visible event in each case gets ``"EOS"``.
    """
    result: List[List[str]] = []
    for case_id, case_alignment in zip(sorted_case_ids, alignments):
        n = len(case_alignment)
        # Pre-compute first visible label at or AFTER each step (same logic as collect_I)
        first_visible: List[Optional[str]] = [None] * n
        future: Optional[str] = None
        for i in range(n - 1, -1, -1):
            (ln, mn), (ll, ml) = case_alignment[i]
            if ln != ">>" and mn != ">>":
                lbl = ll or ml
                if lbl:
                    future = lbl
            first_visible[i] = future

        # Now compute next-visible (skip the current step's own label).
        # We want: after step i fires, what is the next visible event?
        # That equals first_visible[i+1] (or EOS if none).
        per_case: List[str] = []
        for step_idx, ((log_name, model_name), _) in enumerate(case_alignment):
            if model_name == ">>":
                continue  # log-only move: no visible event from net
            # Only emit an entry for visible (non-silent) transitions in the net
            # that correspond to a synchronous move — same filter as collect_I
            # uses when building I[p].
            if log_name == ">>":
                continue  # model-only (silent or skip) move: no event label
            nva_idx = step_idx + 1
            nva = first_visible[nva_idx] if nva_idx < n else None
            per_case.append(nva if nva is not None else "EOS")

        result.append(per_case)

    return result
