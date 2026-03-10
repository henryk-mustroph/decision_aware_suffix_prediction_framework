"""
Decision-aware event labeling for suffix prediction.

Implements the labeling described in the paper:
- For each event in a prefix, determine whether the corresponding transition
  leads to a decision point (place with |p•| > 1).
- If so, build the data state from past events (up to and including the current
  event), query the trained decision model, and produce a soft distribution over
  the next reachable event labels (ν-mapping for silent transitions).
- Non-decision events receive a sentinel label (⊥).

Two modes:
  * Offline (training): uses optimal alignments from decision mining.
  * Online (inference): uses token-based replay on prefixes.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pickle

from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.obj import Marking


# ---------------------------------------------------------------------------
# Compatibility unpickler: the .pkl models were saved when the modules lived
# under ``decision_mining.custom_framework.*``.  They have since been moved to
# ``decision_mining.*``.  This unpickler transparently remaps the old paths.
# ---------------------------------------------------------------------------
_MODULE_RENAMES = {
    "decision_mining.custom_framework.function_estimator_DT_basic":
        "decision_mining.function_estimator_DT_basic",
    "decision_mining.custom_framework.function_estimator_catboost_advanced":
        "decision_mining.function_estimator_catboost_advanced",
    "decision_mining.custom_framework.decision_discovery":
        "decision_mining.decision_discovery",
}


class _CompatUnpickler(pickle.Unpickler):
    """Redirect old module paths so that legacy .pkl files load correctly."""

    def find_class(self, module: str, name: str):
        module = _MODULE_RENAMES.get(module, module)
        return super().find_class(module, name)


def _compat_unpickle(f):
    return _CompatUnpickler(f).load()


# Sentinel for non-decision events
BOTTOM = "⊥"


def _resolve_nu(
    transition: Any,
    net: Any,
    *,
    _visited: Optional[set] = None,
) -> Optional[Any]:
    """Resolve ν(t): the first reachable non-silent transition after *transition*.

    If *transition* itself is non-silent (has a label), return it directly.
    Otherwise follow the unique path of silent transitions through the net.
    Returns ``None`` if no non-silent transition is reachable.
    """
    if _visited is None:
        _visited = set()

    if transition in _visited:
        return None
    _visited.add(transition)

    if getattr(transition, "label", None) is not None:
        return transition

    # transition is silent – follow its output places
    for out_arc in transition.out_arcs:
        place = out_arc.target
        for place_out_arc in place.out_arcs:
            next_trans = place_out_arc.target
            result = _resolve_nu(next_trans, net, _visited=_visited)
            if result is not None:
                return result
    return None


def _build_nu_mapping(
    decision_place: Any,
    net: Any,
) -> Dict[Any, Optional[Any]]:
    """Build {outgoing_transition: ν(outgoing_transition)} for a decision place."""
    mapping: Dict[Any, Optional[Any]] = {}
    for arc in decision_place.out_arcs:
        t = arc.target
        mapping[t] = _resolve_nu(t, net)
    return mapping


def _build_soft_distribution(
    transition_probs: Dict[str, float],
    nu_mapping: Dict[Any, Optional[Any]],
    transition_by_name: Dict[str, Any],
) -> Dict[str, float]:
    """Convert decision-model transition probabilities to event-label probabilities.

    Uses the ν-mapping to aggregate probability mass of silent transitions onto
    the first reachable non-silent transition's label.

    Returns a dict {event_label: probability} (normalised to sum to 1).
    """
    label_mass: Dict[str, float] = defaultdict(float)
    total_defined = 0.0

    for trans_obj, nu_trans in nu_mapping.items():
        trans_name = str(trans_obj.name)
        prob = transition_probs.get(trans_name, 0.0)
        if nu_trans is not None and getattr(nu_trans, "label", None) is not None:
            label_mass[str(nu_trans.label)] += prob
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
        self.nu_mappings: Dict[str, Dict[Any, Optional[Any]]] = {}
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
            if candidate.exists():
                model_path = candidate
            else:
                model_path = Path(model_path_str)
                if not model_path.is_absolute():
                    model_path = (
                        Path(decision_places_bundle_path).parent / model_path_str
                    )
            if model_path.exists():
                with open(model_path, "rb") as f:
                    self.estimators[place_name] = _compat_unpickle(f)

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
    ) -> Dict[str, float]:
        """Query the decision model at *place_name* and return a soft
        next-event-label distribution z_i(a)."""
        estimator = self.estimators.get(place_name)
        if estimator is None:
            return {}

        feature_row = self._build_feature_row(past_events)
        transition_labels, probs = estimator.predict_proba(feature_row)
        transition_probs = dict(zip(transition_labels, probs.tolist()))

        nu_map = self.nu_mappings.get(place_name, {})
        if not nu_map:
            return {}

        return _build_soft_distribution(
            transition_probs, nu_map, self.transition_by_name
        )

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
    ) -> Dict[str, List[List[Tuple[str, Dict[str, float]]]]]:
        """Label every event in every trace using optimal alignments.

        Returns
        -------
        dict  {case_id: list-of-event-labels}
            Each event label is ``(place_name, z_i)`` for a decision event,
            or ``(BOTTOM, {})`` for a non-decision event.
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

        result: Dict[str, List[List[Tuple[str, Dict[str, float]]]]] = {}

        for case_id, case_alignment in zip(sorted_case_ids, alignments):
            case = df[df[case_col] == case_id].reset_index(drop=True)
            case_event_cursor = 0
            past_events: List[Dict[str, Any]] = []
            event_labels: List[Tuple[str, Dict[str, float]]] = []

            for (log_name, model_name), (log_label, model_label) in case_alignment:
                is_sync = (log_name != ">>") and (model_name != ">>")
                is_model_move = (log_name == ">>") and (model_name != ">>")

                if model_name == ">>":
                    # log-only move — no model transition fired
                    continue

                trans = self.transition_by_name.get(model_name)
                if trans is None:
                    continue

                # Identify decision places in t•
                decision_out_places = self._places_after_transition(trans)

                # Record label for synchronized (visible) events only
                if is_sync:
                    if decision_out_places:
                        # Select the one decision place (structured nets have
                        # at most one). If multiple exist pick the first.
                        dp = decision_out_places[0]
                        place_name = str(dp)

                        # Data state η_i includes the current event
                        # (up to and including e_i). We build past_events
                        # *after* appending the current event.
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
                            past_events.append(event_attrs)

                        z_i = self._predict_at_place(place_name, past_events)
                        event_labels.append((place_name, z_i))
                    else:
                        # Non-decision event — still advance the log cursor
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
                            past_events.append(event_attrs)

                        event_labels.append((BOTTOM, {}))
                elif is_model_move:
                    # Silent/model-only move — no visible event, don't append a label
                    pass

            result[case_id] = event_labels

        return result

    # ------------------------------------------------------------------
    # Online labeling (inference): uses token-based replay on prefixes
    # ------------------------------------------------------------------

    def label_prefix_online(
        self,
        prefix_activities: List[str],
        prefix_event_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[str, Dict[str, float]]]:
        """Label a single prefix using token-based replay.

        Parameters
        ----------
        prefix_activities : list[str]
            Sequence of activity labels (event labels) in the prefix.
        prefix_event_data : list[dict] | None
            One dict per event with the raw attribute values. If ``None``,
            only structural (count-based) features are available.

        Returns
        -------
        list of (place_name | BOTTOM, z_i | {})
        """
        labels: List[Tuple[str, Dict[str, float]]] = []
        past_events: List[Dict[str, Any]] = []

        # Replay incrementally: after each event, check if the reached marking
        # contains a decision place.
        for event_idx, activity in enumerate(prefix_activities):
            # Replay the prefix up to (and including) this event
            trace_so_far = Trace(
                [Event({"concept:name": act}) for act in prefix_activities[: event_idx + 1]]
            )
            replayed = token_replay.apply(
                log=[trace_so_far],
                net=self.net,
                initial_marking=self.im,
                final_marking=Marking(),
            )
            reached_marking = replayed[0]["reached_marking"]

            # Update data state
            if prefix_event_data is not None and event_idx < len(prefix_event_data):
                event_attrs = self._filter_attributes(prefix_event_data[event_idx])
                past_events.append(event_attrs)

            # Check if any marked place is a decision place
            decision_place_name = None
            for place in reached_marking:
                if str(place) in self.decision_place_names:
                    decision_place_name = str(place)
                    break

            if decision_place_name is not None:
                z_i = self._predict_at_place(decision_place_name, past_events)
                labels.append((decision_place_name, z_i))
            else:
                labels.append((BOTTOM, {}))

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
        ``(activity_label, {next_event_label: probability})``.

        For non-decision events the entry is ``(BOTTOM, {})``.
        For events beyond the trace length (EOS padding), the entry is
        ``(BOTTOM, {})``.
        """
        # 1) Label full traces
        trace_labels = self.label_traces_offline(
            event_log_df, sorted_case_ids, alignments
        )

        # 2) Map to prefix rows in the dataset
        n_samples = len(dataset)
        decision_rows: List[List[Tuple[str, Dict[str, float]]]] = []

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
            prefix_decision_labels: List[Tuple[str, Dict[str, float]]] = []
            trace_cursor = 0
            for pos in range(prefix_len):
                act = prefix_activities[pos] if pos < len(prefix_activities) else ""
                if act == "EOS" or act == "":
                    prefix_decision_labels.append((BOTTOM, {}))
                    continue

                # Find matching event in the trace labels
                matched = False
                while trace_cursor < len(full_trace_labels):
                    trace_entry = full_trace_labels[trace_cursor]
                    # The trace_labels list is in order of visible events,
                    # so just advance the cursor
                    trace_cursor += 1
                    prefix_decision_labels.append(trace_entry)
                    matched = True
                    break

                if not matched:
                    prefix_decision_labels.append((BOTTOM, {}))

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
        decision_rows: List[List[Tuple[str, Dict[str, float]]]] = []

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
            full_labels: List[Tuple[str, Dict[str, float]]] = []
            real_cursor = 0
            for act in prefix_activities:
                if act == "EOS" or act == "":
                    full_labels.append((BOTTOM, {}))
                else:
                    if real_cursor < len(real_labels):
                        full_labels.append(real_labels[real_cursor])
                        real_cursor += 1
                    else:
                        full_labels.append((BOTTOM, {}))

            decision_rows.append(full_labels)

        dataset.set_decision_data(decision_rows)
