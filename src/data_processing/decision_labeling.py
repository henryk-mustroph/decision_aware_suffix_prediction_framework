"""
Decision-aware event labeling for suffix prediction.

For each visible event in a trace we attach a pair ``(p_i, z_i)``:
    p_i  -- the decision place reached by the event (sentinel ``BOTTOM`` if
            the event does not enable a decision place).
    z_i  -- the decision model's probability distribution over the activities
            that can directly follow at p_i (empty when p_i == BOTTOM).

Labeling is done offline from optimal alignments produced by the decision
miner; the runtime online use of the same decision models during
autoregressive decoding lives in
``suffix_pred.decision_rule_guided_reasoning_inference``.
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import pickle
import json
from pathlib import Path

from decision_mining.decision_discovery import replay_alignment_decisions

_MODULE_RENAMES = {"decision_mining.custom_framework.decision_discovery": "decision_mining.decision_discovery",
                   "decision_mining.custom_framework.function_estimator_catboost_advanced": "decision_mining.function_estimator_catboost_advanced"}

class _CompatUnpickler(pickle.Unpickler):
    """
    Redirect old module paths so that legacy .pkl files load correctly.
    """
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


# Sentinel for non-decision events.
BOTTOM = "⊥"


class DecisionLabeler:
    """
    Produces decision-aware event labels for prefix datasets.

    Args:
    - petri_net : tuple: (net, initial_marking, final_marking) from pm4py.
    - decision_model_dir : str: Path to directory containing per-place .pkl estimator files.
    - decision_places_bundle_path : str: Path to the decision_places_bundle.json produced by DecisionDiscovery.save_results.
    - dynamic_attributes, static_attributes : list[str] | None: (The same attribute lists given to DecisionDiscovery)! during model training.
    """
    def __init__(self,
                 petri_net: Tuple,
                 decision_model_dir: str,
                 decision_places_bundle_path: str,
                 dynamic_attributes: Optional[List[str]] = None,
                 static_attributes: Optional[List[str]] = None,
                 numeric_scalers: Optional[Dict[str, Any]] = None) -> None:
        """
        numeric_scalers: optional mapping ``column_name -> fitted sklearn-style
        transformer``. Applied to numeric columns of the event log during
        offline labeling so the test-time feature space matches the one the
        decision miner trained on (which itself matches the LSTM's scaled
        runtime features). If not provided, the labeler attempts to load a
        sibling ``numeric_scalers.pkl`` next to the bundle JSON; pass an
        explicit empty dict to disable auto-loading.
        """


        self.net, self.im, self.fm = petri_net

        self.dynamic_attributes = list(dynamic_attributes or [])
        self.static_attributes = list(static_attributes or [])
        self.past_attr_keys = self.dynamic_attributes + self.static_attributes

        # Resolve numeric scalers: explicit argument wins; otherwise auto-load
        # a sibling pickle saved by DecisionDiscovery.save_results.
        if numeric_scalers is None:
            scaler_candidate = Path(decision_places_bundle_path).parent / "numeric_scalers.pkl"
            if scaler_candidate.exists():
                with open(scaler_candidate, "rb") as f:
                    loaded_scalers = pickle.load(f)
                self.numeric_scalers: Dict[str, Any] = dict(loaded_scalers or {})
            else:
                self.numeric_scalers = {}
        else:
            self.numeric_scalers = dict(numeric_scalers or {})

        # transition lookup
        self.transition_by_name: Dict[str, Any] = {str(t.name): t for t in self.net.transitions}
        self.transition_by_label: Dict[str, Any] = {}
        for t in self.net.transitions:
            if t.label is not None:
                self.transition_by_label.setdefault(str(t.label), []).append(t)

        # decision places
        self.decision_places: List[Any] = [p for p in self.net.places if len(p.out_arcs) > 1]
        self.decision_place_names: set = {str(p) for p in self.decision_places}
        self.decision_place_by_name: Dict[str, Any] = {str(p): p for p in self.decision_places}

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

            # Prioritise decision_model_dir
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
                    model_path = (Path(decision_places_bundle_path).parent / model_path_str)

            if model_path.exists():
                with open(model_path, "rb") as f:
                    loaded = _compat_unpickle(f)
                    self.estimators[place_name] = _load_estimator_artifact(loaded)

    def _apply_numeric_scalers_to_df(self, df: pd.DataFrame) -> None:
        """
        Transform numeric columns of ``df`` in-place with the persisted
        scalers. Mirrors DecisionDiscovery._apply_numeric_scalers.
        """
        if not self.numeric_scalers:
            return
        for col, scaler in self.numeric_scalers.items():
            if col not in df.columns:
                continue
            series = pd.to_numeric(df[col], errors="coerce")
            mask = series.notna()
            if not mask.any():
                continue
            reshaped = series[mask].to_numpy().reshape(-1, 1)
            transformed = np.asarray(scaler.transform(reshaped)).reshape(-1)
            new_values = series.astype(float).copy()
            new_values.loc[mask] = transformed
            df[col] = new_values

    # Feature-row building: same as DecisionDiscovery._build_feature_row and _filter_attributes
    def _filter_attributes(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
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

    def _build_feature_row(self, past_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Mirror of DecisionDiscovery.build_feature_row.

        Static attributes pass through as single columns (latest non-null
        observation). Each dynamic attribute emits two columns:
          - ``<attr>``           value at the most recent observed event
                                 (position j)
          - ``<attr>_past_avg``  mean over all past events (including j)
                                 for numeric attributes
          - ``<attr>_past_mode`` most-frequent value across the same set
                                 (categorical attributes)
        """
        from collections import Counter

        features: Dict[str, Any] = {}
        if not past_events:
            return features

        static_set = set(self.static_attributes)
        if self.dynamic_attributes:
            allowed_dyn = set(self.dynamic_attributes)
        else:
            observed: set = set()
            for ev in past_events:
                observed.update(ev.keys())
            allowed_dyn = observed - static_set

        # Static columns: latest non-null value across past events.
        if static_set:
            all_keys: set = set()
            for ev in past_events:
                all_keys.update(ev.keys())
            for key in sorted(all_keys & static_set):
                latest: Any = np.nan
                for ev in past_events:
                    v = ev.get(key, None)
                    if v is None:
                        continue
                    if isinstance(v, (float, np.floating)) and np.isnan(v):
                        continue
                    latest = v
                if isinstance(latest, (int, np.integer)):
                    latest = str(int(latest))
                features[key] = latest

        previous_event = past_events[-1]
        # Match the training-side spec: aggregate over earlier synchronous
        # events only (r < j), excluding the most recent (at j).
        history_events = past_events[:-1]

        for key in sorted(allowed_dyn):
            last_val = previous_event.get(key, np.nan)
            if isinstance(last_val, (int, np.integer)):
                last_val = str(int(last_val))
            features[key] = last_val

            # Historical summary over all sync events before j (EXCLUDING j).
            seq = [ev.get(key, np.nan) for ev in history_events]
            valid = [v for v in seq
                     if v is not None and not (isinstance(v, (float, np.floating)) and np.isnan(v))]
            is_numeric_last = (isinstance(last_val, (float, np.floating))
                               and not (isinstance(last_val, float) and np.isnan(last_val)))
            is_numeric_history = any(isinstance(v, (float, np.floating)) for v in valid)
            is_numeric = is_numeric_history or (not valid and is_numeric_last)

            if is_numeric:
                if valid:
                    arr = np.array([float(v) for v in valid if isinstance(v, (float, np.floating))],
                                   dtype=float)
                    features[f"{key}_past_avg"] = float(np.mean(arr))
            else:
                if valid:
                    counts = Counter(str(v) for v in valid)
                    features[f"{key}_past_mode"] = counts.most_common(1)[0][0]
                elif last_val is not None and not (isinstance(last_val, float) and np.isnan(last_val)):
                    features[f"{key}_past_mode"] = str(last_val)

        return features

    # Shallow (proximate) prediction: stops at downstream decision places
    def _predict_shallow(self,
                         place_name: str,
                         past_events: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Predict next visible event label at place (decision point).
        The decision models are trained to predict next visible event label directly (not outgoing transitions)

        Returns:
        - activity_dist: dict with {activity_label: probability}.
        """
        estimator = self.estimators.get(place_name)
        if estimator is None:
            return {}

        feature_row = self._build_feature_row(past_events)
        labels, probs = estimator.predict_proba(feature_row)

        dist = {lbl: float(p) for lbl, p in zip(labels, probs.flatten()) if p > 0}
        return dist

    # Offline labeling (training): uses optimal alignments
    def _places_after_transition(self, trans: Any) -> List[Any]:
        """
        Return the decision places in t-bullet (output places of *trans*).
        """
        return [arc.target for arc in trans.out_arcs if arc.target in self.decision_places]

    def _collect_sync_events(self,
                             case: pd.DataFrame,
                             case_alignment: List[Any]) -> List[Dict[str, Any]]:
        """Ordered (filtered) attribute dicts, one per synchronous move."""
        sync_events: List[Dict[str, Any]] = []
        cursor = 0
        for (log_name, model_name), (log_label, model_label) in case_alignment:
            if model_name == ">>" or log_name == ">>":
                continue
            if self.transition_by_name.get(model_name) is None:
                continue
            candidate_labels = [lbl for lbl in [log_label, model_label] if lbl]
            matched: Dict[str, Any] = {}
            for pos in range(cursor, len(case)):
                if case.iloc[pos].get("concept:name", None) in candidate_labels:
                    matched = self._filter_attributes(case.iloc[pos].to_dict())
                    cursor = pos + 1
                    break
            sync_events.append(matched)
        return sync_events

    def _prepare_df(self, event_log_df: pd.DataFrame) -> pd.DataFrame:
        """Add elapsed-time columns and apply numeric scalers, as the miner did."""
        df = event_log_df.copy()
        case_col = "case:concept:name"
        ts_col = "time:timestamp"
        if ts_col in df.columns:
            case_start = df.groupby(case_col)[ts_col].transform("min")
            df["case_elapsed_time"] = (df[ts_col] - case_start).dt.total_seconds()
            elapsed = df.groupby(case_col)[ts_col].diff().dt.total_seconds()
            df["event_elapsed_time"] = elapsed.fillna(0.0)
        self._apply_numeric_scalers_to_df(df)
        return df

    def label_traces_offline(self,
                             event_log_df: pd.DataFrame,
                             sorted_case_ids: List[str],
                             alignments: List[Any]) -> Dict[str, List[Tuple[str, Dict[str, float]]]]:
        """
        Per visible event, the decision place that governs the *next* visible
        event together with the decision model's distribution over the next
        activity label.

        We replay each alignment with :func:`replay_alignment_decisions` (the
        exact routine the decision miner trained on). For the ``k``-th visible
        event ``e_k`` we attach the decision whose branch is actually resolved
        by the next event ``e_{k+1}`` (``resolved_by == k+1``); the proximate
        such decision (created last) is used. This keys on the resolving event
        rather than global event order, so a decision sitting on a *concurrent*
        branch (resolved by a different event) is not mis-attached here. The
        final event predicts ``EOS`` via a branch that ends the case
        (``resolved_by is None``). Events whose next step crosses no decision
        place get ``(BOTTOM, {})``.

        Outputs ``{case_id: [(p, z), ...]}`` with one entry per visible event,
        index-aligned with :func:`extract_true_next_activities`.
        """
        df = self._prepare_df(event_log_df)
        case_col = "case:concept:name"
        decision_places = set(self.decision_places)
        result: Dict[str, List[Tuple[str, Dict[str, float]]]] = {}

        for case_id, case_alignment in zip(sorted_case_ids, alignments):
            case = df[df[case_col] == case_id].reset_index(drop=True)
            sync_events = self._collect_sync_events(case, case_alignment)
            n_events = len(sync_events)

            records = replay_alignment_decisions(case_alignment,
                                                 im=self.im,
                                                 transition_by_name=self.transition_by_name,
                                                 decision_places=decision_places)
            # Proximate decision keyed on the event that resolves it. Bucket
            # `n_events` collects the case-ending (EOS) decisions.
            proximate: Dict[int, Any] = {}
            for rec in records:
                key = n_events if rec["resolved_by"] is None else int(rec["resolved_by"])
                if key not in proximate or rec["order"] > proximate[key]["order"]:
                    proximate[key] = rec

            event_labels: List[Tuple[str, Dict[str, float]]] = []
            for k in range(n_events):
                rec = proximate.get(k + 1)      # decision resolved by e_{k+1}
                if rec is None:
                    event_labels.append((BOTTOM, {}))
                    continue
                dp_name = str(rec["place"])
                z = self._predict_shallow(dp_name, sync_events[:rec["sync_index"]])
                event_labels.append((dp_name, z))

            result[case_id] = event_labels

        return result

    def collect_decision_instances(self,
                                   event_log_df: pd.DataFrame,
                                   sorted_case_ids: List[str],
                                   alignments: List[Any]
                                   ) -> Tuple[List[List[Tuple[str, Dict[str, float]]]],
                                              List[List[str]]]:
        """
        Per-decision-point evaluation instances, mirroring the miner's
        ``collect_I`` exactly: one entry per decision-place firing.

        Returns ``(decision_data, true_next)`` where, for each case,
        ``decision_data[c][i] == (place_name, z)`` is the model's distribution
        over the next activity at that firing and ``true_next[c][i]`` is the
        replay-resolved target it was trained on. Because each firing is scored
        against its *own* branch's target, concurrent branches do not
        contaminate each other (unlike the per-visible-event view). Feed both
        straight into :func:`compute_dp_diagnostics`.
        """
        df = self._prepare_df(event_log_df)
        case_col = "case:concept:name"
        decision_places = set(self.decision_places)

        decision_data: List[List[Tuple[str, Dict[str, float]]]] = []
        true_next: List[List[str]] = []

        for case_id, case_alignment in zip(sorted_case_ids, alignments):
            case = df[df[case_col] == case_id].reset_index(drop=True)
            sync_events = self._collect_sync_events(case, case_alignment)
            records = replay_alignment_decisions(case_alignment,
                                                 im=self.im,
                                                 transition_by_name=self.transition_by_name,
                                                 decision_places=decision_places)
            per_data: List[Tuple[str, Dict[str, float]]] = []
            per_true: List[str] = []
            for rec in records:
                dp_name = str(rec["place"])
                z = self._predict_shallow(dp_name, sync_events[:rec["sync_index"]])
                per_data.append((dp_name, z))
                per_true.append(rec["target"])
            decision_data.append(per_data)
            true_next.append(per_true)

        return decision_data, true_next

    # Batch labeling for EventLogDataset
    def label_dataset_offline(self,
                              dataset: Any,
                              event_log_df: pd.DataFrame,
                              sorted_case_ids: List[str],
                              alignments: List[Any]) -> None:
        """
        Label all samples in an EventLogDataset using offline alignments.

        Every active (non-padding) position of each sample receives a label,
        which means both the encoder-side prefix and the decoder-side suffix
        of teacher-forced suffix training are decision-labeled.
        """
        # 1) Label full traces
        trace_labels = self.label_traces_offline(event_log_df, sorted_case_ids, alignments)

        # 2) Map to sample rows in the dataset. For each sample, walk every
        # active position; non-EOS positions consume the next visible-event
        # label of the trace, EOS / empty positions get BOTTOM.
        decision_rows: List[List[Tuple[str, Dict[str, float]]]] = []

        for idx in range(len(dataset)):
            case_id = dataset.case_ids[idx]
            active_len = dataset._prefix_length_from_zero_mask(dataset.zero_padding[idx])
            active_activities = dataset._extract_prefix_activity_labels(idx)
            full_trace_labels = trace_labels.get(case_id, [])

            sample_labels: List[Tuple[str, Dict[str, float]]] = []
            trace_cursor = 0
            for pos in range(active_len):
                act = active_activities[pos] if pos < len(active_activities) else ""
                if act in ("EOS", ""):
                    sample_labels.append((BOTTOM, {}))
                    continue
                if trace_cursor < len(full_trace_labels):
                    sample_labels.append(full_trace_labels[trace_cursor])
                    trace_cursor += 1
                else:
                    sample_labels.append((BOTTOM, {}))

            decision_rows.append(sample_labels)

        dataset.set_decision_data(decision_rows)


def compute_dp_diagnostics(decision_data: List[List[Tuple[str, Dict[str, float]]]],
                           true_next_activities: List[List[str]]) -> Dict[str, Dict[str, float]]:
    """
    Per-decision-place accuracy of the trained decision model on held-out data.

    Inputs
    ------
    decision_data : list aligned with the dataset's case order. Each inner list
        contains one ``(p_i, z_i)`` entry per visible event, as produced by
        ``DecisionLabeler.label_traces_offline``.
    true_next_activities : parallel structure to ``decision_data``. Each inner
        list contains the activity that actually followed the corresponding
        event, or ``"EOS"`` for the trace end.

    Returns
    -------
    ``{place_name: {support, top1_accuracy, top3_accuracy, mean_true_prob}}``.
    """
    from collections import defaultdict, Counter

    acc: Dict[str, Dict[str, list]] = defaultdict(
        lambda: {"top1": [], "top3": [], "true_prob": [], "true_acts": []})

    for prefix_labels, prefix_true in zip(decision_data, true_next_activities):
        for entry, true_act in zip(prefix_labels, prefix_true):
            if not isinstance(entry, tuple) or len(entry) < 2:
                continue
            place_name, dist = entry[0], entry[1]
            if place_name == BOTTOM or not dist or true_act in ("EOS", "", None):
                continue
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            top1 = ranked[0][0]
            top3 = {a for a, _ in ranked[:3]}
            acc[place_name]["top1"].append(int(top1 == true_act))
            acc[place_name]["top3"].append(int(true_act in top3))
            acc[place_name]["true_prob"].append(float(dist.get(true_act, 0.0)))
            acc[place_name]["true_acts"].append(true_act)

    # majority_baseline = relative frequency of the single most common realised
    # next activity ("always predict the majority branch"); n_branches = number
    # of distinct realised outcomes. A model is informative only if its top-1
    # accuracy beats majority_baseline (and the place has >= 2 branches).
    out: Dict[str, Dict[str, float]] = {}
    for place, data in acc.items():
        if not data["top1"]:
            continue
        counts = Counter(data["true_acts"])
        n = len(data["true_acts"])
        out[place] = {"support": float(len(data["top1"])),
                      "top1_accuracy": float(np.mean(data["top1"])),
                      "top3_accuracy": float(np.mean(data["top3"])),
                      "mean_true_prob": float(np.mean(data["true_prob"])),
                      "majority_baseline": float(max(counts.values()) / n) if n else 0.0,
                      "n_branches": int(len(counts))}
    return out


def extract_true_next_activities(decision_data: List[List[Tuple[str, Dict[str, float]]]],
                                 alignments: List[Any],
                                 labeler: "DecisionLabeler") -> List[List[str]]:
    """
    Build the per-event true-next-activity lists required by
    :func:`compute_dp_diagnostics`, index-aligned with
    :func:`DecisionLabeler.label_traces_offline`.

    Entry ``k`` of a case is the activity that actually follows the ``k``-th
    visible event, i.e. the ``(k+1)``-th visible activity of the trace, or
    ``"EOS"`` for the last event. This is exactly the target the proximate
    decision place at entry ``k`` was trained to predict.
    """
    result: List[List[str]] = []
    for case_idx, case_alignment in enumerate(alignments):
        per_case_labels = decision_data[case_idx] if case_idx < len(decision_data) else []

        # Visible activity labels in alignment order.
        sync_labels: List[str] = []
        for (log_name, model_name), (log_label, model_label) in case_alignment:
            if log_name == ">>" or model_name == ">>":
                continue
            if labeler.transition_by_name.get(model_name) is None:
                continue
            sync_labels.append(log_label or model_label or "EOS")

        n = len(per_case_labels)
        per_case: List[str] = []
        for k in range(n):
            per_case.append(sync_labels[k + 1] if (k + 1) < len(sync_labels) else "EOS")
        result.append(per_case)

    return result
