"""
Compact, alignment-driven decision mining adopted for suffix prediction from: 
- De Leoni, and Van Der Aalst. "Data-aware process mining: discovering decisions in processes using alignments." ACM symposium on applied computing. 2013.

Adaptions:
- train the models to predict the next visible event, not a transition in the process model 
(correctly finds the event label of the first synchronous move at k ≥ i (both log and model sides are non->>)
- Use only and also historical events as input for decision mining 
(before the current step's event attributes are appended. Only synchronous moves (log_name != ">>") contribute to history. The current event at position i is never included in its own features)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd

from decision_mining.function_estimator_catboost_advanced import FunctionEstimator as fe_advanced
from decision_mining.function_estimator_catboost_advanced import ModelConfig as mc_advanced

@dataclass
class DecisionPointModel:
    place_name: str
    estimator: Any

@dataclass
class DecisionMiningResult:
    models: Dict[str, DecisionPointModel]
    skipped: List[str]

class DecisionDiscovery:
    def __init__(self,
                 petri_net: Tuple,
                 sorted_case_ids: List[str],
                 event_log_df: pd.DataFrame,
                 alignments: List[Any]) -> None:
        
        self.net, self.im, self.fm = petri_net
        self.sorted_case_ids = sorted_case_ids
        self.event_log_df = event_log_df
        
        # create case elapsed time
        self.__create_case_elapsed_time_column()
        # create event elapsed time: Ensure the same time processing as for the data preparation
        self.__create_event_elapsed_time_column()
        
        self.alignments = alignments

        # map transition name -> transition object
        self.transition_by_key: Dict[str, Any] = {str(t.name): t for t in self.net.transitions}

        # decision points (>1 outgoing)
        self.decision_places: List[Any] = [p for p in self.net.places if len(p.out_arcs) > 1]
        
        # dict: key: decision place, value: list of outgoing transitions from this place
        self.routing_transitions_by_place: Dict[Any, List[Any]] = {p: [arc.target for arc in p.out_arcs] for p in self.decision_places}

        # vice versa: dict: key: transition, value: list of places before
        self.in_place_for_routing_transition: Dict[Any, List[Any]] = defaultdict(list)
        for place, transitions in self.routing_transitions_by_place.items():
            for transition in transitions:
                self.in_place_for_routing_transition[transition].append(place)
                
        print('discovery initialization completed!')
        
    def __create_case_elapsed_time_column(self) -> None:
        """
        Create a new column representing elapsed time since the case start.
        """
        case_col = "case:concept:name"
        ts_col = "time:timestamp"
        
        case_start_times = self.event_log_df.groupby(case_col)[ts_col].transform("min")
        time_offset = self.event_log_df[ts_col] - case_start_times
        time_offset_seconds = time_offset.dt.total_seconds()
        self.event_log_df['case_elapsed_time'] = time_offset_seconds
        self.max_case_length = self.event_log_df.groupby(case_col).size().max()
        
    def __create_event_elapsed_time_column(self) -> None:
        """
        Create a new column representing elapsed time since the previous event.
        - start with 0.0 instead of NaN as in the original encoding
        """
        case_col = "case:concept:name"
        ts_col = "time:timestamp"

        elapsed = self.event_log_df.groupby(case_col)[ts_col].diff().dt.total_seconds()
        # First event per case has no previous timestamp.
        self.event_log_df["event_elapsed_time"] = elapsed.fillna(0.0)

    def _filter_attributes(self, attrs: Dict[str, Any], attributes: Optional[List[str]]) -> Dict[str, Any]:
        if not attrs:
            return {}
        if attributes:
            # Strictly respect the attribute whitelist.
            filtered = {k: attrs.get(k, np.nan) for k in attributes}
        else:
            filtered = attrs

        # ints are IDs/categorical; floats are continuous.
        out: Dict[str, Any] = {}
        for k, v in filtered.items():
            if isinstance(v, (int, np.integer)):
                out[k] = str(int(v))
            else:
                out[k] = v
        return out

    def collect_I(self,
                  dynamic_attributes: Optional[List[str]] = None,
                  static_attributes: Optional[List[str]] = None) -> Tuple[Dict[Any, List[Tuple[Dict[str, Any], Any]]], Dict[Any, List[Any]]]:
        """
        collect training data per decision place
        """
        # example list of alignments: [[(('>>', 'skip_2'), ('>>', None)), (('>>', 'init_loop_3'), ('>>', None)), ... ], ...]
        
        # store train data for each transition place
        I: Dict[Any, List[Tuple[Dict[str, Any], Any]]] = defaultdict(list)
        legend = defaultdict(list)
        
        past_attr_keys = (dynamic_attributes or []) + (static_attributes or [])

        # iterate through alignments:
        for case_id, case_alignment in zip(self.sorted_case_ids, self.alignments):
            case = self.event_log_df[
                self.event_log_df["case:concept:name"] == case_id
            ].reset_index(drop=True)
            case_event_cursor = 0

            # Pre-compute the first visible event label as target
            n_steps = len(case_alignment)
            first_visible_at_or_after = [None] * n_steps
            future_visible = None
            for i in range(n_steps - 1, -1, -1):
                (ln, mn), (ll, ml) = case_alignment[i]
                if ln != ">>" and mn != ">>":
                    label = ll or ml
                    if label:
                        future_visible = label
                first_visible_at_or_after[i] = future_visible

            # past events through the case:
            past_events: List[Dict[str, Any]] = []
            for step_idx, ((log_name, model_name), (log_label, model_label)) in enumerate(case_alignment):
                # if log move only, pass
                if model_name != ">>":
                    # get tranistion:
                    trans = self.transition_by_key.get(model_name)
                    if trans is None:
                        continue
                    # get input places for transition:
                    in_places = self.in_place_for_routing_transition.get(trans, [])
                    decision_in_places = [p for p in in_places if p in self.decision_places]

                    # Add to training data:
                    # Label = next visible activity (not transition name)
                    nva = first_visible_at_or_after[step_idx]
                    if nva is None:
                        nva = "EOS"
                    for p in decision_in_places:
                        attrs = {"past_events": list(past_events)}
                        I[p].append((attrs, nva))
                        legend[p].append(((log_name, model_name), (log_label, model_label)))

                    # Update history with synchronized events only.
                    # Crucially, this happens after sample creation, so the
                    # current branching event never leaks into its own features.
                    if log_name != ">>":
                        candidate_labels = [lbl for lbl in [log_label, model_label] if lbl]
                        matched_event = None
                        for pos in range(case_event_cursor, len(case)):
                            event_label = case.iloc[pos].get("concept:name", None)
                            if event_label in candidate_labels:
                                matched_event = case.iloc[pos]
                                case_event_cursor = pos + 1
                                break

                        if matched_event is not None:
                            ev_dict = matched_event.to_dict()
                            event_attrs_past = self._filter_attributes(ev_dict, past_attr_keys)
                            past_events.append(event_attrs_past)
        return I, legend

    def _build_feature_row(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        """
        bulds train features
        """
        
        attrs = dict(attrs) if attrs else {}
        past_events = attrs.pop("past_events", [])
        features: Dict[str, Any] = dict(attrs)

        # Minimal history indicators.
        features["past_events_count"] = float(len(past_events))

        if not past_events:
            return features

        previous_event = past_events[-1]
        older_events = past_events[:-1]
        features["older_events_count"] = float(len(older_events))

        # Keep full data for the immediate previous event.
        for key, value in previous_event.items():
            if isinstance(value, (int, np.integer)):
                value = str(int(value))
            features[f"{key}_prev_event"] = value

        # Summary for the older event sequence.
        if older_events:
            summary_keys: set[str] = set()
            for ev in older_events:
                summary_keys.update(ev.keys())

            for key in sorted(summary_keys):
                seq = [ev.get(key, np.nan) for ev in older_events]

                valid_vals = [v for v in seq if not ( v is None or (isinstance(v, (float, np.floating)) and np.isnan(v)))]
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

        # Keep short lag features as additional sequence signal.
        keys: set[str] = set()
        for ev in past_events:
            keys.update(ev.keys())

        for key in sorted(keys):
            seq = [ev.get(key, np.nan) for ev in past_events]

            def _is_float_value(v: Any) -> bool:
                return isinstance(v, (float, np.floating)) and not np.isnan(v)

            # Convention: floats are continuous; ints are categorical IDs.
            is_continuous = any(_is_float_value(v) for v in seq)

            def _as_cat(v: Any) -> Any:
                if v is None:
                    return np.nan
                if isinstance(v, float) and np.isnan(v):
                    return np.nan
                if isinstance(v, bool):
                    return v
                return str(v)

            def _as_num(v: Any) -> Any:
                if isinstance(v, (float, np.floating)):
                    return v
                return np.nan

            getv = _as_num if is_continuous else _as_cat

            # most recent (prev1), then prev2, prev3
            if len(seq) >= 1:
                features[f"{key}_prev1"] = getv(seq[-1])
            if len(seq) >= 2:
                features[f"{key}_prev2"] = getv(seq[-2])
            if len(seq) >= 3:
                features[f"{key}_prev3"] = getv(seq[-3])

        return features

    def mine_decision_models(self,
                             dynamic_attributes: Optional[List[str]] = None,
                             static_attributes: Optional[List[str]] = None,
                             mc_config = None) -> DecisionMiningResult:        
        """
        train and return decision modeles per decision place
        """
        
        # train data and legend
        I, _legend = self.collect_I(dynamic_attributes=dynamic_attributes,
                                    static_attributes=static_attributes)

        # models for places
        models: Dict[str, DecisionPointModel] = {}
        # list of places that have no transition/ training data
        skipped: List[str] = []

        for place, samples in I.items():
            place_name = place
            if not samples:
                skipped.append(place_name)
                continue

            rows: List[Dict[str, Any]] = []
            labels: List[str] = []

            for ass, chosen in samples:
                rows.append(self._build_feature_row(ass))
                labels.append(str(chosen))

            # only contains the past_event_counts as past data
            X_raw = pd.DataFrame(rows)
            
            est = fe_advanced.fit_from_xy(X_raw, labels, feature_cols=None, model_cfg=mc_config or mc_advanced())
            models[place_name] = DecisionPointModel(place_name=place_name, estimator=est)

        return DecisionMiningResult(models=models, skipped=skipped)

    def extract_guards(self,
                       mining_result: DecisionMiningResult,
                       *,
                       use_advanced_estimator: bool = False) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        
        guards_by_place: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for place_name, model in mining_result.models.items():
            est = model.estimator

            if bool(use_advanced_estimator):
                if not hasattr(est, "extract_probabilistic_guards_advanced"):
                    raise RuntimeError(
                        f"Estimator for place '{place_name}' does not support advanced guard extraction. "
                        "Create models using the advanced FunctionEstimator."
                    )
                guards_by_place[place_name] = est.extract_probabilistic_guards_advanced()
            else:
                if not hasattr(est, "extract_probabilistic_guards"):
                    raise RuntimeError(
                        f"Estimator for place '{place_name}' does not support basic guard extraction. "
                        "Create models using the basic FunctionEstimator."
                    )
                guards_by_place[place_name] = est.extract_probabilistic_guards()

        return guards_by_place

    def save_results(self,
                     *,
                     guards: Dict[str, Dict[str, List[Dict[str, Any]]]],
                     mining_result: Optional[DecisionMiningResult] = None,
                     output_dir: Optional[str] = None,
                     guards_json_path: Optional[str] = None,
                     guards_flat_csv_path: Optional[str] = None,
                     skipped_places_path: Optional[str] = None,
                     per_place_json_path: Optional[str] = None,
                     model_dir: Optional[str] = None) -> Dict[str, str]:
        """
        Persist mined decision artifacts to user-configurable paths.
        - Returns a dict with the concrete output paths written.
        """
        base_dir = Path(output_dir) if output_dir is not None else Path.cwd() / "decision_mining_results"
        base_dir.mkdir(parents=True, exist_ok=True)

        guards_json = Path(guards_json_path) if guards_json_path else base_dir / "guards.json"
        guards_csv = Path(guards_flat_csv_path) if guards_flat_csv_path else base_dir / "guards_flat.csv"
        skipped_csv = Path(skipped_places_path) if skipped_places_path else base_dir / "skipped_places.csv"
        per_place_json = Path(per_place_json_path) if per_place_json_path else base_dir / "decision_places_bundle.json"
        models_dir = Path(model_dir) if model_dir else base_dir / "models"

        guards_json.parent.mkdir(parents=True, exist_ok=True)
        guards_csv.parent.mkdir(parents=True, exist_ok=True)
        skipped_csv.parent.mkdir(parents=True, exist_ok=True)
        per_place_json.parent.mkdir(parents=True, exist_ok=True)
        models_dir.mkdir(parents=True, exist_ok=True)

        def _safe_name(name: str) -> str:
            return (str(name).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_"))

        def _incoming_transition_tuples(place_obj: Any) -> list[tuple[str, Optional[str]]]:
            if place_obj is None or not hasattr(place_obj, "in_arcs"):
                return []

            tuples: list[tuple[str, Optional[str]]] = []
            for arc in list(getattr(place_obj, "in_arcs", [])):
                trans = getattr(arc, "source", None)
                if trans is None:
                    continue
                trans_name = str(getattr(trans, "name", ""))
                trans_label = getattr(trans, "label", None)
                label_value = str(trans_label) if trans_label is not None else None
                tuples.append((trans_name, label_value))

            # Deduplicate while preserving order.
            deduped: list[tuple[str, Optional[str]]] = []
            seen: set[tuple[str, Optional[str]]] = set()
            for item in tuples:
                if item in seen:
                    continue
                seen.add(item)
                deduped.append(item)
            return deduped

        def _normalize_guards_for_json(guard_dict: Dict[Any, Dict[Any, List[Dict[str, Any]]]]) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
            normalized: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
            for place_key, by_label in guard_dict.items():
                place_name = str(place_key)
                normalized[place_name] = {}
                for transition_label, guard_list in by_label.items():
                    normalized[place_name][str(transition_label)] = guard_list
            return normalized

        guards_serializable = _normalize_guards_for_json(guards)

        # 1) Full guards as JSON (interpretable + probabilistic + set/range based)
        with guards_json.open("w", encoding="utf-8") as f:
            json.dump(guards_serializable, f, indent=2, ensure_ascii=False, default=str)

        # 1b) Per-decision-place bundle: previous incoming transition tuple(s), trained model path, and guards.
        per_place_records: list[dict[str, Any]] = []
        models_by_place = {} if mining_result is None else dict(mining_result.models)
        for place_key, guard_for_place in guards.items():
            place_name = str(place_key)
            safe_place_name = _safe_name(place_name)
            model_path = models_dir / f"{safe_place_name}.pkl"

            model_obj = None
            if place_key in models_by_place:
                model_obj = models_by_place[place_key]
            else:
                for mk, mv in models_by_place.items():
                    if str(mk) == place_name:
                        model_obj = mv
                        break

            estimator = None if model_obj is None else getattr(model_obj, "estimator", None)
            if estimator is not None:
                with model_path.open("wb") as f:
                    artifact = estimator.to_artifact() if hasattr(estimator, "to_artifact") else estimator
                    pickle.dump(artifact, f)
                model_path_str = str(model_path)
            else:
                model_path_str = ""

            place_obj_for_arcs = place_key if hasattr(place_key, "in_arcs") else None
            if place_obj_for_arcs is None and model_obj is not None:
                place_obj_for_arcs = getattr(model_obj, "place_name", None)
            previous_transitions = _incoming_transition_tuples(place_obj_for_arcs)

            per_place_records.append({"place_name": place_name, "previous_transitions": previous_transitions, "model_path": model_path_str, "guards": guard_for_place})

        with per_place_json.open("w", encoding="utf-8") as f:
            json.dump(per_place_records, f, indent=2, ensure_ascii=False, default=str)

        # 2) Flat guard table for quick analysis
        flat_rows: List[Dict[str, Any]] = []
        for place_name, by_label in guards.items():
            for activity_label, guard_list in by_label.items():
                for g in guard_list:
                    flat_rows.append({"place_name": str(place_name),
                                      "activity_label": str(activity_label),
                                      "rule": g.get("rule", ""),
                                      "raw_rule": g.get("raw_rule", ""),
                                      "prob": g.get("prob", np.nan),
                                      "prob_model": g.get("prob_model", np.nan),
                                      "prob_emp": g.get("prob_emp", np.nan),
                                      "support": g.get("support", np.nan),
                                      "coverage": g.get("coverage", np.nan),
                                      "lift": g.get("lift", np.nan),
                                      "intervals": json.dumps(g.get("intervals", {}), ensure_ascii=False),
                                      "categorical_allowed": json.dumps(g.get("categorical_allowed", {}), ensure_ascii=False),
                                      "categorical_excluded": json.dumps(g.get("categorical_excluded", {}), ensure_ascii=False)})
        pd.DataFrame(flat_rows).to_csv(guards_csv, index=False)

        # 3) Skipped places
        skipped = [] if mining_result is None else list(mining_result.skipped)
        pd.DataFrame({"skipped_place": [str(p) for p in skipped]}).to_csv(skipped_csv, index=False)

        return {"output_dir": str(base_dir),
                "guards_json_path": str(guards_json),
                "guards_flat_csv_path": str(guards_csv),
                "skipped_places_path": str(skipped_csv),
                "per_place_json_path": str(per_place_json),
                "model_dir": str(models_dir)}

    def print_summary_and_visualize(self, guards: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
        """
        Print a summary of event label decision rules.

        Guards are keyed by event label (the next visible event the decision model predicts), not by outgoing transition name.
        """

        def _guard_prob(g: Dict[str, Any]) -> float:
            if "prob_model" in g:
                return float(g.get("prob_model", 0.0))
            if "prob" in g:
                return float(g.get("prob", 0.0))
            return float(g.get("prob_emp", 0.0))

        def _format_guard(g: Dict[str, Any]) -> str:
            parts = []
            if g.get("intervals"):
                parts.append("intervals=" + str(g["intervals"]))
            if g.get("categorical_allowed") or g.get("categorical_excluded"):
                parts.append(
                    "cats=" + str({
                        "allowed": g.get("categorical_allowed", {}),
                        "excluded": g.get("categorical_excluded", {}),
                    }))

            parts.append(f"p={_guard_prob(g):.3f}")
            if "support" in g:
                parts.append(f"n={g.get('support', 0)}")
            if "lift" in g:
                parts.append(f"lift={g.get('lift', 0.0):.2f}")

            rule = g.get("rule", "")
            raw_rule = g.get("raw_rule", "")
            if rule and rule != "(true)":
                parts.append("rule=" + rule)
            elif raw_rule and raw_rule != "(true)":
                parts.append("raw_rule=" + raw_rule)
            return "; ".join(parts)

        # Text summary
        for place_name, by_activity in guards.items():
            print(f"\n=== {place_name} ===")
            if not by_activity:
                print("  (no rules emitted for this place)")
                continue
            for activity_label, guard_list in by_activity.items():
                print(f"  → {activity_label} ({len(guard_list)} rules)")
                for g in guard_list:
                    print("    *", _format_guard(g))

    def visualize_bpmn_with_rules(self, guards: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
        """
        Convert the Petri net to BPMN and annotate tasks with decision-rule notes.
        
        This is probably more suitable than the data-aware Petri net annotaiton, since we cannot annotate 
        silent transitions but only real event label transitions. Silent transitions are not shown in a BPMN

        Each task that appears as a predicted activity in the decision models gets a "note" box (yellow)
        attached via a dotted line, showing which decision places predict it and with what probability / rule.
        """
        from pm4py.objects.conversion.wf_net.variants import to_bpmn
        from pm4py.objects.bpmn.obj import BPMN as BPMNObj
        from pm4py.visualization.bpmn import visualizer as bpmn_vis

        def _guard_prob(g: Dict[str, Any]) -> float:
            if "prob_model" in g:
                return float(g.get("prob_model", 0.0))
            if "prob" in g:
                return float(g.get("prob", 0.0))
            return float(g.get("prob_emp", 0.0))

        def _best_rule_for_activity(guard_list):
            if not guard_list:
                return ""
            best = max(guard_list, key=_guard_prob)
            rule = best.get("rule", "")
            prob = _guard_prob(best)
            if rule and rule != "(true)":
                return f"{rule} (p={prob:.2f})"
            return f"p={prob:.2f}"

        bpmn_graph = to_bpmn.apply(self.net, self.im, self.fm)
        gviz = bpmn_vis.apply(bpmn_graph)

        # Invert guards: activity → list of (place_name, best_rule_summary)
        activity_notes: Dict[str, list] = {}
        for place_name, by_activity in guards.items():
            for activity_label, guard_list in by_activity.items():
                summary = _best_rule_for_activity(guard_list)
                if summary:
                    activity_notes.setdefault(activity_label, []).append(
                        f"{place_name}: {summary}"
                    )

        # Find BPMN task node IDs in graphviz and attach annotations
        task_gv_ids: Dict[str, str] = {}
        for node in bpmn_graph.get_nodes():
            if isinstance(node, BPMNObj.Task):
                task_gv_ids[node.get_name()] = str(id(node))

        for activity_label, notes in activity_notes.items():
            gv_id = task_gv_ids.get(activity_label)
            if not gv_id:
                continue
            annotation = "\n".join(notes)
            note_id = f"rule_{activity_label.replace(' ', '_')}"
            gviz.node(note_id, label=annotation,
                      shape="note", style="filled", fillcolor="#ffffcc",
                      fontsize="8", fontname="Helvetica")
            gviz.edge(note_id, gv_id,
                      style="dotted", arrowhead="none", constraint="false")

        bpmn_vis.view(gviz)