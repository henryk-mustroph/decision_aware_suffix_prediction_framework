"""
Compact, alignment-driven decision mining. 
Based on algorithm from:
- De Leoni, and Van Der Aalst. "Data-aware process mining: discovering decisions in processes using alignments." ACM symposium on applied computing. 2013.

- Adaptions:
Use also historical events as input for decision mining

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import re
import numpy as np
import pandas as pd
from pm4py.visualization.petri_net import visualizer as pn_vis

from decision_mining.custom_framework.function_estimator_DT_basic import FunctionEstimator as fe_basic
from decision_mining.custom_framework.function_estimator_catboost_advanced import FunctionEstimator as fe_advanced

from decision_mining.custom_framework.function_estimator_DT_basic import ModelConfig as mc_basic
from decision_mining.custom_framework.function_estimator_catboost_advanced import ModelConfig as mc_advanced

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

        # decision places (>1 outgoing)
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
        # big problem of algorithm: if (>>, None) is my log move, so a silent transition is fired: I do no know which transition it is: solved
        # Now I have for each alignment a list of tuples: a tuple: (('>>', 'skip_2'), ('>>', None)) first element has name of transition taken

        # list of alignments: [[(('>>', 'skip_2'), ('>>', None)), (('>>', 'init_loop_3'), ('>>', None)), ... ], ...]
        
        # store train data for each transition place
        I: Dict[Any, List[Tuple[Dict[str, Any], Any]]] = defaultdict(list)
        legend = defaultdict(list)
        
        current_attr_keys = (dynamic_attributes or []) + (static_attributes or [])
        past_attr_keys = dynamic_attributes or []

        # iterate through alignments:
        for case_id, case_alignment in zip(self.sorted_case_ids, self.alignments):
            # past events through the case:
            past_events: List[Dict[str, Any]] = []
            for (log_name, model_name), (log_label, model_label) in case_alignment:
                # if log move only, pass
                if model_name != ">>":
                    # get tranistion:
                    trans = self.transition_by_key.get(model_name)
                    if trans is None:
                        continue
                    # get input places for transition:
                    in_places = self.in_place_for_routing_transition.get(trans, [])

                    # Add to training data:
                    # if places is not None
                    for p in in_places:
                        # if place is a transition place
                        if p in self.decision_places:
                            # check if sync move
                            if log_name != ">>":
                                attrs: Dict[str, Any] = {}
                                case = self.event_log_df[self.event_log_df["case:concept:name"] == case_id]
                                event = case[case["concept:name"] == model_label]

                                if not event.empty:
                                    ev_dict = event.iloc[0].to_dict()
                                    event_attrs_current = self._filter_attributes(ev_dict, current_attr_keys)
                                    event_attrs_past = self._filter_attributes(ev_dict, past_attr_keys)
                                    attrs = dict(event_attrs_current)
                                attrs["past_events"] = list(past_events)

                                # add log data
                                I[p].append((attrs, model_name))
                            else:
                                attrs = {}
                                attrs["past_events"] = list(past_events)
                                I[p].append((attrs, model_name))

                            if log_name != ">>" and not event.empty:
                                # Store only dynamic attributes in past_events.
                                past_events.append(event_attrs_past)

                            legend[p].append(((log_name, model_name), (log_label, model_label)))
        return I, legend

    def _build_feature_row(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        attrs = dict(attrs) if attrs else {}
        past_events = attrs.pop("past_events", [])
        features: Dict[str, Any] = dict(attrs)

        # Minimal history indicators (useful for guards + for downsampling "empty" silent rows).
        features["past_events_count"] = float(len(past_events))

        if not past_events:
            return features

        # Minimal sequential encoding: keep only the most recent values.
        # This preserves order information without exploding feature count.
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
                             method: Optional[str] = 'basic',
                             dynamic_attributes: Optional[List[str]] = None,
                             static_attributes: Optional[List[str]] = None,
                             mc_config = None) -> DecisionMiningResult:
        # maybe add model config!
        
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

            # only contains the past_event_counts as past data?
            X_raw = pd.DataFrame(rows)

            if method == 'basic':
                est = fe_basic.fit_from_xy(X_raw, labels, feature_cols=None, model_cfg=mc_config or mc_basic())
            elif method == 'advanced':
                est = fe_advanced.fit_from_xy(X_raw, labels, feature_cols=None, model_cfg=mc_config or mc_advanced())
            else:
                raise ValueError('Only baisc and advanced exist!')

            models[place_name] = DecisionPointModel(place_name=place_name, estimator=est)

        return DecisionMiningResult(models=models, skipped=skipped)

    def extract_guards(self,
                       mining_result: DecisionMiningResult,
                       *,
                       use_advanced_estimator: bool = False,
                       ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        
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

    def print_summary_and_visualize(self, guards: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> None:
        """
        Print a summary of the discovered guards and visualize the Petri net with guard annotations.
        """

        def _guard_prob(g: Dict[str, Any]) -> float:
            # FunctionEstimator emits prob_model / prob_emp; older code used prob.
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

            # Prefer the simplified rule; but if it collapsed to (true) while raw_rule had conditions,
            # show raw_rule so you can see what was removed (often NaN-dummy splits).
            if rule and rule != "(true)":
                parts.append("rule=" + rule)
            else:
                if raw_rule and raw_rule != "(true)":
                    parts.append("raw_rule=" + raw_rule)

            return "; ".join(parts)
        
        def _best_guard_text(guard_list):
            if not guard_list:
                return ""
            # pick guard with highest probability (fallback to first)
            best = max(guard_list, key=_guard_prob)
            rule = best.get("rule", "")
            prob = _guard_prob(best)
            if rule and rule != "(true)":
                return f"{rule} | p={prob:.2f}"
            return f"p={prob:.2f}"

        # Optional: exclude likely-silent transitions from printing.
        # Adjust this regex to match your silent transition naming convention.
        silent_label_rx = re.compile(r"^(skip_|tau|silent|>>)")

        for place_name, by_label in guards.items():
            print(f"\\n=== {place_name} ===")
            if not by_label:
                print("  (no guards emitted for this place)")
                continue

            any_printed = False
            for label, guard_list in by_label.items():
                if silent_label_rx.search(str(label)):
                    continue
                any_printed = True
                
                # Use class label from transition object if available
                trans_obj = self.transition_by_key.get(str(label))
                display_label = trans_obj.label if trans_obj and hasattr(trans_obj, 'label') and trans_obj.label else str(label)
                
                print(f"- {display_label} ({len(guard_list)} guards)")
                for g in guard_list:
                    print("  *", _format_guard(g))

            if not any_printed:
                print("  (all labels filtered as silent)")

        # Visualize the Petri net with guard annotations

        # Build a transition -> guard text map
        transition_guard_text = {}
        for place_name, by_label in guards.items():
            for label, guard_list in by_label.items():
                transition_guard_text[str(label)] = _best_guard_text(guard_list)

        decorations = {}
        for t in self.net.transitions:
            name = str(t.name)
            guard_text = transition_guard_text.get(name, "")
            
            # Use class label for visualization display
            base_label = t.label if hasattr(t, 'label') and t.label else name
            
            label_text = f"{base_label}\\n{guard_text}" if guard_text else base_label
            decorations[t] = {"label": label_text}

        params = {"decorations": decorations}
        gviz = pn_vis.apply(self.net, self.im, self.fm, parameters=params)
        pn_vis.view(gviz)