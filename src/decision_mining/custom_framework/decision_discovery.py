"""
Compact, alignment-driven decision mining. 
Based on algorithm from:
- De Leoni, and Van Der Aalst. "Data-aware process mining: discovering decisions in processes using alignments." ACM symposium on applied computing. 2013.

- Adaptions:

"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from decision_mining.custom_framework.function_estimator import FunctionEstimator, ModelConfig

@dataclass
class DecisionPointModel:
    place_name: str
    estimator: FunctionEstimator


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
        # create event elapsed time
        self.__create_event_elapsed_time_column()
        
        self.alignments = alignments

        # map transition name -> transition object
        self.transition_by_key: Dict[str, Any] = {str(t.name): t for t in self.net.transitions}

        # decision places (>1 outgoing)
        self.decision_places: List[Any] = [p for p in self.net.places if len(p.out_arcs) > 1]
        self.routing_transitions_by_place: Dict[Any, List[Any]] = {p: [arc.target for arc in p.out_arcs] for p in self.decision_places}

        self.in_place_for_routing_transition: Dict[Any, List[Any]] = defaultdict(list)
        for place, transitions in self.routing_transitions_by_place.items():
            for transition in transitions:
                self.in_place_for_routing_transition[transition].append(place)
                
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

        # Convention: ints are IDs/categorical; floats are continuous.
        out: Dict[str, Any] = {}
        for k, v in filtered.items():
            if isinstance(v, (int, np.integer)):
                out[k] = str(int(v))
            else:
                out[k] = v
        return out

    def collect_I(self,
                  attributes: Optional[List[str]] = None,
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
        
        # Determine which attributes are used for:
        # - current event features (dynamic + static)
        # - past_events (dynamic only)
        if dynamic_attributes is None and static_attributes is None:
            current_attr_keys = attributes
            past_attr_keys = attributes
        else:
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
                if isinstance(v, (int, np.integer)):
                    return str(int(v))
                return str(v)

            def _as_num(v: Any) -> Any:
                if isinstance(v, (float, np.floating)):
                    return v
                if isinstance(v, (int, np.integer)):
                    return float(v)
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
                             attributes: Optional[List[str]] = None,
                             dynamic_attributes: Optional[List[str]] = None,
                             static_attributes: Optional[List[str]] = None,
                             model_cfg: Optional[ModelConfig] = None) -> DecisionMiningResult:
        # train data and legend
        I, _legend = self.collect_I(attributes=attributes,
                                    dynamic_attributes=dynamic_attributes,
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

            X_raw = pd.DataFrame(rows)
            
            est = FunctionEstimator.fit_from_xy(X_raw,  labels, feature_cols=None, model_cfg=model_cfg or ModelConfig())
            
            models[place_name] = DecisionPointModel(place_name=place_name, estimator=est)

        return DecisionMiningResult(models=models, skipped=skipped)

    def extract_probabilistic_guards(self,
                                     mining_result: DecisionMiningResult,
                                     min_leaf_prob: float = 0.2,
                                     min_leaf_lift: float = 2.0,
                                     min_leaf_support: int = 20,
                                     surrogate_max_depth: int = 4,
                                     surrogate_min_leaf: int = 20,
                                     always_keep_best: bool = True) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        
        guards_by_place: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        for place_name, model in mining_result.models.items():
            #
            guards_by_place[place_name] = model.estimator.extract_probabilistic_guards_simple(min_leaf_prob=min_leaf_prob,
                                                                                              min_leaf_lift=min_leaf_lift,
                                                                                              min_leaf_support=min_leaf_support,
                                                                                              surrogate_max_depth=surrogate_max_depth,
                                                                                              surrogate_min_leaf=surrogate_min_leaf,
                                                                                              always_keep_best=always_keep_best)
        return guards_by_place