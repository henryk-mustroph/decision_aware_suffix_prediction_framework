"""
Compact, alignment-driven decision mining. 
Based on algorithm from:
- De Leoni, and Van Der Aalst. "Data-aware process mining: discovering decisions in processes using alignments." ACM symposium on applied computing. 2013.

- Adaptions:

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict, Counter

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
        if ts_col not in self.event_log_df.columns or case_col not in self.event_log_df.columns:
            self.event_log_df["case_elapsed_time"] = np.nan
            # backwards-compat alias (older typo)
            self.event_log_df["case_elappsed_time"] = self.event_log_df["case_elapsed_time"]
            return

        # Ensure timestamps are proper datetimes; compute on a sorted view but keep original row order.
        self.event_log_df[ts_col] = pd.to_datetime(self.event_log_df[ts_col], errors="coerce")
        work = self.event_log_df[[case_col, ts_col]].copy()
        work["_orig_index"] = self.event_log_df.index
        work = work.sort_values([case_col, ts_col], kind="mergesort")
        case_start_times = work.groupby(case_col)[ts_col].transform("min")
        work["_case_elapsed"] = (work[ts_col] - case_start_times).dt.total_seconds()
        work = work.set_index("_orig_index")
        series = work["_case_elapsed"].reindex(self.event_log_df.index)
        self.event_log_df["case_elapsed_time"] = series
        # backwards-compat alias (older typo)
        self.event_log_df["case_elappsed_time"] = series

    def __create_event_elapsed_time_column(self) -> None:
        """
        Create a new column representing elapsed time since the previous event.
        """
        case_col = "case:concept:name"
        ts_col = "time:timestamp"
        if ts_col not in self.event_log_df.columns or case_col not in self.event_log_df.columns:
            self.event_log_df["event_elapsed_time"] = np.nan
            return

        self.event_log_df[ts_col] = pd.to_datetime(self.event_log_df[ts_col], errors="coerce")
        work = self.event_log_df[[case_col, ts_col]].copy()
        work["_orig_index"] = self.event_log_df.index
        work = work.sort_values([case_col, ts_col], kind="mergesort")
        work["_event_elapsed"] = work.groupby(case_col)[ts_col].diff().dt.total_seconds()
        work = work.set_index("_orig_index")
        series = work["_event_elapsed"].reindex(self.event_log_df.index)
        # diff() yields NaN for the first event of each case; also NaN if timestamps are missing.
        self.event_log_df["event_elapsed_time"] = series.fillna(0.0)

    def _filter_attributes(self, attrs: Dict[str, Any], attributes: Optional[List[str]]) -> Dict[str, Any]:
        if not attrs:
            return {}
        if attributes:
            # Strictly respect the attribute whitelist.
            return {k: attrs.get(k, np.nan) for k in attributes}
        return attrs

    def collect_I(self, attributes: Optional[List[str]]) -> Tuple[Dict[Any, List[Tuple[Dict[str, Any], Any]]], Dict[Any, List[Any]]]:
        """
        collect training data per decision place
        """
        # big problem of algorithm: if (>>, None) is my log move, so a silent transition is fired: I do no know which transition it is: solved
        # Now I have for each alignment a list of tuples: a tuple: (('>>', 'skip_2'), ('>>', None)) first element has name of transition taken

        # list of alignments: [[(('>>', 'skip_2'), ('>>', None)), (('>>', 'init_loop_3'), ('>>', None)), ... ], ...]
        
        # store train data for each transition place
        I: Dict[Any, List[Tuple[Dict[str, Any], Any]]] = defaultdict(list)
        legend = defaultdict(list)
        
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
                                    event_attrs = self._filter_attributes(event.iloc[0].to_dict(), attributes)
                                    attrs = dict(event_attrs)
                                attrs["past_events"] = list(past_events)

                                # add log data
                                I[p].append((attrs, model_name))
                            else:
                                attrs = {}
                                attrs["past_events"] = list(past_events)
                                I[p].append((attrs, model_name))

                            if log_name != ">>" and not event.empty:
                                past_events.append(event_attrs)

                            legend[p].append(((log_name, model_name), (log_label, model_label)))
        return I, legend

    def _is_number(self, value: Any) -> bool:
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, float, np.number)):
            return not (isinstance(value, float) and np.isnan(value))
        return False

    def _numeric_values(self, events: List[Dict[str, Any]], key: str) -> List[float]:
        values: List[float] = []
        for ev in events:
            val = ev.get(key)
            if self._is_number(val):
                values.append(float(val))
        return values

    def _categorical_values(self, events: List[Dict[str, Any]], key: str) -> List[Any]:
        values: List[Any] = []
        for ev in events:
            val = ev.get(key)
            if val is None:
                continue
            if isinstance(val, float) and np.isnan(val):
                continue
            if not self._is_number(val):
                values.append(val)
        return values

    def _windowed_numeric_stats(self, values: List[float], windows: List[int], prefix: str) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}
        if not values:
            return feats
        arr = np.asarray(values, dtype=float)
        feats[f"{prefix}_count"] = int(arr.size)
        feats[f"{prefix}_mean"] = float(np.mean(arr))
        feats[f"{prefix}_std"] = float(np.std(arr))
        feats[f"{prefix}_min"] = float(np.min(arr))
        feats[f"{prefix}_max"] = float(np.max(arr))
        feats[f"{prefix}_sum"] = float(np.sum(arr))

        for k in windows:
            tail = arr[-k:]
            feats[f"{prefix}_w{k}_mean"] = float(np.mean(tail))
            feats[f"{prefix}_w{k}_std"] = float(np.std(tail))
            feats[f"{prefix}_w{k}_min"] = float(np.min(tail))
            feats[f"{prefix}_w{k}_max"] = float(np.max(tail))
            feats[f"{prefix}_w{k}_sum"] = float(np.sum(tail))
        return feats

    def _windowed_categorical_stats(self, values: List[Any], windows: List[int], prefix: str, max_unique_for_mode: int) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}
        if not values:
            return feats
        feats[f"{prefix}_count"] = int(len(values))
        feats[f"{prefix}_unique"] = int(len(set(values)))
        if feats[f"{prefix}_unique"] <= max_unique_for_mode:
            feats[f"{prefix}_mode"] = Counter(values).most_common(1)[0][0]
        for k in windows:
            tail = values[-k:]
            feats[f"{prefix}_w{k}_unique"] = int(len(set(tail)))
            if feats[f"{prefix}_w{k}_unique"] <= max_unique_for_mode:
                feats[f"{prefix}_w{k}_mode"] = Counter(tail).most_common(1)[0][0]
        return feats

    def _exp_decay_numeric_stats(self, values: List[float], prefix: str, alpha: float = 0.8) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}
        if not values:
            return feats
        arr = np.asarray(values, dtype=float)
        n = int(arr.size)
        feats[f"{prefix}_count"] = n
        feats[f"{prefix}_mean"] = float(np.mean(arr))
        feats[f"{prefix}_std"] = float(np.std(arr))
        feats[f"{prefix}_min"] = float(np.min(arr))
        feats[f"{prefix}_max"] = float(np.max(arr))
        feats[f"{prefix}_sum"] = float(np.sum(arr))
        feats[f"{prefix}_first"] = float(arr[0])
        feats[f"{prefix}_last"] = float(arr[-1])
        if n > 1:
            x = np.arange(n, dtype=float)
            x_centered = x - float(np.mean(x))
            denom = float(np.sum(x_centered ** 2))
            if denom > 0.0:
                slope = float(np.sum(x_centered * (arr - float(np.mean(arr)))) / denom)
                feats[f"{prefix}_trend_slope"] = slope

        # Exponential-decay recency: last event has weight 1, previous alpha, etc.
        a = float(alpha)
        if 0.0 < a < 1.0 and n > 0:
            weights = np.power(a, np.arange(n - 1, -1, -1, dtype=float))
            w_sum = float(np.sum(weights))
            if w_sum > 0.0:
                w_mean = float(np.sum(weights * arr) / w_sum)
                w_var = float(np.sum(weights * ((arr - w_mean) ** 2)) / w_sum)
                feats[f"{prefix}_ewm_mean"] = w_mean
                feats[f"{prefix}_ewm_std"] = float(np.sqrt(max(w_var, 0.0)))
        return feats

    def _exp_decay_categorical_stats(self, values: List[Any], prefix: str, alpha: float = 0.8, max_unique_for_mode: int = 30) -> Dict[str, Any]:
        feats: Dict[str, Any] = {}
        if not values:
            return feats
        n = int(len(values))
        feats[f"{prefix}_count"] = n
        uniq = list(set(values))
        feats[f"{prefix}_unique"] = int(len(uniq))
        feats[f"{prefix}_last"] = values[-1]

        if feats[f"{prefix}_unique"] <= max_unique_for_mode:
            feats[f"{prefix}_mode"] = Counter(values).most_common(1)[0][0]
            a = float(alpha)
            if 0.0 < a < 1.0 and n > 0:
                weights = np.power(a, np.arange(n - 1, -1, -1, dtype=float))
                scores: Dict[Any, float] = defaultdict(float)
                for v, w in zip(values, weights):
                    scores[v] += float(w)
                if scores:
                    feats[f"{prefix}_ewm_mode"] = max(scores.items(), key=lambda kv: kv[1])[0]
        return feats

    def _delay_features(self, past_events: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Use event_elapsed_time as the time feature for delay-related stats.
        delay_values: List[float] = self._numeric_values(past_events, "event_elapsed_time")
        if not delay_values:
            return {}
        return self._exp_decay_numeric_stats(delay_values, "event_elapsed_time", alpha=0.8)

    def _build_feature_row(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        attrs = dict(attrs) if attrs else {}
        past_events = attrs.pop("past_events", [])
        features: Dict[str, Any] = dict(attrs)
        features["past_events_count"] = int(len(past_events))

        if not past_events:
            return features

        keys: set[str] = set()
        for ev in past_events:
            keys.update(ev.keys())

        for key in sorted(keys):
            if key in ("time:timestamp", "timestamp", "concept:name", "case:concept:name"):
                continue
            num_vals = self._numeric_values(past_events, key)
            if num_vals:
                features.update(self._exp_decay_numeric_stats(num_vals, f"{key}_prefix", alpha=0.8))
            cat_vals = self._categorical_values(past_events, key)
            if cat_vals:
                features.update(self._exp_decay_categorical_stats(cat_vals, f"{key}_prefix", alpha=0.8, max_unique_for_mode=30))

        features.update(self._delay_features(past_events))
        return features

    def mine_decision_models(self, attributes: Optional[List[str]] = None, model_cfg: Optional[ModelConfig] = None) -> DecisionMiningResult:
        I, _legend = self.collect_I(attributes)

        models: Dict[str, DecisionPointModel] = {}
        skipped: List[str] = []

        for place, samples in I.items():
            place_name = str(getattr(place, "name", place))
            if not samples:
                skipped.append(place_name)
                continue

            rows: List[Dict[str, Any]] = []
            labels: List[str] = []

            for ass, chosen in samples:
                rows.append(self._build_feature_row(ass))
                labels.append(str(chosen))

            X_raw = pd.DataFrame(rows)
            
            est = FunctionEstimator.fit_from_xy(X_raw,  labels,feature_cols=None, model_cfg=model_cfg or ModelConfig())
            
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
            
            guards_by_place[place_name] = model.estimator.extract_probabilistic_guards_simple(min_leaf_prob=min_leaf_prob,
                                                                                              min_leaf_lift=min_leaf_lift,
                                                                                              min_leaf_support=min_leaf_support,
                                                                                              surrogate_max_depth=surrogate_max_depth,
                                                                                              surrogate_min_leaf=surrogate_min_leaf,
                                                                                              always_keep_best=always_keep_best)
        return guards_by_place

