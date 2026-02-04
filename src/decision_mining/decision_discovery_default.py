"""
Adopted from:
- De Leoni, M., & Van Der Aalst, W. M. (2013, March). Data-aware process mining: discovering decisions in processes using alignments. In Proceedings of the 28th annual ACM symposium on applied computing (pp. 1454-1461).

Adapted and extended for the use in suffix prediction
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from pm4py.algo.decision_mining import algorithm as decision_mining_algorithm
from pm4py.visualization.decisiontree import visualizer as decisiontree_visualizer

@dataclass
class DecisionPointModel:
    place_name: str
    outgoing_transitions: List[str]                 # labels of outgoing transitions
    classifier: Any                                 # sklearn DecisionTreeClassifier
    feature_names: List[str]
    classes: List[str]                              # same order as clf.classes_
    rules_by_class: Dict[str, List[str]]            # readable rules (DNF pieces) per class
    guard_by_class: Dict[str, str]                  # DNF guard string per class


@dataclass
class DecisionMiningResult:
    decision_points: Dict[str, List[str]]           # place -> outgoing transition labels
    decision_models: Dict[str, DecisionPointModel]  # place -> model

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
        
        self.activity_key = activity_key
        self.case_id_key = case_id_key
        self.time_key = time_key
    
    def _filter_event_log_df(self) -> pd.DataFrame:
        """
        Filter the event log dataframe to only the cases in case_ids,
        and normalize column names to pm4py defaults:
          - case:concept:name
          - concept:name
          - time:timestamp
        Also sorts by (case, time) for stable behavior.
        """
        df = self.event_log_df.copy()

        if self.case_ids is not None:
            case_ids_set = set(self.case_ids)
            df = df[df[self.case_id_key].isin(case_ids_set)]

        rename_map = {}
        if self.case_id_key in df.columns and self.case_id_key != "case:concept:name":
            rename_map[self.case_id_key] = "case:concept:name"
        if self.activity_key in df.columns and self.activity_key != "concept:name":
            rename_map[self.activity_key] = "concept:name"
        if self.time_key in df.columns and self.time_key != "time:timestamp":
            rename_map[self.time_key] = "time:timestamp"

        if rename_map:
            df = df.rename(columns=rename_map)

        # Ensure timestamp is datetime
        if "time:timestamp" in df.columns:
            df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        # Sort for determinism (important for trace attribute replication + alignments)
        sort_cols = ["case:concept:name"]
        if "time:timestamp" in df.columns:
            sort_cols.append("time:timestamp")
        df = df.sort_values(sort_cols).reset_index(drop=True)

        return df
    
    # new
    def _augment_with_trace_attributes(self,
                                       df: pd.DataFrame,
                                       trace_attributes: Optional[List[str]] = None,
                                       agg: str = "first") -> pd.DataFrame:
        """
        pm4py decision mining expects features as event-level columns.
        If you want to use trace-level attributes (case attributes), replicate them
        to every event row of that case.

        agg: how to reduce if a trace attribute appears on multiple events with different values.
            - "first" (default) is typical
            - "last", "mode" (implemented), or "unique" (raises if >1 unique)
        """
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
                return s2.mode().iloc[0] if not s2.mode().empty else s2.iloc[0]
            if agg == "unique":
                uniq = s2.unique()
                if len(uniq) > 1:
                    raise ValueError(f"Trace attribute has >1 unique value in a case: {uniq}")
                return uniq[0]
            raise ValueError(f"Unknown agg='{agg}'")

        case_col = "case:concept:name"
        trace_df = df.groupby(case_col, as_index=False)[trace_attributes].agg(reducer)

        # Merge back (replicate to all rows in case)
        out = df.drop(columns=trace_attributes).merge(trace_df, on=case_col, how="left")
        return out
    
    # new
    @staticmethod
    def _uniq_preserve_order(xs: Optional[List[str]]) -> Optional[List[str]]:
        if xs is None:
            return None
        seen = set()
        out = []
        for x in xs:
            if x is None:
                continue
            if x not in seen:
                out.append(x)
                seen.add(x)
        return out      
        
    def decision_mining_datasets(self,
                                 attributes: Optional[List[str]] = None,
                                 trace_attributes: Optional[List[str]] = None,
                                 trace_attr_agg: str = "first",
                                 # Optional knobs for later use:
                                 extract_rules: bool = True) -> DecisionMiningResult:
        """
        Generates decision trees per decision point using pm4py, and (optionally)
        extracts rules/guards from those trees.

        attributes: event-level attributes to use as features
        trace_attributes: case-level attributes (replicated per event)
        """
        # only train and val set cases
        df = self._filter_event_log_df()
        # necessary? 
        df = self._augment_with_trace_attributes(df, trace_attributes=trace_attributes, agg=trace_attr_agg)

        print(df.head(5))

        # Merge attribute lists, unique, keep order
        feat_cols = []
        if attributes:
            feat_cols.extend(attributes)
        if trace_attributes:
            feat_cols.extend(trace_attributes)
        feat_cols = self._uniq_preserve_order(feat_cols)  # can be None
        
        print(feat_cols)

        # decision points: place -> outgoing transitions (labels)
        decisions = decision_mining_algorithm.get_decision_points(self.net, labels=True)

        print(decisions)

        decision_models: Dict[str, DecisionPointModel] = {}
        decision_points: Dict[str, List[str]] = {}

        for place_name, outgoing in decisions.items():
            # skip decision points that only lead to silent (None) transitions
            if all(x is None for x in outgoing):
                continue
            print(outgoing)

            clf, feature_names, classes = decision_mining_algorithm.get_decision_tree(df,
                                                                                      self.net,
                                                                                      self.im,
                                                                                      self.fm,
                                                                                      decision_point=place_name,
                                                                                      attributes=feat_cols)
            
            # human labels as strings (pm4py)
            human_labels = [str(c) for c in list(classes)]

            # sklearn encoded values (often ints)
            encoded_values = [str(c) for c in list(clf.classes_)]

            # map encoded -> human (assumes same ordering; in practice this is what pm4py does)
            if len(encoded_values) == len(human_labels):
                class_value_to_label = dict(zip(encoded_values, human_labels))
            else:
                # fallback: no mapping available
                class_value_to_label = {v: v for v in encoded_values}

            # Normalize to lists of strings
            outgoing_transitions = [str(x) for x in outgoing] if outgoing is not None else []
            # classes = [str(c) for c in list(classes)]

            # rules_by_class: Dict[str, List[str]] = {}
            # guard_by_class: Dict[str, str] = {}

            rules_by_class, guard_by_class = self._extract_tree_guards(
                                                    clf=clf,
                                                    feature_names=list(feature_names),
                                                    human_labels=human_labels,
                                                    class_value_to_label=class_value_to_label,
                                                )


            decision_points[place_name] = outgoing_transitions
            decision_models[place_name] = DecisionPointModel(place_name=place_name,
                                                             outgoing_transitions=outgoing_transitions,
                                                             classifier=clf,
                                                             feature_names=list(feature_names),
                                                             classes=list(classes),
                                                             rules_by_class=rules_by_class,
                                                             guard_by_class=guard_by_class)

        return DecisionMiningResult(decision_points=decision_points, decision_models=decision_models)
    
    @staticmethod
    def _extract_tree_guards(clf: Any,
                             feature_names: List[str],
                             human_labels: List[str],
                             class_value_to_label: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
        """
        Extract root->leaf rules and return them keyed by *human* class labels.
        """
        tree = clf.tree_
        FEATURE_UNDEF = -2

        def node_rule(feat_idx: int, thresh: float, direction: str) -> str:
            fname = feature_names[feat_idx]
            if direction == "left":
                return f"({fname} <= {thresh:.6g})"
            else:
                return f"({fname} > {thresh:.6g})"

        rules_by_label: Dict[str, List[str]] = {lab: [] for lab in human_labels}

        stack: List[Tuple[int, List[str]]] = [(0, [])]
        while stack:
            node_id, conds = stack.pop()
            feat_idx = int(tree.feature[node_id])

            if feat_idx == FEATURE_UNDEF:
                # leaf
                counts = tree.value[node_id][0]
                pred_idx = int(np.argmax(counts))

                encoded = str(clf.classes_[pred_idx])            # e.g., "0"
                label = class_value_to_label.get(encoded, encoded)  # e.g., "None"

                rule = " AND ".join(conds) if conds else "(true)"
                rules_by_label.setdefault(label, []).append(rule)
                continue

            thresh = float(tree.threshold[node_id])
            left_id = int(tree.children_left[node_id])
            right_id = int(tree.children_right[node_id])

            stack.append((right_id, conds + [node_rule(feat_idx, thresh, "right")]))
            stack.append((left_id,  conds + [node_rule(feat_idx, thresh, "left")]))

        guard_by_label: Dict[str, str] = {}
        for lab in human_labels:
            disj = rules_by_label.get(lab, [])
            if not disj:
                guard_by_label[lab] = "(false)"
            elif len(disj) == 1:
                guard_by_label[lab] = disj[0]
            else:
                guard_by_label[lab] = " OR ".join([f"({r})" for r in disj])

        return rules_by_label, guard_by_label

    def visualize_decision_tree(self,
                                decision_model: DecisionPointModel,
                                image_format: str = "png",
                                outfile: Optional[str] = None,
                                view: bool = True) -> Any:
        """
        Visualize a decision tree. If outfile is provided, tries to render there.
        """
        clf = decision_model.classifier
        feature_names = decision_model.feature_names
        classes = decision_model.classes

        gviz = decisiontree_visualizer.apply(clf,
                                             feature_names,
                                             classes,
                                             parameters={decisiontree_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: image_format})

        if outfile is not None:
            # gviz is a graphviz object; render if supported
            try:
                gviz.render(outfile, format=image_format, cleanup=True)
            except Exception:
                # fallback: pm4py's viewer-only
                pass

        if view:
            decisiontree_visualizer.view(gviz)

        return gviz