
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import importlib.util

import networkx as nx
import pandas as pd

from pm4py.algo.decision_mining import algorithm as decision_mining_algorithm
from pm4py.visualization.decisiontree import visualizer as decisiontree_visualizer

@dataclass
class DecisionMiningResult:
    decision_table: Dict[str, List[Tuple[Dict[str, Any], Any]]]
    decision_points: Dict[str, List[str]]
    decision_trees: Dict[str, Dict[str, Any]]


class DecisionDiscovery:
    def __init__(self,
                 petri_net: Tuple,
                 event_log_df: pd.DataFrame,
                 case_ids: Optional[List[str]],
                 activity_key: str = "concept:name",
                 case_id_key: str = "case:concept:name"):
        
        self.net, self.im, self.fm = petri_net
        self.event_log_df = event_log_df
        self.case_ids = case_ids
        self.activity_key = activity_key
        self.case_id_key = case_id_key
    
    def _filter_event_log_df(self) -> pd.DataFrame:
        """
        filter the event log dataframe on only the cases in case ids
        """
        if self.case_ids is None:
            return self.event_log_df
        # ensure uniqueness of ids
        case_ids_set = set(self.case_ids)
        return self.event_log_df[self.event_log_df[self.case_id_key].isin(case_ids_set)]

    def _compute_transition_sccs(self) -> Dict[Any, int]:
        """
        SCCs = Strongly Connected Components.
        In a directed graph, an SCC is a set of nodes where each node can reach every other node via directed paths.
        
        Loops correspond to cycles, which appear as SCCs.
        At a decision point (place with multiple outgoing transitions), if some outgoing transitions stay within the same SCC while others leave it, that place is treated as a loop decision (continue vs exit).
        If all outgoing transitions remain in one SCC, we treat it as a normal XOR decision.
        """
        graph = nx.DiGraph()
        for transition in self.net.transitions:
            graph.add_node(transition)
        for place in self.net.places:
            incoming = [arc.source for arc in place.in_arcs]
            outgoing = [arc.target for arc in place.out_arcs]
            for source in incoming:
                for target in outgoing:
                    if hasattr(source, "label") and hasattr(target, "label"):
                        graph.add_edge(source, target)
        sccs = list(nx.strongly_connected_components(graph))
        scc_map: Dict[Any, int] = {}
        for idx, scc in enumerate(sccs):
            for transition in scc:
                scc_map[transition] = idx
        return scc_map

    def _split_decision_points(self, labels: bool) -> Tuple[List[str], List[str]]:
        decision_points = decision_mining_algorithm.get_decision_points(
            self.net, labels=labels, parameters={"labels": labels}
        )

        transition_scc = self._compute_transition_sccs()
        place_by_name = {place.name: place for place in self.net.places}

        xor_points: List[str] = []
        loop_points: List[str] = []

        for place_name in decision_points.keys():
            place = place_by_name.get(place_name)
            if place is None:
                continue
            outgoing_transitions = [arc.target for arc in place.out_arcs]
            scc_ids = [transition_scc.get(t) for t in outgoing_transitions]
            unique_sccs = {scc for scc in scc_ids if scc is not None}

            if len(unique_sccs) >= 2:
                loop_points.append(place_name)
            else:
                xor_points.append(place_name)

        return xor_points, loop_points
    
    def decision_mining_datasets(
        self,
        attributes: Optional[List[str]] = None,
        use_trace_attributes: bool = False,
        k: int = 1,
        labels: bool = True,
        pre_decision_points: Optional[List[str]] = None,
    ) -> DecisionMiningResult:
        """
        Generate decision tables and trees using pm4py's decision mining algorithm.
        """
        filtered_df = self._filter_event_log_df()
        parameters = {
            "labels": labels,
            "case_id_key": self.case_id_key,
            "activity_key": self.activity_key,
        }

        xor_points, loop_points = self._split_decision_points(labels=labels)

        if pre_decision_points is not None:
            xor_points = [p for p in xor_points if p in pre_decision_points]
            loop_points = [p for p in loop_points if p in pre_decision_points]

        xor_table, xor_points_dict = decision_mining_algorithm.get_decisions_table(
            filtered_df,
            self.net,
            self.im,
            self.fm,
            attributes=attributes,
            use_trace_attributes=use_trace_attributes,
            k=k,
            pre_decision_points=xor_points,
            parameters=parameters,
        )

        loop_table, loop_points_dict = decision_mining_algorithm.get_decisions_table(
            filtered_df,
            self.net,
            self.im,
            self.fm,
            attributes=attributes,
            use_trace_attributes=use_trace_attributes,
            k=k,
            pre_decision_points=loop_points,
            parameters=parameters,
        )

        decision_trees: Dict[str, Dict[str, Any]] = {"xor": {}, "loop": {}}
        for place_name in xor_points_dict.keys():
            clf, feature_names, classes = decision_mining_algorithm.get_decision_tree(
                filtered_df,
                self.net,
                self.im,
                self.fm,
                decision_point=place_name,
                attributes=attributes,
                parameters=parameters,
            )
            decision_trees["xor"][place_name] = {
                "classifier": clf,
                "feature_names": feature_names,
                "classes": classes,
            }

        for place_name in loop_points_dict.keys():
            clf, feature_names, classes = decision_mining_algorithm.get_decision_tree(
                filtered_df,
                self.net,
                self.im,
                self.fm,
                decision_point=place_name,
                attributes=attributes,
                parameters=parameters,
            )
            decision_trees["loop"][place_name] = {
                "classifier": clf,
                "feature_names": feature_names,
                "classes": classes,
            }

        decision_table = {"xor": xor_table, "loop": loop_table}
        decision_points = {"xor": xor_points_dict, "loop": loop_points_dict}

        return DecisionMiningResult(decision_table=decision_table,
                                    decision_points=decision_points,
                                    decision_trees=decision_trees)

    def visualize_decision_tree(
        self,
        place_name: str,
        attributes: Optional[List[str]] = None,
        use_trace_attributes: bool = False,
        k: int = 1,
        labels: bool = True,
        image_format: str = "png",
    ) -> Any:
        """
        Visualize the decision tree for a specific decision point.
        Returns the Graphviz object if visualization is available.
        """
        if importlib.util.find_spec("sklearn") is None or importlib.util.find_spec("graphviz") is None:
            raise ImportError("Visualization requires scikit-learn and graphviz installed")

        filtered_df = self._filter_event_log_df()
        parameters = {
            "labels": labels,
            "case_id_key": self.case_id_key,
            "activity_key": self.activity_key,
        }

        clf, feature_names, classes = decision_mining_algorithm.get_decision_tree(
            filtered_df,
            self.net,
            self.im,
            self.fm,
            decision_point=place_name,
            attributes=attributes,
            parameters=parameters,
        )

        gviz = decisiontree_visualizer.apply(
            clf,
            feature_names,
            classes,
            parameters={
                decisiontree_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: image_format
            },
        )
        return gviz