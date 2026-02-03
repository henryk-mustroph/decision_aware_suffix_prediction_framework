
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import pandas as pd

from pm4py.algo.decision_mining import algorithm as decision_mining_algorithm
from pm4py.visualization.decisiontree import visualizer as decisiontree_visualizer


@dataclass
class DecisionMiningResult:
    # decision_table: Dict[str, List[Tuple[Dict[str, Any], Any]]]
    decision_points: Dict[str, List[str]]
    decision_trees: Dict[str, Dict[str, Any]]


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
    
    def _filter_event_log_df(self):
        """
        filter the event log dataframe on only the cases in case ids (train and val)
        
        Renames case/activity/time columns to:
        - case:concept:name
        - concept:name
        - time:timestamp
        """
        if self.case_ids is None:
            return self.event_log_df
        # ensure uniqueness of ids
        case_ids_set = set(self.case_ids)
        event_log_df = self.event_log_df[self.event_log_df[self.case_id_key].isin(case_ids_set)]
    
        rename_map = {}
        if self.case_id_key in event_log_df.columns and self.case_id_key != "case:concept:name":
            rename_map[self.case_id_key] = "case:concept:name"
        
        if self.activity_key in event_log_df.columns and self.activity_key != "concept:name":
            rename_map[self.activity_key] = "concept:name"
        
        if self.time_key in event_log_df.columns and self.time_key != "time:timestamp":
            rename_map[self.time_key] = "time:timestamp"
        
        if rename_map:
            event_log_df = event_log_df.rename(columns=rename_map)
        
        return event_log_df        

    def _compute_transition_sccs(self):
        """
        SCCs = Strongly Connected Components. In a directed graph, an SCC is a set of nodes where each node can reach every other node via directed paths.
        
        Loops correspond to cycles, which appear as SCCs.
        At a decision point (place with multiple outgoing transitions), if some outgoing transitions stay within the same SCC while others leave it, that place is treated as a loop decision (continue vs exit).
        If all outgoing transitions remain in one SCC, we treat it as a normal XOR decision.
        
        - Nodes: places + transitions
        - Edges: follow arcs exactly (place → transition, transition → place)
        """
        graph = nx.DiGraph()
        
        # add nodes
        for p in self.net.places:
            graph.add_node(p)
        for t in self.net.transitions:
            graph.add_node(t)
            
        # add edges (arc direction)
        for p in self.net.places:
            # place -> transition
            for a in p.out_arcs:          
                graph.add_edge(p, a.target)
            # transition -> place
            for a in p.in_arcs:           
                graph.add_edge(a.source, p)
        
        sccs = list(nx.strongly_connected_components(graph))
        
        scc_map: Dict[Any, int] = {}
        for idx, comp in enumerate(sccs):
            for node in comp:
                scc_map[node] = idx
                
        return scc_map

    def _split_decision_points(self, labels: bool):
        xor_places = []
        loop_decision_places = []
        
        # Get XOR decision points from (De Leonie: Decision mining): a place in the Petri net where multiple outgoing transitions exist
        decision_points = decision_mining_algorithm.get_decision_points(self.net, labels=labels)
        decision_places = list(decision_points.keys())
        place_by_name = {pl.name: pl for pl in self.net.places}
        
        scc_map = self._compute_transition_sccs()
    
        for p in decision_places:
            pl_obj = place_by_name[p]
            p_scc = scc_map[pl_obj]
            out_ts = [a.target for a in pl_obj.out_arcs]  # transitions

            inside = [t for t in out_ts if scc_map[t] == p_scc]
            outside = [t for t in out_ts if scc_map[t] != p_scc]

            if inside and outside:
                loop_decision_places.append((pl_obj, inside, outside))
            else:
                xor_places.append((pl_obj, out_ts))

        return xor_places, loop_decision_places
        
    
    
    
    def decision_mining_datasets(self,
                                 attributes: Optional[List[str]] = None,
                                 use_trace_attributes: bool = True,
                                 trace_attributes: Optional[List[str]] = None) -> DecisionMiningResult:
        """
        Generate decision tables and trees using pm4py's decision mining algorithm.
        """
        # event log without cases of test set
        filtered_df = self._filter_event_log_df()
        # print(filtered_df.head())

        if trace_attributes is None:
            trace_attributes = []

        xor_places, loop_places = self._split_decision_points(labels=True)
        
        xor_names = [p.name for p, _ in xor_places]
        loop_names = [p.name for p, _, _ in loop_places]

        decision_points_all = decision_mining_algorithm.get_decision_points(self.net, labels=True)
        xor_points_dict = {k: v for k, v in decision_points_all.items() if k in xor_names}
        loop_points_dict = {k: v for k, v in decision_points_all.items() if k in loop_names}
        
        # print("XOR decision places:")
        # for p, outs in xor_places:
        #     print(p.name, [(t.name, t.label) for t in outs])

        # print("\nLoop decision places (continue vs exit):")
        # for p, inside, outside in loop_places:
        #     print(p.name, "INSIDE:", [(t.name, t.label) for t in inside], "OUTSIDE:", [(t.name, t.label)  for t in outside])

        decision_trees: Dict[str, Dict[str, Any]] = {"xor": {}, "loop": {}}
        
        for place_name in xor_points_dict.keys():
            clf, feature_names, classes = decision_mining_algorithm.get_decision_tree(filtered_df,
                                                                                      self.net,
                                                                                      self.im,
                                                                                      self.fm,
                                                                                      decision_point=place_name,
                                                                                      attributes=attributes)
            decision_trees["xor"][place_name] = {"classifier": clf,
                                                 "feature_names": feature_names,
                                                 "classes": classes}

        for place_name in loop_points_dict.keys():
            clf, feature_names, classes = decision_mining_algorithm.get_decision_tree(filtered_df,
                                                                                      self.net,
                                                                                      self.im,
                                                                                      self.fm,
                                                                                      decision_point=place_name,
                                                                                      attributes=attributes)
            decision_trees["loop"][place_name] = {"classifier": clf,
                                                  "feature_names": feature_names,
                                                  "classes": classes}

        
        decision_points = {"xor": xor_points_dict, "loop": loop_points_dict}

        return DecisionMiningResult(decision_points=decision_points,
                                    decision_trees=decision_trees)

    def visualize_decision_tree(self,
                                decision_tree: Dict = {},
                                image_format: str = "png") -> Any:
        """
        Visualize the decision tree for a specific decision point.
        Returns the Graphviz object if visualization is available.
        """
        
        clf = decision_tree.get("classifier")
        feature_names = decision_tree.get("feature_names")
        classes = decision_tree.get("classes")

        gviz = decisiontree_visualizer.apply(clf,
                                             feature_names,
                                             classes,
                                             parameters={decisiontree_visualizer.Variants.CLASSIC.value.Parameters.FORMAT: image_format})
        decisiontree_visualizer.view(gviz)