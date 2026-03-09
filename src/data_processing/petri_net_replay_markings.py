from copy import deepcopy
from typing import Optional
import pandas as pd
import pm4py
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from pm4py.objects.petri_net import semantics
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.objects.log.obj import Trace, Event
from pm4py.objects.petri_net.obj import Marking

class InductiveMiner:
    def __init__(self, path_to_csv_log, case_id_col="case:concept:name", activity_col="concept:name", timestamp_col="time:timestamp", resource_col="org:resource"):
        self.path_to_csv_log = path_to_csv_log
        self.case_id_col = case_id_col
        self.activity_col = activity_col
        self.timestamp_col = timestamp_col
        self.resource_col = resource_col

    def _create_event_log(self, case_ids: Optional[list] = None):
        df = pd.read_csv(self.path_to_csv_log)
        
        # Rename: for example important for helpdesk:
        rename = {self.case_id_col: "case:concept:name",
                  self.activity_col: "concept:name",
                  self.timestamp_col: "time:timestamp",
                  self.resource_col: "org:resource"}
            
        df = df.rename(columns=rename)
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], errors="coerce")

        if case_ids is not None:
            df = df[df["case:concept:name"].isin(case_ids)]
        
        params = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: "case:concept:name"}
        ev_log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG, parameters=params)
        return df, ev_log

    def discover_petri_net(self, visulaize: bool=True, store_loc_file_path:Optional[str]=None, case_ids: Optional[list] = None):
        _, event_log = self._create_event_log(case_ids=case_ids)
        net, initial_marking, final_marking = pm4py.discover_petri_net_inductive(event_log,
                                                                                 multi_processing=False)
    
        if visulaize:
            gviz = pn_visualizer.apply(net, initial_marking, final_marking)
            pn_visualizer.view(gviz)
            if store_loc_file_path:
                pn_visualizer.save(gviz, store_loc_file_path)
            
        return net, initial_marking, final_marking
    
class PNReplayMarkings:
    def __init__(self, discovered_pn, prefix_df, event_label):
        self.net, self.im, self.fm = discovered_pn
        self.prefix_df = prefix_df
        self.event_label = event_label
    
    # does not work currently
    def get_markings(self):
        prefix_event_labels = self.prefix_df[self.event_label].tolist()
        markings = []

        label_to_transitions = {}
        for transition in self.net.transitions:
            if transition.label is None:
                continue
            label_to_transitions.setdefault(transition.label, []).append(transition)

        for prefix in prefix_event_labels:
            current_marking = deepcopy(self.im)
            for label in prefix:
                transitions = label_to_transitions[label]
                for transition in transitions:
                    if semantics.is_enabled(transition, self.net, current_marking):
                        current_marking = semantics.execute(transition, self.net, current_marking)
                        break
            markings.append(deepcopy(current_marking))

        return markings

    # uses token replay - works
    def get_markings_token_replay(self):
        prefix_event_labels = self.prefix_df[self.event_label].tolist()
        markings = []

        for prefix in prefix_event_labels:
            trace_pref = Trace([Event({"concept:name": act}) for act in prefix])
            replayed_traces = token_replay.apply(
                log=[trace_pref],
                net=self.net,
                initial_marking=self.im,
                final_marking=Marking(),
            )
            result = replayed_traces[0]
            reached_marking = result["reached_marking"]
            markings.append(reached_marking)

        return markings
        