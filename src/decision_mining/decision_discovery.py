"""
Alignment-driven decision mining for suffix prediction.

Adapted from:
  De Leoni & van der Aalst, "Data-aware process mining: discovering decisions
  in processes using alignments", ACM SAC 2013.

Per decision place, the miner fits two models on the same labels:
  - a CatBoost base model that predicts the next visible event label, used at
    inference to reweight the LSTM softmax.
  - a sklearn decision-tree surrogate that yields human-readable guard rules.

Feature layout (per training row):
  - static attributes:  one column per attribute (case-level constant value)
  - dynamic attributes: two columns per attribute
      <attr>           value at the event just before the decision point
      <attr>_past_avg  mean of even older events  (numeric attributes)
      <attr>_past_mode most-frequent value across even older events (categorical)
"""
from __future__ import annotations

import json
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from decision_mining.function_estimator_catboost_advanced import (
    FunctionEstimator,
    ModelConfig,
)


def replay_alignment_decisions(case_alignment: List[Any],
                               *,
                               im: Any,
                               transition_by_name: Dict[str, Any],
                               decision_places: set) -> List[Dict[str, Any]]:
    """
    Replay one optimal alignment as a token-flow firing sequence and resolve,
    for every decision-place firing, the *next visible activity* it leads to.

    The Petri nets produced by the Inductive Miner route most choices through
    silent transitions (``skip_*``, ``init_loop_*``, ``tauSplit/Join``). The
    paper's rule ``a* = x_i`` (label of the transition consuming from p) only
    yields a usable next-activity target when that transition is visible; for a
    silent routing transition it would collapse to EOS. We therefore follow the
    *token* produced by the firing through subsequent silent transitions until
    a synchronous (visible) move consumes a descendant of it, and use that
    visible label as the target. Token *lineage* is tracked so that the two
    concurrent branches created by a ``tauSplit`` do not contaminate each other:
    a decision on one branch is only resolved by a visible event that actually
    consumes a token descended from that decision.

    For a synchronous decision firing the descendant is consumed at the same
    step, so the target is simply that visible label (the paper's ``a* = x_i``).
    A decision whose branch reaches the final marking with no further visible
    event resolves to ``"EOS"`` (the paper's ``a* = EOS`` termination case).

    Returns one record per decision-place firing::

        {"place":       <pm4py place object>,
         "sync_index":  <# synchronous events fired strictly before this firing>,
         "target":      <next visible activity label, or "EOS">,
         "resolved_by": <index of the visible event that resolved it, or None
                         when the branch ends the case (target == "EOS")>,
         "order":       <creation order within the trace>}

    ``sync_index`` indexes the data state: the instance is conditioned on the
    first ``sync_index`` synchronous events (the most recent one and the average
    over the earlier ones, per the data-state definition).
    """
    # marking: place -> list of tokens; each token is the set of still-pending
    # decision ids whose target is "the next visible activity that consumes a
    # descendant of this token".
    marking: Dict[Any, List[set]] = defaultdict(list)
    for place, count in im.items():
        for _ in range(int(count)):
            marking[place].append(set())

    pending: Dict[int, List[Any]] = {}   # id -> [place_obj, sync_index, target, resolved_by]
    next_id = 0
    sync_count = 0

    for (log_name, model_name), (log_label, model_label) in case_alignment:
        # Log-only move: no transition fires, nothing flows in the net.
        if model_name == ">>":
            continue
        trans = transition_by_name.get(model_name)
        if trans is None:
            continue

        # Consume one token from each input place; the firing inherits the
        # union of their pending decision ids.
        consumed: set = set()
        consumed_from: set = set()
        for arc in trans.in_arcs:
            src = arc.source
            if marking[src]:
                consumed |= marking[src].pop()
                consumed_from.add(src)

        # A firing that consumes a token from a decision place *is* a decision.
        for arc in trans.in_arcs:
            if arc.source in decision_places and arc.source in consumed_from:
                pending[next_id] = [arc.source, sync_count, None, None]
                consumed.add(next_id)
                next_id += 1

        is_sync = log_name != ">>"
        if is_sync:
            # Visible event: it resolves every pending decision carried by the
            # token it consumes (incl. a synchronous decision firing itself).
            label = log_label or model_label or "EOS"
            for did in list(consumed):
                if did in pending and pending[did][2] is None:
                    pending[did][2] = label
                    pending[did][3] = sync_count   # index of this visible event
            consumed = set()          # resolved; produced tokens start fresh
            sync_count += 1

        # Propagate the (still unresolved) pending ids onto produced tokens.
        for arc in trans.out_arcs:
            marking[arc.target].append(set(consumed))

    records: List[Dict[str, Any]] = []
    for did in sorted(pending):
        place_obj, sync_index, target, resolved_by = pending[did]
        records.append({"place": place_obj,
                        "sync_index": int(sync_index),
                        "target": target if target is not None else "EOS",
                        "resolved_by": None if resolved_by is None else int(resolved_by),
                        "order": int(did)})
    return records


@dataclass
class DecisionPointModel:
    place_name: str
    estimator: Any


@dataclass
class DecisionMiningResult:
    models: Dict[str, DecisionPointModel]
    skipped: List[str]


def build_feature_row(attrs: Dict[str, Any],
                      *,
                      dynamic_attributes: List[str],
                      static_attributes: List[str]) -> Dict[str, Any]:
    """
    Build a single feature row from a sample's collected attributes.

    Convention:
      - ``attrs["past_events"]`` is a list of dicts of dynamic attribute
        values, one per past event (chronological).
      - All other keys in ``attrs`` are static (case-level constants).
    """
    attrs = dict(attrs) if attrs else {}
    past_events = attrs.pop("past_events", [])

    static_set = set(static_attributes)
    dynamic_set = set(dynamic_attributes)

    features: Dict[str, Any] = {}

    # Static attributes pass through as single columns.
    for key, value in attrs.items():
        if isinstance(value, (int, np.integer)):
            value = str(int(value))
        features[key] = value

    if not past_events:
        return features

    if dynamic_set:
        allowed_dyn = dynamic_set
    else:
        observed: set = set()
        for ev in past_events:
            observed.update(ev.keys())
        allowed_dyn = observed - static_set

    previous_event = past_events[-1]
    older_events = past_events[:-1]

    for key in sorted(allowed_dyn):
        # Bare attribute name: value at the event immediately before the
        # decision point. (Dynamic and static attribute names are disjoint, so
        # this does not collide with the static columns above.)
        last_val = previous_event.get(key, np.nan)
        if isinstance(last_val, (int, np.integer)):
            last_val = str(int(last_val))
        features[key] = last_val

        # One summary across even older events; choose by observed value type.
        # When older_events is empty (decision at prefix length 1), fall back
        # to the bare value so the column is always present and train /
        # inference rows have the same schema.
        seq = [ev.get(key, np.nan) for ev in older_events]
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
            elif is_numeric_last:
                features[f"{key}_past_avg"] = float(last_val)
        else:
            if valid:
                counts = Counter(str(v) for v in valid)
                features[f"{key}_past_mode"] = counts.most_common(1)[0][0]
            elif last_val is not None and not (isinstance(last_val, float) and np.isnan(last_val)):
                features[f"{key}_past_mode"] = str(last_val)

    return features


class DecisionDiscovery:
    def __init__(self,
                 petri_net: Tuple,
                 sorted_case_ids: List[str],
                 event_log_df: pd.DataFrame,
                 alignments: List[Any],
                 numeric_scalers: Optional[Dict[str, Any]] = None) -> None:
        """
        Args
        ----
        numeric_scalers: optional ``{column_name: fitted scaler}`` mapping.
            Applied to numeric columns of ``event_log_df`` (including the
            computed elapsed-time columns) BEFORE feature collection so the
            decision miner trains in the same space the LSTM sees at runtime.
            Persisted by ``save_results`` and auto-loaded by ``DecisionLabeler``.
        """
        self.net, self.im, self.fm = petri_net
        self.sorted_case_ids = sorted_case_ids
        self.event_log_df = event_log_df
        self.alignments = alignments
        self.numeric_scalers: Dict[str, Any] = dict(numeric_scalers or {})

        self.__create_case_elapsed_time_column()
        self.__create_event_elapsed_time_column()
        self._apply_numeric_scalers()

        self.dynamic_attribute_names: List[str] = []
        self.static_attribute_names: List[str] = []

        self.transition_by_key: Dict[str, Any] = {str(t.name): t for t in self.net.transitions}
        self.decision_places: List[Any] = [p for p in self.net.places if len(p.out_arcs) > 1]

    def __create_case_elapsed_time_column(self) -> None:
        case_col = "case:concept:name"
        ts_col = "time:timestamp"
        case_start = self.event_log_df.groupby(case_col)[ts_col].transform("min")
        self.event_log_df["case_elapsed_time"] = (
            self.event_log_df[ts_col] - case_start
        ).dt.total_seconds()

    def __create_event_elapsed_time_column(self) -> None:
        case_col = "case:concept:name"
        ts_col = "time:timestamp"
        elapsed = self.event_log_df.groupby(case_col)[ts_col].diff().dt.total_seconds()
        self.event_log_df["event_elapsed_time"] = elapsed.fillna(0.0)

    def _apply_numeric_scalers(self) -> None:
        if not self.numeric_scalers:
            return
        for col, scaler in self.numeric_scalers.items():
            if col not in self.event_log_df.columns:
                continue
            series = pd.to_numeric(self.event_log_df[col], errors="coerce")
            mask = series.notna()
            if not mask.any():
                continue
            transformed = np.asarray(
                scaler.transform(series[mask].to_numpy().reshape(-1, 1))
            ).reshape(-1)
            new_values = series.astype(float).copy()
            new_values.loc[mask] = transformed
            self.event_log_df[col] = new_values

    def _filter_attributes(self,
                           attrs: Dict[str, Any],
                           attributes: Optional[List[str]]) -> Dict[str, Any]:
        if not attrs:
            return {}
        filtered = {k: attrs.get(k, np.nan) for k in attributes} if attributes else attrs
        out: Dict[str, Any] = {}
        for k, v in filtered.items():
            if isinstance(v, (int, np.integer)):
                out[k] = str(int(v))
            else:
                out[k] = v
        return out

    def _collect_sync_events(self,
                             case: pd.DataFrame,
                             case_alignment: List[Any],
                             dyn_keys: List[str]) -> List[Dict[str, Any]]:
        """
        Ordered list of (filtered) attribute dicts, one per synchronous move of
        the alignment, matched to the corresponding event row in ``case``.
        Index ``c`` of the returned list is the ``c``-th visible event.
        """
        sync_events: List[Dict[str, Any]] = []
        cursor = 0
        for (log_name, model_name), (log_label, model_label) in case_alignment:
            if model_name == ">>" or log_name == ">>":
                continue
            if self.transition_by_key.get(model_name) is None:
                continue
            candidate_labels = [lbl for lbl in [log_label, model_label] if lbl]
            matched: Dict[str, Any] = {}
            for pos in range(cursor, len(case)):
                if case.iloc[pos].get("concept:name", None) in candidate_labels:
                    matched = self._filter_attributes(case.iloc[pos].to_dict(), dyn_keys)
                    cursor = pos + 1
                    break
            sync_events.append(matched)
        return sync_events

    def collect_I(self,
                  dynamic_attributes: Optional[List[str]] = None,
                  static_attributes: Optional[List[str]] = None
                  ) -> Dict[Any, List[Tuple[Dict[str, Any], Any]]]:
        """
        For every decision place p, collect supervised samples ``(η, a*)``.

        We replay each optimal alignment (:func:`replay_alignment_decisions`)
        to obtain, for every firing that consumes a token from a decision place
        p, the next visible activity ``a*`` reached on that branch (or ``EOS``)
        together with the number of synchronous events that preceded it. The
        data state ``η`` is then built from those preceding synchronous events
        (the most recent one and the average over earlier ones).
        """
        dyn_keys = list(dynamic_attributes or [])
        sta_keys = list(static_attributes or [])
        self.dynamic_attribute_names = dyn_keys
        self.static_attribute_names = sta_keys

        decision_places = set(self.decision_places)
        I: Dict[Any, List[Tuple[Dict[str, Any], Any]]] = defaultdict(list)

        for case_id, case_alignment in zip(self.sorted_case_ids, self.alignments):
            case = self.event_log_df[
                self.event_log_df["case:concept:name"] == case_id
            ].reset_index(drop=True)

            static_attrs_case: Dict[str, Any] = {}
            if sta_keys and len(case) > 0:
                static_attrs_case = self._filter_attributes(case.iloc[0].to_dict(), sta_keys)

            sync_events = self._collect_sync_events(case, case_alignment, dyn_keys)
            records = replay_alignment_decisions(case_alignment,
                                                 im=self.im,
                                                 transition_by_name=self.transition_by_key,
                                                 decision_places=decision_places)
            for rec in records:
                c = rec["sync_index"]
                eta = {"past_events": list(sync_events[:c]), **static_attrs_case}
                I[rec["place"]].append((eta, rec["target"]))

        return I

    def _build_feature_row(self, attrs: Dict[str, Any]) -> Dict[str, Any]:
        return build_feature_row(attrs,
                                 dynamic_attributes=self.dynamic_attribute_names,
                                 static_attributes=self.static_attribute_names)

    def mine_decision_models(self,
                             dynamic_attributes: Optional[List[str]] = None,
                             static_attributes: Optional[List[str]] = None,
                             mc_config: Optional[ModelConfig] = None) -> DecisionMiningResult:
        I = self.collect_I(dynamic_attributes=dynamic_attributes,
                           static_attributes=static_attributes)

        models: Dict[str, DecisionPointModel] = {}
        skipped: List[str] = []

        for place, samples in I.items():
            if not samples:
                skipped.append(place)
                continue
            rows = [self._build_feature_row(ass) for ass, _ in samples]
            labels = [str(chosen) for _, chosen in samples]
            X_raw = pd.DataFrame(rows)
            est = FunctionEstimator.fit_from_xy(X_raw, labels,
                                                model_cfg=mc_config or ModelConfig())
            models[place] = DecisionPointModel(place_name=place, estimator=est)

        return DecisionMiningResult(models=models, skipped=skipped)

    def extract_guards(self,
                       mining_result: DecisionMiningResult,
                       **_ignored: Any) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        return {place_name: model.estimator.extract_guards()
                for place_name, model in mining_result.models.items()}

    def save_results(self,
                     *,
                     guards: Dict[str, Dict[str, List[Dict[str, Any]]]],
                     mining_result: Optional[DecisionMiningResult] = None,
                     output_dir: Optional[str] = None) -> Dict[str, str]:
        base_dir = Path(output_dir) if output_dir else Path.cwd() / "decision_mining_results"
        base_dir.mkdir(parents=True, exist_ok=True)
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        guards_json = base_dir / "guards.json"
        guards_csv = base_dir / "guards_flat.csv"
        skipped_csv = base_dir / "skipped_places.csv"
        per_place_json = base_dir / "decision_places_bundle.json"

        def _safe(name: str) -> str:
            return (str(name).replace("/", "_").replace("\\", "_")
                    .replace(" ", "_").replace(":", "_"))

        def _incoming_transition_tuples(place_obj: Any) -> List[Tuple[str, Optional[str]]]:
            if place_obj is None or not hasattr(place_obj, "in_arcs"):
                return []
            out: List[Tuple[str, Optional[str]]] = []
            seen: set = set()
            for arc in list(getattr(place_obj, "in_arcs", [])):
                trans = getattr(arc, "source", None)
                if trans is None:
                    continue
                key = (str(getattr(trans, "name", "")),
                       str(getattr(trans, "label", None)) if getattr(trans, "label", None) is not None else None)
                if key in seen:
                    continue
                seen.add(key)
                out.append(key)
            return out

        # 1) Full guards JSON
        guards_norm = {str(p): {str(act): gs for act, gs in by_act.items()}
                        for p, by_act in guards.items()}
        with guards_json.open("w", encoding="utf-8") as f:
            json.dump(guards_norm, f, indent=2, ensure_ascii=False, default=str)

        # 2) Per-place bundle (place → previous transitions + model path + guards)
        models_by_place = {} if mining_result is None else dict(mining_result.models)
        per_place_records: List[Dict[str, Any]] = []
        for place_key, guard_for_place in guards.items():
            place_name = str(place_key)
            model_path = models_dir / f"{_safe(place_name)}.pkl"
            mobj = models_by_place.get(place_key)
            if mobj is None:
                for mk, mv in models_by_place.items():
                    if str(mk) == place_name:
                        mobj = mv
                        break
            estimator = None if mobj is None else getattr(mobj, "estimator", None)
            if estimator is not None:
                with model_path.open("wb") as f:
                    pickle.dump(estimator.to_artifact(), f)
                model_path_str = str(model_path)
            else:
                model_path_str = ""
            place_obj_for_arcs = place_key if hasattr(place_key, "in_arcs") else None
            per_place_records.append({"place_name": place_name,
                                      "previous_transitions": _incoming_transition_tuples(place_obj_for_arcs),
                                      "model_path": model_path_str,
                                      "guards": guard_for_place})
        with per_place_json.open("w", encoding="utf-8") as f:
            json.dump(per_place_records, f, indent=2, ensure_ascii=False, default=str)

        # 3) Flat guards table
        flat_rows: List[Dict[str, Any]] = []
        for place_name, by_act in guards.items():
            for act, gs in by_act.items():
                for g in gs:
                    flat_rows.append({"place_name": str(place_name),
                                      "activity_label": str(act),
                                      "rule": g.get("rule", ""),
                                      "prob_model": g.get("prob_model", np.nan),
                                      "prob_emp": g.get("prob_emp", np.nan),
                                      "support": g.get("support", np.nan),
                                      "coverage": g.get("coverage", np.nan),
                                      "lift": g.get("lift", np.nan),
                                      "intervals": json.dumps(g.get("intervals", {}), ensure_ascii=False),
                                      "categorical_allowed": json.dumps(g.get("categorical_allowed", {}), ensure_ascii=False),
                                      "categorical_excluded": json.dumps(g.get("categorical_excluded", {}), ensure_ascii=False)})
        pd.DataFrame(flat_rows).to_csv(guards_csv, index=False)

        # 4) Skipped places
        skipped = [] if mining_result is None else list(mining_result.skipped)
        pd.DataFrame({"skipped_place": [str(p) for p in skipped]}).to_csv(skipped_csv, index=False)

        # 5) Numeric scalers
        numeric_scalers_path = ""
        if self.numeric_scalers:
            sp = base_dir / "numeric_scalers.pkl"
            with sp.open("wb") as f:
                pickle.dump(dict(self.numeric_scalers), f)
            numeric_scalers_path = str(sp)

        return {"output_dir": str(base_dir),
                "guards_json_path": str(guards_json),
                "guards_flat_csv_path": str(guards_csv),
                "skipped_places_path": str(skipped_csv),
                "per_place_json_path": str(per_place_json),
                "model_dir": str(models_dir),
                "numeric_scalers_path": numeric_scalers_path}
