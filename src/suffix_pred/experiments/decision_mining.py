"""
Decision mining 
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pickle
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch

from .configs import DATASETS, DatasetConfig, DatasetPaths, resolve_dataset_paths

DatasetLike = Union[str, DatasetConfig]

def _as_dataset(ds: DatasetLike) -> DatasetConfig:
    return DATASETS[ds] if isinstance(ds, str) else ds

# Shared pm4py helpers
def _prepare_split_log(base_df: pd.DataFrame, keep_case_ids, ds: DatasetConfig) -> pd.DataFrame:
    """Filter to cases, rename to pm4py columns, parse timestamps, sort."""
    el = ds.event_log
    out = base_df[base_df[el.case_name].isin(set(keep_case_ids))].copy()
    rename = {}
    if el.case_name != "case:concept:name":
        rename[el.case_name] = "case:concept:name"
    if ds.concept_name != "concept:name":
        rename[ds.concept_name] = "concept:name"
    if el.timestamp_name != "time:timestamp":
        rename[el.timestamp_name] = "time:timestamp"
    if rename:
        out = out.rename(columns=rename)
    if "time:timestamp" in out.columns:
        out["time:timestamp"] = pd.to_datetime(out["time:timestamp"], errors="coerce")
    sort_cols = ["case:concept:name"] + (["time:timestamp"] if "time:timestamp" in out.columns else [])
    return out.sort_values(sort_cols).reset_index(drop=True)

def _alignments(log_df: pd.DataFrame, net, im, fm) -> List:
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
    from pm4py.utils import get_properties
    params = get_properties(log_df)
    params["ret_tuple_as_trans_desc"] = True
    return [a["alignment"] for a in alignments.apply(log_df, net, im, fm, parameters=params)]

def _case_ids(paths: DatasetPaths, ds: DatasetConfig, split: str) -> List:
    df = pd.read_csv(paths.raw_prefix_csv(ds, split))
    return sorted(df[ds.event_log.case_name].dropna().unique().tolist())

def _numeric_scalers(paths: DatasetPaths, ds: DatasetConfig) -> Dict[str, object]:
    """
    The LSTM's StandardScalers for the numeric decision attributes, so the
    decision miner trains in the same feature space the LSTM sees at runtime.
    """
    data = torch.load(paths.normal_tensor(ds, "train"), weights_only=False)
    encoders = data.encoder_decoder.continuous_encoders
    attrs = list(ds.dynamic_attributes) + list(ds.static_attributes)
    return {a: encoders[a] for a in attrs if a in encoders}

# Mining
def mine_decision_models(ds: DatasetLike, *, root=None, mc_config=None) -> Tuple:
    """
    Discover and persist per-decision-place models. Returns
    (mining_result, guards, result_paths).
    """
    from decision_mining.decision_discovery import DecisionDiscovery
    from decision_mining.function_estimator_catboost_advanced import ModelConfig as CatBoostConfig

    ds = _as_dataset(ds)
    paths = resolve_dataset_paths(ds, root=root)

    with open(paths.petri_net_pkl, "rb") as f:
        net, im, fm = pickle.load(f)

    case_ids_trainval = sorted(set(_case_ids(paths, ds, "train")) | set(_case_ids(paths, ds, "val")))
    event_log_df = pd.read_csv(paths.raw_event_log)
    log_trainval = _prepare_split_log(event_log_df, case_ids_trainval, ds)
    alignments_trainval = _alignments(log_trainval, net, im, fm)
    numeric_scalers = _numeric_scalers(paths, ds)

    print(f"{ds.key}: mining over {len(case_ids_trainval)} train+val cases "
          f"({len(log_trainval)} events); numeric scalers: {list(numeric_scalers)}")

    dd = DecisionDiscovery(petri_net=(net, im, fm),
                           sorted_case_ids=case_ids_trainval,
                           event_log_df=log_trainval,
                           alignments=alignments_trainval,
                           numeric_scalers=numeric_scalers)
    res = dd.mine_decision_models(dynamic_attributes=list(ds.dynamic_attributes),
                                  static_attributes=list(ds.static_attributes),
                                  mc_config=mc_config or CatBoostConfig())
    guards = dd.extract_guards(mining_result=res)

    output_dir = str(paths.decision_bundle.parent)   # data_aware_Petri_net/
    result_paths = dd.save_results(guards=guards, mining_result=res, output_dir=output_dir)
    print(f"{ds.key}: saved decision artifacts -> {output_dir}")
    return res, guards, result_paths

# Held-out diagnostics (per decision place): Check decision mining model accuracy
def decision_diagnostics(ds: DatasetLike, *, root=None) -> Tuple[pd.DataFrame, dict]:
    """
    Per-decision-place top-1/top-3/mean-true-prob on the held-out test set.
    Returns (diagnostics_df, weighted_summary).
    """
    from data_processing.decision_labeling import DecisionLabeler, compute_dp_diagnostics

    ds = _as_dataset(ds)
    paths = resolve_dataset_paths(ds, root=root)

    with open(paths.petri_net_pkl, "rb") as f:
        net, im, fm = pickle.load(f)

    case_ids_test = _case_ids(paths, ds, "test")
    log_test = _prepare_split_log(pd.read_csv(paths.raw_event_log), case_ids_test, ds)
    alignments_test = _alignments(log_test, net, im, fm)

    numeric_scalers = None
    if paths.numeric_scalers.exists():
        with open(paths.numeric_scalers, "rb") as f:
            numeric_scalers = pickle.load(f)

    labeler = DecisionLabeler(petri_net=(net, im, fm),
                              decision_model_dir=str(paths.decision_model_dir),
                              decision_places_bundle_path=str(paths.decision_bundle),
                              dynamic_attributes=list(ds.dynamic_attributes),
                              static_attributes=list(ds.static_attributes),
                              numeric_scalers=numeric_scalers)

    decision_data, true_next = labeler.collect_decision_instances(
        event_log_df=log_test, sorted_case_ids=case_ids_test, alignments=alignments_test)
    dp_diag = compute_dp_diagnostics(decision_data=decision_data, true_next_activities=true_next)

    diag_df = (pd.DataFrame.from_dict(dp_diag, orient="index")
               .reset_index().rename(columns={"index": "decision_place"})
               .sort_values("decision_place").reset_index(drop=True))

    weighted = {}
    if not diag_df.empty and "support" in diag_df.columns:
        total = float(diag_df["support"].sum())
        if total > 0:
            weighted = {"weighted_top1": float((diag_df["top1_accuracy"] * diag_df["support"]).sum() / total),
                        "weighted_top3": float((diag_df["top3_accuracy"] * diag_df["support"]).sum() / total),
                        "total_support": int(total)}

    return diag_df, weighted

# Cross-dataset diagnostics: one comparable table over every dataset
def decision_diagnostics_all(datasets: List[DatasetLike] = None,
                             *,
                             root=None,
                             informative_lift: float = 0.05,
                             min_support: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run :func:`decision_diagnostics` for several datasets and stack the per-place
    rows into one comparable table, so the four pipelines can be read side by
    side rather than one notebook at a time.

    A decision place is flagged ``informative`` when its mined model genuinely
    beats "always predict the majority branch" on held-out data:
        - ``n_branches >= 2``                     (there is a real choice), AND
        - ``support >= min_support``              (enough held-out firings to trust), AND
        - ``top1_accuracy - majority_baseline > informative_lift``.
    This is the same rule the per-pipeline cell uses; only places that clear it
    are worth steering during guided decode, so the per-dataset coverage figure
    is the headline number for "should guidance help this dataset at all?".

    Returns
    -------
    (places_df, summary_df)
        ``places_df``  : one row per (dataset, decision_place) with the diagnostic
                         columns plus ``lift_over_majority`` and ``informative``.
        ``summary_df`` : one row per dataset with the support-weighted top-1/top-3,
                         the informative-place count, and the share of held-out
                         decision instances those informative places cover.
    """
    if datasets is None:
        datasets = list(DATASETS.keys())

    place_frames: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for ds_like in datasets:
        ds = _as_dataset(ds_like)
        try:
            diag_df, weighted = decision_diagnostics(ds, root=root)
        except FileNotFoundError as exc:
            print(f"{ds.key}: skipped (missing artifacts) -> {exc}")
            continue

        need = {"top1_accuracy", "majority_baseline", "n_branches", "support"}
        if diag_df.empty or not need.issubset(diag_df.columns):
            print(f"{ds.key}: no scorable decision places")
            continue

        diag_df = diag_df.copy()
        diag_df.insert(0, "dataset", ds.key)
        diag_df["lift_over_majority"] = (diag_df["top1_accuracy"] - diag_df["majority_baseline"]).round(4)
        diag_df["informative"] = ((diag_df["n_branches"] >= 2)
                                  & (diag_df["support"] >= min_support)
                                  & (diag_df["lift_over_majority"] > informative_lift))
        place_frames.append(diag_df)

        tot_sup = float(diag_df["support"].sum()) or 1.0
        inf_sup = float(diag_df.loc[diag_df["informative"], "support"].sum())
        summary_rows.append({"dataset": ds.key,
                             "net_noise_threshold": float(getattr(ds.event_log, "net_noise_threshold", 0.0)),
                             "n_places": int(len(diag_df)),
                             "n_informative": int(diag_df["informative"].sum()),
                             "informative_support_pct": round(100.0 * inf_sup / tot_sup, 1),
                             "weighted_top1": round(weighted.get("weighted_top1", float("nan")), 4),
                             "weighted_top3": round(weighted.get("weighted_top3", float("nan")), 4),
                             "total_support": int(tot_sup)})

    cols = ["dataset", "decision_place", "support", "n_branches", "majority_baseline",
            "top1_accuracy", "top3_accuracy", "lift_over_majority", "informative"]
    if place_frames:
        places_df = pd.concat(place_frames, ignore_index=True)
        places_df = places_df[[c for c in cols if c in places_df.columns]
                              + [c for c in places_df.columns if c not in cols]]
        places_df = places_df.sort_values(["dataset", "support"],
                                          ascending=[True, False]).reset_index(drop=True)
    else:
        places_df = pd.DataFrame(columns=cols)
    summary_df = pd.DataFrame(summary_rows)
    return places_df, summary_df
