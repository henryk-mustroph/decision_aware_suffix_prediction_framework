"""
Unified dataset construction.

- build_base_dataset(ds): raw CSV -> 'normal' tensor datasets (+ raw prefix CSVs, + discovered Petri net)
- build_decision_labeled_dataset(ds):'normal' tensors -> 'decision_labeled', tensors with dense guard tensors
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pickle
import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .configs import (DATASETS,
                      DatasetConfig,
                      DatasetPaths,
                      resolve_dataset_paths)

DatasetLike = Union[str, DatasetConfig]

def _as_dataset(ds: DatasetLike) -> DatasetConfig:
    return DATASETS[ds] if isinstance(ds, str) else ds

# BPIC20 activity cleaning
def _strip_by_suffix(value):
    if pd.isna(value):
        return value
    s = str(value)
    if s == "EOS":
        return s
    s = re.sub(r"\s+by\s+[^\]]+(?=\])", "", s)
    s = re.sub(r"\s+by\s+.*$", "", s)
    return s

def _ensure_clean_log(ds: DatasetConfig, paths: DatasetPaths) -> None:
    """
    Write the cleaned activity CSV expected at `event_log_location`.

    Always regenerated from the raw source when that source is present, so a
    pipeline run never silently reuses a stale cleaned log. Falls back to an
    existing cleaned file only when there is no source to rebuild from.
    """
    if not ds.event_log.clean_activity_by_suffix:
        return
    if paths.raw_source_log is None or not paths.raw_source_log.exists():
        if paths.raw_event_log.exists():
            return
        raise FileNotFoundError(
            f"{ds.key}: raw source log {paths.raw_source_log} not found for cleaning")
    df_raw = pd.read_csv(paths.raw_source_log)
    df_raw[ds.concept_name] = df_raw[ds.concept_name].map(_strip_by_suffix)
    paths.raw_event_log.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(paths.raw_event_log, index=False)
    print(f"{ds.key}: wrote cleaned log to {paths.raw_event_log}")

# Base dataset build
def build_base_dataset(ds: DatasetLike,
                       *,
                       root=None,
                       discover_petri_net: bool = True,
                       net_noise_threshold: Optional[float] = None,
                       export_prefix_csv: bool = True,
                       seed: int = 17) -> dict:
    """
    Build and persist the 'normal' tensor datasets (train/val/test).
    Returns the saved tensor-dataset paths.

    net_noise_threshold: Inductive-Miner noise filter for Petri-net discovery.
        When None (default) the per-dataset value from
        ``ds.event_log.net_noise_threshold`` is used; pass a float to override
        it (e.g. for a one-off sweep without touching the config).
    """
    from data_processing.labeling_encoding import PrefixesDataFrameLoader, EventLogLoader

    ds = _as_dataset(ds)
    paths = resolve_dataset_paths(ds, root=root)
    np.random.seed(seed)

    if net_noise_threshold is None:
        net_noise_threshold = float(getattr(ds.event_log, "net_noise_threshold", 0.0))
    print(f"{ds.key}: Petri-net discovery noise_threshold = {net_noise_threshold}")

    _ensure_clean_log(ds, paths)
    if not paths.raw_event_log.exists():
        raise FileNotFoundError(f"{ds.key}: raw event log not found at {paths.raw_event_log}")

    props = ds.event_log.to_event_log_properties(ds.concept_name)
    event_log_location = str(paths.raw_event_log)

    prefix_loader = PrefixesDataFrameLoader(event_log_location=event_log_location,
                                            event_log_properties=props)

    # 1) Raw prefix CSVs (used later as the case-id source for alignments).
    split_case_ids: dict[str, List] = {}
    if export_prefix_csv:
        paths.raw_prefix_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "val", "test"):
            pref_df = prefix_loader.get_dataset(split)
            pref_df.to_csv(paths.raw_prefix_csv(ds, split), index=False)
            if ds.event_log.case_name in pref_df.columns:
                split_case_ids[split] = pref_df[ds.event_log.case_name].dropna().unique().tolist()

    # 2) Petri-net discovery (optional; required before decision labeling).
    if discover_petri_net:
        _discover_and_save_petri_net(ds, paths, event_log_location, split_case_ids, net_noise_threshold)

    # 3) Encode + save tensor datasets.
    el_loader = EventLogLoader(event_log_location=event_log_location,
                               event_log_properties=props,
                               prefix_df=prefix_loader)
    paths.normal_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for split in ("train", "val", "test"):
        dataset = el_loader.get_dataset(split)
        out = paths.normal_tensor(ds, split)
        torch.save(dataset, out)
        saved[split] = out
        print(f"{ds.key}: saved {split} -> {out}  ({len(dataset)} prefixes)")
    return saved

def _discover_and_save_petri_net(ds: DatasetConfig,
                                 paths: DatasetPaths,
                                 event_log_location: str,
                                 split_case_ids: dict,
                                 net_noise_threshold: float = 0) -> None:
    """
    Inductive-Miner discovery over train+val cases (mirrors base loader).
    """
    import importlib
    import data_processing.petri_net_replay_markings as pnm
    importlib.reload(pnm)
    from data_processing.petri_net_replay_markings import InductiveMiner

    case_ids = list(dict.fromkeys(
        split_case_ids.get("train", []) + split_case_ids.get("val", [])))

    el = ds.event_log
    resource_col = next((c for c in el.cat_dynamic if c != ds.concept_name), None)
    miner = InductiveMiner(path_to_csv_log=event_log_location,
                           case_id_col=el.case_name,
                           activity_col=ds.concept_name,
                           timestamp_col=el.timestamp_name,
                           resource_col=resource_col)
    paths.petri_net_png.parent.mkdir(parents=True, exist_ok=True)
    
    net, im, fm = miner.discover_petri_net(visulaize=True,
                                           case_ids=case_ids or None,
                                           store_loc_file_path=str(paths.petri_net_png),
                                           noise_threshold=net_noise_threshold)
    
    with open(paths.petri_net_pkl, "wb") as f:
        pickle.dump((net, im, fm), f)
    print(f"{ds.key}: saved Petri net -> {paths.petri_net_pkl}")

# Decision-labeled dataset build
def build_decision_labeled_dataset(ds: DatasetLike, *, root=None) -> dict:
    """
    Build the decision-labeled train/val tensors used for decision-aware training.
    """
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.algo.conformance.alignments.petri_net import algorithm as alignment_algo
    from pm4py.utils import get_properties
    from data_processing.decision_labeling import DecisionLabeler

    ds = _as_dataset(ds)
    paths = resolve_dataset_paths(ds, root=root)
    el = ds.event_log

    # 1) Petri net + normal tensors.
    with open(paths.petri_net_pkl, "rb") as f:
        net, im, fm = pickle.load(f)
    train_set = torch.load(paths.normal_tensor(ds, "train"), weights_only=False)
    val_set = torch.load(paths.normal_tensor(ds, "val"), weights_only=False)

    # 2) Case IDs (train+val), in CSV order.
    def _case_ids(split: str) -> List:
        df = pd.read_csv(paths.raw_prefix_csv(ds, split))
        return df[el.case_name].dropna().unique().tolist()

    all_case_ids = list(dict.fromkeys(_case_ids("train") + _case_ids("val")))

    # 3) Original event log -> pm4py column names + datetime.
    el_df = pd.read_csv(paths.raw_event_log).rename(columns={
        el.case_name: "case:concept:name",
        ds.concept_name: "concept:name",
        el.timestamp_name: "time:timestamp",
    })
    if el.date_format is not None:
        el_df["time:timestamp"] = pd.to_datetime(el_df["time:timestamp"],
                                                  format=el.date_format)
    el_df = dataframe_utils.convert_timestamp_columns_in_df(el_df)

    el_aligned = el_df[el_df["case:concept:name"].isin(all_case_ids)].copy()

    # 4) Optimal alignments (tuple transition descriptors, as the notebook used).
    params = get_properties(el_aligned)
    params["ret_tuple_as_trans_desc"] = True
    aligned_results = alignment_algo.apply(el_aligned, net, im, fm, parameters=params)
    all_alignments = [r["alignment"] for r in aligned_results]
    sorted_case_ids = (el_aligned.drop_duplicates(subset="case:concept:name", keep="first")
                       ["case:concept:name"].tolist())
    print(f"{ds.key}: {len(all_alignments)} alignments for {len(sorted_case_ids)} cases")

    # 5) Offline decision labeling (decision-model attribute subset).
    labeler = DecisionLabeler(petri_net=(net, im, fm),
                              decision_model_dir=str(paths.decision_model_dir),
                              decision_places_bundle_path=str(paths.decision_bundle),
                              dynamic_attributes=list(ds.dynamic_attributes),
                              static_attributes=list(ds.static_attributes))
    labeler.label_dataset_offline(train_set, el_aligned, sorted_case_ids, all_alignments)
    labeler.label_dataset_offline(val_set, el_aligned, sorted_case_ids, all_alignments)

    # 6) Dense guard tensors keyed on the activity (concept_name) feature.
    concept_idx = _concept_feature_index(train_set, ds.concept_name)
    for name, dataset in (("train", train_set), ("val", val_set)):
        dataset.prepare_guard_tensors(concept_idx)
        print(f"{ds.key} {name}: guard_targets {tuple(dataset._guard_targets.shape)}, "
              f"guard_mask {tuple(dataset._guard_mask.shape)}")

    # 7) Save.
    paths.decision_labeled_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for split, dataset in (("train", train_set), ("val", val_set)):
        out = paths.decision_tensor(ds, split)
        torch.save(dataset, out)
        saved[split] = out
        print(f"{ds.key}: saved decision-labeled {split} -> {out}")
    return saved

def _concept_feature_index(dataset, concept_name: str) -> int:
    """
    Index of the activity feature inside all_categories[0].
    """
    for i, cat in enumerate(dataset.all_categories[0]):
        if cat[0] == concept_name:
            return i
    raise ValueError(f"concept feature '{concept_name}' not found in dataset categories")
