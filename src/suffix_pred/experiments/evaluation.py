"""
Unified evaluation entry point.

Decode strategy is per-architecture (the three models decode differently):
- UED -> probabilistic MC sampling  (plain) / guided 'mcsa'
- FS  -> arg-max 'mode'             (plain) / guided 'mode'
- GAN -> beam search                (plain) / guided 'beam'
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TORCH_NUM_THREADS", "1")

import pickle
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .configs import ExperimentConfig, ExperimentPaths, resolve_paths

@dataclass
class EvaluationResult:
    outputs: List[dict]
    dls_df: Any                  
    per_prefix: Any              
    avg: float
    summary: dict
    # guided decode only
    reasoning: Optional[List[dict]] = None 

# Model loading (dispatch by architecture)
def load_model(cfg: ExperimentConfig, paths: ExperimentPaths):
    """Load the trained checkpoint for the configured architecture."""
    path = str(paths.model_checkpoint)
    key = cfg.model.key
    if key == "UED":
        from suffix_pred.models.K_UED_LSTM import DropoutUncertaintyEncoderDecoderLSTM
        dropout = cfg.model.extra.get("eval_dropout", cfg.model.dropout)
        return DropoutUncertaintyEncoderDecoderLSTM.load(path, dropout=dropout)
    if key == "FS":
        from suffix_pred.models.FS_LSTM import FullShared_Join_LSTM
        return FullShared_Join_LSTM.load(path)
    if key == "GAN":
        from suffix_pred.models.GAN_LSTM import TaymouriAdversarialLSTM
        return TaymouriAdversarialLSTM.load(path)
    raise NotImplementedError(f"load_model not implemented for model '{key}'")

def _decoding_config(cfg: ExperimentConfig):
    """Build a DecodingConfig matching the model's plain-decode mode."""
    from suffix_pred.evalaution.decode_test_set_suffixes import DecodingConfig
    ex = cfg.model.extra
    mode = ex.get("decode_mode", "mode")
    kwargs: Dict[str, Any] = {"concept_name": cfg.dataset.concept_name,
                              "eos_value": cfg.dataset.eos_value}
    if mode == "probabilistic":
        kwargs["probabilistic_samples"] = cfg.probabilistic_samples
        kwargs["num_processes"] = cfg.num_processes
    if mode == "beam":
        kwargs["beam_width"] = ex.get("beam_width", 3)
    return DecodingConfig(**kwargs), mode

def _score_dls(cfg: ExperimentConfig, outputs: List[dict]):
    """Average Damerau-Levenshtein similarity (overall + per prefix length)."""
    from suffix_pred.evalaution.evalaution_metrics import (
        evaluate_dls, _evaluate_probabilistic, dls_per_prefix_length, average_dls)

    if bool(cfg.model.extra.get("is_probabilistic", False)):
        # Authoritative: probabilistic models (UED, plain or guided) average DLS
        # over their sampled suffixes regardless of the cached row "mode" string
        # (older guided caches stored "guided_mcsa", which the mode-sniffing path
        # in evaluate_dls would otherwise mis-score as deterministic).
        df = _evaluate_probabilistic(outputs, reduction="mean")
    else:
        df = evaluate_dls(outputs)
    return df, dls_per_prefix_length(df), average_dls(df)

def _read_cache(path):
    """Return a cached list of decoded rows, or None if absent/unreadable."""
    try:
        with open(path, "rb") as f:
            cached = pickle.load(f)
        return cached if isinstance(cached, list) else None
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None

# Plain decode
def _evaluate_plain(cfg: ExperimentConfig, paths: ExperimentPaths, *,
                    force: bool = False) -> EvaluationResult:
    from suffix_pred.evalaution.decode_test_set_suffixes import TestSetSuffixDecoder

    mode = cfg.model.extra.get("decode_mode", "mode")
    outputs = None if force else _read_cache(paths.eval_outputs)
    if outputs is None:
        if not force:
            raise FileNotFoundError(
                f"No stored outputs at {paths.eval_outputs.name}; "
                "re-run with RUN_EVAL=True to decode from scratch.")
        model = load_model(cfg, paths)
        test_dataset = torch.load(str(paths.test_dataset), weights_only=False)
        decode_cfg, mode = _decoding_config(cfg)
        decoder = TestSetSuffixDecoder(model=model, dataset=test_dataset, config=decode_cfg)
        paths.eval_outputs.parent.mkdir(parents=True, exist_ok=True)
        # All decode modes (mode / beam / probabilistic) run multi-process.
        outputs = decoder.decode(mode=mode, random_order=False,
                                 cache_path=str(paths.eval_outputs),
                                 reuse_cache=False,
                                 parallel_inference=True,
                                 num_processes=cfg.num_processes)

    df, per_prefix, avg = _score_dls(cfg, outputs)
    summary = {"dataset": cfg.dataset.key, "model": cfg.model.key,
               "variant": cfg.variant.value, "mode": mode,
               "average_dls": avg, "n_rows": len(outputs)}
    return EvaluationResult(outputs, df, per_prefix, avg, summary)


# Guided decode (decision-rule guided reasoning)
_GUIDED_MODE_LABEL = {"mcsa": "guided_probabilistic", "mode": "guided_mode", "beam": "guided_beam"}

def _guided_mode_label(kind: str) -> str:
    return _GUIDED_MODE_LABEL.get(kind, f"guided_{kind}")

# explainable_decision_steps = non-conflicting steps whose chosen branch has a
# data-aware rule; rule_explained_rate = explained / explainable is the primary
# explainability indicator (it factors out conflicts and rule-base coverage
# gaps). non_trivial_* restrict to decision points with >= 2 supported outcomes;
# the all-steps explained_rate is diluted by quasi-deterministic places.
_COUNTER_KEYS = ("decision_steps",
                 "conflicts",
                 "explained_steps",
                 "explainable_decision_steps",
                 "non_trivial_decision_steps",
                 "non_trivial_explained_steps",
                 "matched_trivial_rule",
                 "rule_unmatched",
                 "no_rule_for_branch",
                 "no_matching_rule",
                 "conflict_not_supported")

def _default_reasoning_row() -> dict:
    return {"decision_steps": 0,
            "conflicts": 0,
            "conflict_rate": 0.0,
            "explained_steps": 0,
            "explained_rate": 0.0,
            "explainable_decision_steps": 0,
            "rule_explained_rate": 0.0,
            "non_trivial_decision_steps": 0,
            "non_trivial_explained_steps": 0,
            "non_trivial_explained_rate": 0.0,
            "trace": []}

def _aggregate_reasonings(reasonings: List[dict]) -> dict:
    if not reasonings:
        return _default_reasoning_row()
    agg = {k: int(sum(r.get(k, 0) for r in reasonings)) for k in _COUNTER_KEYS}
    ds = agg["decision_steps"]
    nt = agg["non_trivial_decision_steps"]
    ex = agg["explainable_decision_steps"]
    agg["trivial_decision_steps"] = ds - nt
    agg["conflict_rate"] = float(agg["conflicts"] / ds) if ds else 0.0
    agg["explained_rate"] = float(agg["explained_steps"] / ds) if ds else 0.0
    agg["non_trivial_explained_rate"] = float(agg["non_trivial_explained_steps"] / nt) if nt else 0.0
    agg["rule_explained_rate"] = float(agg["explained_steps"] / ex) if ex else 0.0
    # The structured interpretation (decision point, transition, attribute
    # checks) of the first non-empty trace, for inspection.
    agg["trace"] = next((r["trace"] for r in reasonings if r.get("trace")), [])
    return agg

def _load_petri_net(paths: ExperimentPaths):
    with open(paths.petri_net, "rb") as f:
        return pickle.load(f)   # (net, im, fm)

def _build_decision_labeler(cfg: ExperimentConfig, paths: ExperimentPaths):
    from data_processing.decision_labeling import DecisionLabeler
    numeric_scalers = None
    if paths.numeric_scalers.exists():
        with open(paths.numeric_scalers, "rb") as f:
            numeric_scalers = pickle.load(f)
    
    return DecisionLabeler(petri_net=_load_petri_net(paths),
                           decision_model_dir=str(paths.decision_model_dir),
                           decision_places_bundle_path=str(paths.decision_bundle),
                           dynamic_attributes=list(cfg.dataset.dynamic_attributes),
                           static_attributes=list(cfg.dataset.static_attributes),
                           numeric_scalers=numeric_scalers)

def _guided_extra_kwargs(cfg: ExperimentConfig, kind: str) -> Dict[str, Any]:
    if kind == "mcsa":
        return dict(samples_per_case=cfg.probabilistic_samples,
                    sample_argmax=False, use_variance_cat=True,
                    variational_dropout_sampling=False)
    if kind == "beam":
        return dict(beam_width=cfg.model.extra.get("beam_width", 3))
    return {}

def _assemble_guided_row(item, kind: str):
    """Turn one evaluator.evaluate() item into (output_row, reasoning_row)."""
    case_id, prefix_len, prefix, target_suffix, decoded_suffixes, reasoning = item
    reasonings = reasoning if isinstance(reasoning, list) else [reasoning]
    reasonings = [r or _default_reasoning_row() for r in reasonings]
    out = {"case_id": case_id, "prefix_len": int(prefix_len),
           "prefix": prefix, "target_suffix": target_suffix,
           "decoded_suffixes": decoded_suffixes, "mode": _guided_mode_label(kind)}
    reason = {"case_id": case_id, "prefix_len": int(prefix_len),
              "reasonings": reasonings, "reasoning": _aggregate_reasonings(reasonings)}
    return out, reason

# Parallel guided-decode worker (module-level for pickling). Each process
# rebuilds the DecisionLabeler + guided evaluator from artifact paths, then
# decodes a chunk of cases by restricting the evaluator's ``cases``.
_GUIDED_WORKER = None
_GUIDED_FULL_CASES = None
_GUIDED_KIND = None

def _init_guided_worker(model, dataset, kind, concept_name, eos_value,
                        guidance_kwargs, petri_net_path, decision_model_dir,
                        decision_bundle, dynamic_attributes, static_attributes,
                        numeric_scalers, extra_kwargs):
    global _GUIDED_WORKER, _GUIDED_FULL_CASES, _GUIDED_KIND
    from data_processing.decision_labeling import DecisionLabeler
    from suffix_pred.decision_rule_guided_reasoning_inference import (
        DecisionGuidanceConfig, get_decision_guided_evaluator)
    with open(petri_net_path, "rb") as f:
        net, im, fm = pickle.load(f)
    labeler = DecisionLabeler(petri_net=(net, im, fm),
                              decision_model_dir=decision_model_dir,
                              decision_places_bundle_path=decision_bundle,
                              dynamic_attributes=dynamic_attributes,
                              static_attributes=static_attributes,
                              numeric_scalers=numeric_scalers)
    _GUIDED_WORKER = get_decision_guided_evaluator(
        kind=kind, model=model, dataset=dataset, decision_labeler=labeler,
        guidance_config=DecisionGuidanceConfig(**guidance_kwargs),
        decision_places_bundle_path=decision_bundle,
        concept_name=concept_name, eos_value=eos_value, **extra_kwargs)
    _GUIDED_FULL_CASES = dict(_GUIDED_WORKER.cases)
    _GUIDED_KIND = kind

def _guided_worker_chunk(case_ids):
    if _GUIDED_WORKER is None:
        raise RuntimeError("Guided worker evaluator is not initialized.")
    ev = _GUIDED_WORKER
    ev.cases = {cid: _GUIDED_FULL_CASES[cid] for cid in case_ids if cid in _GUIDED_FULL_CASES}
    out_rows, reason_rows = [], []
    for item in ev.evaluate(random_order=False, return_reasoning=True):
        o, r = _assemble_guided_row(item, _GUIDED_KIND)
        out_rows.append(o)
        reason_rows.append(r)
    return out_rows, reason_rows

def _summarize_guided(cfg, outputs, reasoning_rows, kind) -> EvaluationResult:
    """Score DLS + aggregate reasoning into an EvaluationResult (shared)."""
    all_reasonings = [r for row in reasoning_rows for r in row.get("reasonings", [])]
    df, per_prefix, avg = _score_dls(cfg, outputs)
    agg = _aggregate_reasonings(all_reasonings)
    summary = {"dataset": cfg.dataset.key, "model": cfg.model.key,
               "variant": cfg.variant.value, "mode": _guided_mode_label(kind),
               "average_dls": avg, "n_rows": len(outputs),
               "decision_steps": agg["decision_steps"],
               "conflict_rate": agg["conflict_rate"],
               "explained_rate": agg["explained_rate"],
               "explainable_decision_steps": agg["explainable_decision_steps"],
               "rule_explained_rate": agg["rule_explained_rate"],
               "non_trivial_decision_steps": agg["non_trivial_decision_steps"],
               "non_trivial_explained_rate": agg["non_trivial_explained_rate"],
               "matched_trivial_rule": agg["matched_trivial_rule"],
               "rule_unmatched": agg["rule_unmatched"],
               "no_rule_for_branch": agg["no_rule_for_branch"],
               "no_matching_rule": agg["no_matching_rule"],
               "conflict_not_supported": agg["conflict_not_supported"]}
    return EvaluationResult(outputs, df, per_prefix, avg, summary, reasoning=reasoning_rows)

def _evaluate_guided(cfg: ExperimentConfig, paths: ExperimentPaths,
                     *, parallel: bool = True, force: bool = False) -> EvaluationResult:
    import concurrent.futures
    from suffix_pred.decision_rule_guided_reasoning_inference import (
        DecisionGuidanceConfig, get_decision_guided_evaluator)
    from .configs import require_predicted_decision_attrs

    # Guided decode feeds the decision model the suffix model's PREDICTED dynamic
    # attributes; fail fast if the model cannot predict the ones it needs.
    require_predicted_decision_attrs(cfg.dataset, cfg.model.key)

    kind = cfg.model.extra.get("guided_kind", "mode")

    cached = None if force else _read_cache(paths.eval_outputs)
    if cached is not None:
        reasoning_rows = _read_cache(paths.eval_reasoning) or []
        return _summarize_guided(cfg, cached, reasoning_rows, kind)

    if not force:
        raise FileNotFoundError(
            f"No stored outputs at {paths.eval_outputs.name}; "
            "re-run with RUN_EVAL=True to decode from scratch.")

    model = load_model(cfg, paths)
    test_dataset = torch.load(str(paths.test_dataset), weights_only=False)
    decision_labeler = _build_decision_labeler(cfg, paths)

    g = cfg.guidance
    # Support set S = {a | z(a) >= tau} must be carved at the SAME tau the
    # decision model was labeled/trained with (cfg.model.tau), not the looser
    # GuidanceConfig.support_threshold. Decoupling them (0.05 vs 0.2) inflated
    # the explainability denominator and collapsed rule_explained_rate. Keep this
    # consistent with the conformance path (_evaluate_conformance).
    guidance_kwargs = dict(epsilon=g.epsilon, beta_max=g.beta_max,
                           alpha=g.alpha, support_threshold=float(cfg.model.tau),
                           min_base_entropy=g.min_base_entropy,
                           min_decision_confidence=g.min_decision_confidence,
                           max_guided_steps=g.max_guided_steps)
    extra_kwargs = _guided_extra_kwargs(cfg, kind)

    # Local evaluator: enumerates cases and serves the serial fallback.
    evaluator = get_decision_guided_evaluator(
        kind=kind, model=model, dataset=test_dataset,
        decision_labeler=decision_labeler,
        guidance_config=DecisionGuidanceConfig(**guidance_kwargs),
        decision_places_bundle_path=str(paths.decision_bundle),
        concept_name=cfg.dataset.concept_name, eos_value=cfg.dataset.eos_value,
        **extra_kwargs)

    case_ids = list(evaluator.cases.keys())
    worker_count = max(1, int(cfg.num_processes or 1))
    use_parallel = parallel and worker_count > 1 and len(case_ids) > 1

    outputs: List[dict] = []
    reasoning_rows: List[dict] = []

    if not use_parallel:
        for item in evaluator.evaluate(random_order=False, return_reasoning=True):
            o, r = _assemble_guided_row(item, kind)
            outputs.append(o)
            reasoning_rows.append(r)
    else:
        numeric_scalers = None
        if paths.numeric_scalers.exists():
            with open(paths.numeric_scalers, "rb") as f:
                numeric_scalers = pickle.load(f)
        chunk_size = max(1, (len(case_ids) + worker_count - 1) // worker_count)
        case_chunks = [case_ids[i:i + chunk_size] for i in range(0, len(case_ids), chunk_size)]
        initargs = (model, test_dataset, kind, cfg.dataset.concept_name,
                    cfg.dataset.eos_value, guidance_kwargs, str(paths.petri_net),
                    str(paths.decision_model_dir), str(paths.decision_bundle),
                    list(cfg.dataset.dynamic_attributes),
                    list(cfg.dataset.static_attributes), numeric_scalers, extra_kwargs)
        from tqdm.auto import tqdm as _tqdm
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count, initializer=_init_guided_worker,
                initargs=initargs) as executor:
            futures = [executor.submit(_guided_worker_chunk, ch) for ch in case_chunks]
            for fut in _tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc=f"guided_{kind} chunks"):
                o_rows, r_rows = fut.result()
                outputs.extend(o_rows)
                reasoning_rows.extend(r_rows)

    paths.eval_outputs.parent.mkdir(parents=True, exist_ok=True)
    with open(paths.eval_outputs, "wb") as f:
        pickle.dump(outputs, f)
    with open(paths.eval_reasoning, "wb") as f:
        pickle.dump(reasoning_rows, f)

    return _summarize_guided(cfg, outputs, reasoning_rows, kind)

# Public API
def evaluate(cfg: ExperimentConfig, *, root=None, force: bool = False) -> EvaluationResult:
    """
    Run evaluation, dispatching on the variant's decode strategy.

    ``force=True``: always decode fresh from the current checkpoint and overwrite
    the stored result files. No intermediate caches are consulted.
    ``force=False``: read from the stored result files only; raises
    FileNotFoundError if they do not exist (run once with force=True first).
    """
    paths = resolve_paths(cfg, root=root)
    if cfg.variant.decode == "plain":
        return _evaluate_plain(cfg, paths, force=force)
    return _evaluate_guided(cfg, paths, force=force)

# Decision-constraint conformance (plain decode, guidance disabled)
def _conformance_cache_paths(paths: ExperimentPaths):
    out = paths.eval_outputs.with_name(paths.eval_outputs.stem + "_conformance.pkl")
    reason = paths.eval_reasoning.with_name(paths.eval_reasoning.stem + "_conformance.pkl")
    return out, reason


def _conformance_summary(cfg: ExperimentConfig, reasoning_rows, kind: str, tau: float, n_rows: int) -> dict:
    from suffix_pred.evalaution.evalaution_metrics import aggregate_decision_conformance
    all_reasonings = [r for row in reasoning_rows for r in row.get("reasonings", [])]
    agg = aggregate_decision_conformance(all_reasonings)
    return {"dataset": cfg.dataset.key, "model": cfg.model.key,
            "variant": cfg.variant.value,
            "mode": "plain_" + _guided_mode_label(kind).replace("guided_", ""), "tau": tau,
            "decision_steps": agg["decision_steps"], "conflicts": agg["conflicts"],
            "conflict_rate": agg["conflict_rate"],
            "decision_conformance": agg["decision_conformance"], "n_rows": n_rows}


def evaluate_conformance(cfg: ExperimentConfig, *, tau: Optional[float] = None,
                         root=None, parallel: bool = True,
                         force: bool = False) -> dict:
    """
    Decision-constraint conformance ("Constraint" column, Xu et al. 2018) for a
    *plain*-decoded trained model: the fraction of decision-labeled decoder
    steps whose decoded activity lies in the tau-support S = { a | z(a) >= tau }
    of the decision model.

    The decision-rule-guided evaluator is run with guidance disabled
    (``beta_max = 0`` -> ``_masked_distribution`` is a no-op), so the decode is
    the model's own plain prediction; only the per-step conflict bookkeeping is
    reused. ``tau`` defaults to the model's training tau, so the metric measures
    exactly the constraint the semantic loss was trained against. Use this to
    compare ``clean`` vs ``decision_train`` checkpoints.

    Returns a summary dict (decision_steps, conflict_rate, decision_conformance).
    ``force=True``: always decode fresh and overwrite stored result files.
    ``force=False``: read from stored result files only; raises FileNotFoundError
    if they do not exist (run once with force=True first).
    """
    import concurrent.futures
    from suffix_pred.decision_rule_guided_reasoning_inference import (
        DecisionGuidanceConfig, get_decision_guided_evaluator)
    from .configs import require_predicted_decision_attrs

    # Conformance scores the decoded decision steps against the decision model,
    # which consumes the suffix model's predicted attributes; fail fast on mismatch.
    require_predicted_decision_attrs(cfg.dataset, cfg.model.key)

    paths = resolve_paths(cfg, root=root)
    kind = cfg.model.extra.get("guided_kind", "mode")
    tau = float(cfg.model.tau if tau is None else tau)
    out_cache, reason_cache = _conformance_cache_paths(paths)

    cached_reason = None if force else _read_cache(reason_cache)
    cached_out = None if force else _read_cache(out_cache)
    if cached_reason is not None and cached_out is not None:
        return _conformance_summary(cfg, cached_reason, kind, tau, len(cached_out))

    if not force:
        raise FileNotFoundError(
            f"No stored conformance outputs at {out_cache.name}; "
            "re-run with RUN_EVAL=True to decode from scratch.")

    model = load_model(cfg, paths)
    test_dataset = torch.load(str(paths.test_dataset), weights_only=False)
    decision_labeler = _build_decision_labeler(cfg, paths)

    # Guidance disabled: beta_max = 0 keeps the decode plain; only conflict
    # bookkeeping (against the tau-support) is collected.
    guidance_kwargs = dict(epsilon=cfg.guidance.epsilon, beta_max=0.0,
                           alpha=cfg.guidance.alpha, support_threshold=tau)
    extra_kwargs = _guided_extra_kwargs(cfg, kind)

    evaluator = get_decision_guided_evaluator(
        kind=kind, model=model, dataset=test_dataset,
        decision_labeler=decision_labeler,
        guidance_config=DecisionGuidanceConfig(**guidance_kwargs),
        decision_places_bundle_path=str(paths.decision_bundle),
        concept_name=cfg.dataset.concept_name, eos_value=cfg.dataset.eos_value,
        **extra_kwargs)

    case_ids = list(evaluator.cases.keys())
    worker_count = max(1, int(cfg.num_processes or 1))
    use_parallel = parallel and worker_count > 1 and len(case_ids) > 1

    outputs: List[dict] = []
    reasoning_rows: List[dict] = []

    if not use_parallel:
        for item in evaluator.evaluate(random_order=False, return_reasoning=True):
            o, r = _assemble_guided_row(item, kind)
            outputs.append(o)
            reasoning_rows.append(r)
    else:
        numeric_scalers = None
        if paths.numeric_scalers.exists():
            with open(paths.numeric_scalers, "rb") as f:
                numeric_scalers = pickle.load(f)
        chunk_size = max(1, (len(case_ids) + worker_count - 1) // worker_count)
        case_chunks = [case_ids[i:i + chunk_size] for i in range(0, len(case_ids), chunk_size)]
        initargs = (model, test_dataset, kind, cfg.dataset.concept_name,
                    cfg.dataset.eos_value, guidance_kwargs, str(paths.petri_net),
                    str(paths.decision_model_dir), str(paths.decision_bundle),
                    list(cfg.dataset.dynamic_attributes),
                    list(cfg.dataset.static_attributes), numeric_scalers, extra_kwargs)
        from tqdm.auto import tqdm as _tqdm
        with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count, initializer=_init_guided_worker,
                initargs=initargs) as executor:
            futures = [executor.submit(_guided_worker_chunk, ch) for ch in case_chunks]
            for fut in _tqdm(concurrent.futures.as_completed(futures),
                             total=len(futures), desc=f"conformance ({kind}) chunks"):
                o_rows, r_rows = fut.result()
                outputs.extend(o_rows)
                reasoning_rows.extend(r_rows)

    out_cache.parent.mkdir(parents=True, exist_ok=True)
    with open(out_cache, "wb") as f:
        pickle.dump(outputs, f)
    with open(reason_cache, "wb") as f:
        pickle.dump(reasoning_rows, f)

    return _conformance_summary(cfg, reasoning_rows, kind, tau, len(outputs))

def plot_dls(result: EvaluationResult, cfg: ExperimentConfig, ax=None):
    """DLS-by-prefix-length curve."""
    import matplotlib.pyplot as plt

    pp = result.per_prefix
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 6))
    ax.plot(pp["prefix_len"], pp["dls"], marker="o",
            label=f"{result.summary['mode']} (avg={result.avg:.3f})")
    ax.set_title(f"{cfg.dataset.key} {cfg.model.key}-LSTM ({cfg.variant.value}): "
                 f"DLS by prefix length")
    ax.set_xlabel("Prefix length")
    ax.set_ylabel("Damerau-Levenshtein Similarity (DLS)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return ax

def print_reasoning_example(result: EvaluationResult, max_examples: int = 1) -> None:
    """
    Show the structured interpretation for explained decision steps:
    decision point, local transition e -> a_hat, and per-attribute
    (attr, value, satisfies-rule) checks.
    """
    if not result.reasoning:
        print("No reasoning available (guided variants only).")
        return
    shown = 0
    for row in result.reasoning:
        for r in row.get("reasonings", []):
            trace = r.get("trace") or []
            steps = [s for s in trace if s.get("explained") or s.get("matched_rule")]
            if not steps:
                continue
            print(f"case {row['case_id']} | prefix_len {row['prefix_len']} | "
                  f"explained {r.get('explained_steps', 0)}/{r.get('decision_steps', 0)}")
            for s in steps:
                place = s.get("place")
                inp, nxt = s.get("input_event"), s.get("next_event")
                print(f"  {place}: {inp} -> {nxt}")
                for chk in s.get("attribute_checks", []):
                    print(f"     ({chk.get('attr')}, {chk.get('value')}, "
                          f"{bool(chk.get('in_rule_set', False))})")
            shown += 1
            if shown >= max_examples:
                return
    if shown == 0:
        print("No explained reasoning steps found in the cached results.")
