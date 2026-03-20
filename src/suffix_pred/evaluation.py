import os
import pickle
import random
import concurrent.futures
from dataclasses import dataclass
from typing import Literal, Optional
import pandas as pd
from tqdm.auto import tqdm

from .inference import get_evaluator

DecodeMode = Literal["mode", "probabilistic", "beam"]
ProbReduction = Literal["mean", "best"]

_PROB_WORKER_DECODER = None

def _init_probabilistic_worker(model,
                               dataset,
                               concept_name: str,
                               eos_value: str,
                               samples_per_case: int):
    
    """
    Create one probabilistic decoder per process to avoid per-task reinitialization.
    """
    # currently eval is only for prob. sampling with MCSA
    
    global _PROB_WORKER_DECODER
    _PROB_WORKER_DECODER = get_evaluator(kind="mcsa",
                                         model=model,
                                         dataset=dataset,
                                         concept_name=concept_name,
                                         eos_value=eos_value,
                                         samples_per_case=samples_per_case,
                                         sample_argmax=False,
                                         use_variance_cat=True,
                                         variational_dropout_sampling=False)

def _collect_probabilistic_case_chunk(case_ids: list[str]) -> list[dict]:
    """
    Collect probabilistic suffix samples for a chunk of case ids in a worker.
    """
    if _PROB_WORKER_DECODER is None:
        raise RuntimeError("Probabilistic worker decoder is not initialized.")

    rows: list[dict] = []
    decoder = _PROB_WORKER_DECODER

    for case_id in case_ids:
        full_case = decoder.cases.get(case_id)
        if full_case is None:
            continue

        for prefix_len, prefix, zero_mask, statics, suffix in decoder._iterate_case(full_case):
            prefix_activity = decoder._decode_activity_prefix(prefix)
            target_suffix = decoder._decode_activity_suffix(suffix)
            
            decoded_suffixes = decoder.predict_probabilistic_suffix(prefix=prefix,
                                                                    prefix_len=prefix_len,
                                                                    static_inputs=statics,
                                                                    mask=zero_mask,
                                                                    include_model_states=False)

            rows.append({"case_id": case_id,
                         "prefix_len": int(prefix_len),
                         "prefix": prefix_activity,
                         "target_suffix": target_suffix,
                         "decoded_suffixes": decoded_suffixes,
                         "mode": "probabilistic"})

    return rows

@dataclass
class DLSConfig:
    concept_name: str = "Activity"
    eos_value: str = "EOS"
    probabilistic_samples: Optional[int] = 100
    beam_width: Optional[int] = 3
    num_processes: Optional[int] = 8

def damerau_levenshtein_distance(seq_a: list, seq_b: list) -> int:
    """
    Compute Damerau-Levenshtein edit distance between two token sequences.
    """
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1,
                           dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + cost)
            if (i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[n][m]


def dls_score(target_activity_seq: list, predicted_activity_seq: list) -> float:
    """
    Compute Damerau-Levenshtein Similarity.
    """
    denominator = max(len(target_activity_seq), len(predicted_activity_seq), 1)
    distance = damerau_levenshtein_distance(target_activity_seq, predicted_activity_seq)
    return 1.0 - (distance / denominator)


class DLSEvaluation:
    """
    Activity-only DLS evaluation for mode, probabilistic and beam decoding.
    """
    def __init__(self, model, dataset, config: DLSConfig | None = None):
        self.model = model
        self.dataset = dataset
        self.config = config or DLSConfig()

    def _build_decoder(self, mode: DecodeMode):
        common = {"model": self.model,
                  "dataset": self.dataset,
                  "concept_name": self.config.concept_name,
                  "eos_value": self.config.eos_value}

        if mode == "mode":
            return get_evaluator(kind="mode", **common)

        if mode == "probabilistic":
            return get_evaluator(kind="mcsa",
                                 samples_per_case=self.config.probabilistic_samples,
                                 sample_argmax=False,
                                 use_variance_cat=True,
                                 variational_dropout_sampling=False,
                                 **common)

        if mode == "beam":
            return get_evaluator(kind="beam",
                                 beam_width=self.config.beam_width,
                                 **common)

        raise ValueError("Unsupported mode. Use one of: 'mode', 'probabilistic', 'beam'.")

    def evaluate(self, mode: DecodeMode, random_order: bool = False) -> pd.DataFrame:
        """
        Evaluate DLS for each (case, prefix length).

        Outputs:
        - case_id, prefix_len, dls, mode
        """
        if mode == "probabilistic":
            return self.evaluate_probabilistic(random_order=random_order, reduction="mean")

        decoder = self._build_decoder(mode)

        rows: list[dict] = []
        for case_id, prefix_len, _prefix, target_suffix, decoded_suffixes in decoder.evaluate(
            random_order=random_order):
            prediction = decoded_suffixes[0] if len(decoded_suffixes) > 0 else []
            dls_value = float(dls_score(target_suffix, prediction))

            rows.append({"case_id": case_id,
                         "prefix_len": int(prefix_len),
                         "dls": dls_value,
                         "mode": mode})

        return pd.DataFrame(rows)

    def collect_inference_outputs(self,
                                  mode: DecodeMode,
                                  random_order: bool = False,
                                  cache_path: Optional[str] = None,
                                  reuse_cache: bool = False,
                                  parallel_inference: bool = True,
                                  num_processes: Optional[int] = None) -> list[dict]:
        """
        Collect decoded suffix outputs and optionally persist them as a pickle.

        Output row:
        - case_id, prefix_len, prefix, target_suffix, decoded_suffixes, mode
        """
        if cache_path is not None and reuse_cache:
            try:
                with open(cache_path, "rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, list):
                    return cached
            except FileNotFoundError:
                pass

        if mode == "probabilistic":
            outputs = self.collect_probabilistic_inference_samples(random_order=random_order,
                                                                   cache_path=cache_path,
                                                                   reuse_cache=reuse_cache,
                                                                   parallel_inference=parallel_inference,
                                                                   num_processes=num_processes)
            return outputs

        decoder = self._build_decoder(mode)
        outputs: list[dict] = []
        for case_id, prefix_len, prefix, target_suffix, decoded_suffixes in decoder.evaluate(random_order=random_order):
            outputs.append({"case_id": case_id,
                            "prefix_len": int(prefix_len),
                            "prefix": prefix,
                            "target_suffix": target_suffix,
                            "decoded_suffixes": decoded_suffixes,
                            "mode": mode})

        if cache_path is not None:
            with open(cache_path, "wb") as handle:
                pickle.dump(outputs, handle)

        return outputs

    def evaluate_from_inference_outputs(self,
                                        outputs: list[dict],
                                        probabilistic_reduction: ProbReduction = "mean",
                                        parallel_scoring: bool = False,
                                        num_processes: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate DLS from precomputed inference outputs produced by collect_inference_outputs.
        """
        if len(outputs) == 0:
            return pd.DataFrame(columns=["case_id", "prefix_len", "dls", "mode"])

        mode = str(outputs[0].get("mode", "")).strip().lower()
        if mode == "probabilistic":
            return self.evaluate_probabilistic_from_samples(samples=outputs,
                                                            reduction=probabilistic_reduction,
                                                            parallel_scoring=parallel_scoring,
                                                            num_processes=num_processes)

        rows: list[dict] = []
        for row in outputs:
            decoded_suffixes = row.get("decoded_suffixes", [])
            prediction = decoded_suffixes[0] if len(decoded_suffixes) > 0 else []
            dls_value = float(dls_score(row.get("target_suffix", []), prediction))
            
            rows.append({"case_id": row.get("case_id"),
                         "prefix_len": int(row.get("prefix_len", 0)),
                         "dls": dls_value,
                         "mode": row.get("mode", mode)})
        return pd.DataFrame(rows)

    def sample_suffix_predictions(self,
                                  mode: DecodeMode,
                                  random_order: bool = False,
                                  max_examples: int = 10,
                                  max_probabilistic_suffixes: Optional[int] = 5) -> pd.DataFrame:
        """
        Return example rows with prefix/target/predicted suffixes for inspection.

        Columns:
        - case_id
        - prefix_len
        - prefix
        - target_suffix
        - predicted_suffixes
        - mode
        """
        if max_examples <= 0:
            return pd.DataFrame(columns=["case_id", "prefix_len", "prefix", "target_suffix", "predicted_suffixes", "mode"])

        outputs = self.collect_inference_outputs(mode=mode,
                                                 random_order=random_order,
                                                 parallel_inference=(mode == "probabilistic"))

        rows: list[dict] = []
        for row in outputs[:max_examples]:
            decoded_suffixes = row.get("decoded_suffixes", [])
            predictions = decoded_suffixes
            
            if (mode == "probabilistic" and max_probabilistic_suffixes is not None and max_probabilistic_suffixes >= 0):
                predictions = decoded_suffixes[:max_probabilistic_suffixes]

            rows.append({"case_id": row.get("case_id"),
                         "prefix_len": int(row.get("prefix_len", 0)),
                         "prefix": row.get("prefix", []),
                         "target_suffix": row.get("target_suffix", []),
                         "predicted_suffixes": predictions,
                         "mode": row.get("mode", mode)})

        return pd.DataFrame(rows)

    def sample_suffix_predictions_all_modes(self,
                                            random_order: bool = False,
                                            max_examples_per_mode: int = 10,
                                            max_probabilistic_suffixes: Optional[int] = 5,
                                            modes: Optional[list[DecodeMode]] = None) -> dict[str, pd.DataFrame]:
        """
        Output example prefix/target/prediction tables for multiple decode modes.
        By default this collects examples for: mode (arg-max), probabilistic, and beam.
        """
        selected_modes = modes or ["mode", "probabilistic", "beam"]

        results: dict[str, pd.DataFrame] = {}
        for mode in selected_modes:
            results[mode] = self.sample_suffix_predictions(mode=mode,
                                                           random_order=random_order,
                                                           max_examples=max_examples_per_mode,
                                                           max_probabilistic_suffixes=max_probabilistic_suffixes)

        return results

    def evaluate_probabilistic(self,
                               random_order: bool = False,
                               reduction: ProbReduction = "mean",
                               parallel_scoring: bool = False,
                               num_processes: Optional[int] = None,
                               inference_cache_path: Optional[str] = None,
                               reuse_inference_cache: bool = False,
                               parallel_inference: bool = True) -> pd.DataFrame:
        """
        Evaluate probabilistic DLS using T sampled suffixes per prefix.
        """
        
        if reduction not in ("mean", "best"):
            raise ValueError("Unsupported reduction. Use one of: 'mean', 'best'.")

        samples = self.collect_probabilistic_inference_samples(random_order=random_order,
                                                               parallel_inference=parallel_inference,
                                                               num_processes=num_processes,
                                                               cache_path=inference_cache_path,
                                                               reuse_cache=reuse_inference_cache)

        return self.evaluate_probabilistic_from_samples(samples=samples,
                                                        reduction=reduction,
                                                        parallel_scoring=parallel_scoring,
                                                        num_processes=num_processes)

    def collect_probabilistic_inference_samples(self,
                                                random_order: bool = False,
                                                cache_path: Optional[str] = None,
                                                reuse_cache: bool = False,
                                                parallel_inference: bool = True,
                                                num_processes: Optional[int] = None) -> list[dict]:
        """
        Run probabilistic inference and return raw sampled suffixes per prefix.

        If cache_path is provided, samples are persisted as .pkl.
        If reuse_cache=True and cache exists, cached samples are loaded.

        Output:
        - case_id, prefix_len, prefix, target_suffix, decoded_suffixes, mode = "probabilistic"
        """
        if cache_path is not None and reuse_cache:
            try:
                with open(cache_path, "rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, list):
                    return cached
            except FileNotFoundError:
                pass

        decoder = self._build_decoder("probabilistic")
        case_ids = list(decoder.cases.keys())
        if random_order:
            random.shuffle(case_ids)

        worker_count = max(1, int(num_processes or self.config.num_processes or 1))
        use_parallel = parallel_inference and worker_count > 1 and len(case_ids) > 1

        samples: list[dict] = []

        if not use_parallel:
            for case_id, prefix_len, prefix, target_suffix, decoded_suffixes in decoder.evaluate(random_order=random_order):
                samples.append({"case_id": case_id,
                                "prefix_len": int(prefix_len),
                                "prefix": prefix,
                                "target_suffix": target_suffix,
                                "decoded_suffixes": decoded_suffixes,
                                "mode": "probabilistic"})
        else:
            chunk_size = max(1, (len(case_ids) + worker_count - 1) // worker_count)
            case_chunks = [case_ids[i : i + chunk_size] for i in range(0, len(case_ids), chunk_size)]

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_probabilistic_worker,
                initargs=(self.model,
                          self.dataset,
                          self.config.concept_name,
                          self.config.eos_value,
                          int(self.config.probabilistic_samples or 0))
                ) as executor:
                futures = [executor.submit(_collect_probabilistic_case_chunk, case_chunk) for case_chunk in case_chunks]
                
                for future in tqdm(concurrent.futures.as_completed(futures),
                                   total=len(futures),
                                   desc="Probabilistic inference chunks"):
                    
                    chunk_rows = future.result()
                    samples.extend(chunk_rows)

        if cache_path is not None:
            with open(cache_path, "wb") as handle:
                pickle.dump(samples, handle)

        return samples

    @staticmethod
    def _score_probabilistic_row(row: dict, reduction: ProbReduction) -> dict:
        target_suffix = row.get("target_suffix", [])
        decoded_suffixes = row.get("decoded_suffixes", [])
        if len(decoded_suffixes) == 0:
            decoded_suffixes = [[]]

        sample_dls = [dls_score(target_suffix, seq) for seq in decoded_suffixes]
        dls_mean = float(sum(sample_dls) / len(sample_dls))
        dls_min = float(min(sample_dls))
        dls_best = float(max(sample_dls))
        dls_value = dls_mean if reduction == "mean" else dls_best

        return {"case_id": row.get("case_id"),
                "prefix_len": int(row.get("prefix_len", 0)),
                "dls": dls_value,
                "dls_mean": dls_mean,
                "dls_min": dls_min,
                "dls_best": dls_best,
                "dls_max": dls_best,
                "mode": "probabilistic"}

    def evaluate_probabilistic_from_samples(
        self,
        samples: list[dict],
        reduction: ProbReduction = "mean",
        parallel_scoring: bool = False,
        num_processes: Optional[int] = None) -> pd.DataFrame:
        """
        Evaluate DLS from previously collected probabilistic inference samples.

        This enables a two-step workflow:
        1) collect_probabilistic_inference_samples(..., cache_path='*.pkl')
        2) evaluate_probabilistic_from_samples(...)
        """
        if reduction not in ("mean", "best"):
            raise ValueError("Unsupported reduction. Use one of: 'mean', 'best'.")

        if not parallel_scoring:
            rows = [self._score_probabilistic_row(row, reduction) for row in samples]
            return pd.DataFrame(rows)

        max_workers = num_processes or self.config.num_processes or 1
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._score_probabilistic_row, row, reduction) for row in samples]
            rows = [future.result() for future in concurrent.futures.as_completed(futures)]

        return pd.DataFrame(rows)

    @staticmethod
    def dls_per_prefix_length(results_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate mean DLS by prefix length.
        """
        if len(results_df) == 0:
            return pd.DataFrame(columns=["prefix_len", "dls"])

        has_probabilistic_bounds = {"dls_min", "dls_max"}.issubset(results_df.columns)
        if has_probabilistic_bounds:
            grouped = (results_df.groupby("prefix_len", as_index=False).agg(dls=("dls", "mean"), dls_min=("dls_min", "mean"), dls_max=("dls_max", "mean")))
        else:
            grouped = results_df.groupby("prefix_len", as_index=False)["dls"].mean()

        return grouped.sort_values("prefix_len").reset_index(drop=True)

    @staticmethod
    def average_dls(results_df: pd.DataFrame) -> float:
        """
        Compute overall average DLS.
        """
        if len(results_df) == 0:
            return 0.0
        return float(results_df["dls"].mean())
