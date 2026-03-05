import os
import pickle
import concurrent.futures
from dataclasses import dataclass
from typing import Literal, Optional

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"

import pandas as pd

from .inference import get_evaluator


DecodeMode = Literal["mean", "mode", "probabilistic", "beam"]
ProbReduction = Literal["mean", "best"]


@dataclass
class DLSConfig:
    concept_name: str = "Activity"
    eos_value: str = "EOS"
    probabilistic_samples: Optional[int] = 100
    beam_width: Optional[int] = 3
    num_processes: Optional[int] = 8


def damerau_levenshtein_distance(seq_a: list, seq_b: list) -> int:
    """Compute Damerau-Levenshtein edit distance between two token sequences."""
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
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
            if (
                i > 1
                and j > 1
                and seq_a[i - 1] == seq_b[j - 2]
                and seq_a[i - 2] == seq_b[j - 1]
            ):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[n][m]


def dls_score(target_activity_seq: list, predicted_activity_seq: list) -> float:
    """Compute Damerau-Levenshtein Similarity in [0, 1]."""
    denominator = max(len(target_activity_seq), len(predicted_activity_seq), 1)
    distance = damerau_levenshtein_distance(target_activity_seq, predicted_activity_seq)
    return 1.0 - (distance / denominator)


class DLSEvaluation:
    """Activity-only DLS evaluation for mean, mode, probabilistic and beam decoding."""

    def __init__(self, model, dataset, config: DLSConfig | None = None):
        self.model = model
        self.dataset = dataset
        self.config = config or DLSConfig()

    def _build_decoder(self, mode: DecodeMode):
        common = {
            "model": self.model,
            "dataset": self.dataset,
            "concept_name": self.config.concept_name,
            "eos_value": self.config.eos_value,
        }

        if mode == "mode":
            return get_evaluator(kind="mode", **common)

        if mode == "mean":
            return get_evaluator(
                kind="mcsa",
                samples_per_case=1,
                sample_argmax=True,
                use_variance_cat=False,
                variational_dropout_sampling=False,
                **common,
            )

        if mode == "probabilistic":
            return get_evaluator(
                kind="mcsa",
                samples_per_case=self.config.probabilistic_samples,
                sample_argmax=False,
                use_variance_cat=True,
                variational_dropout_sampling=False,
                **common,
            )

        if mode == "beam":
            return get_evaluator(
                kind="beam",
                beam_width=self.config.beam_width,
                **common,
            )

        raise ValueError("Unsupported mode. Use one of: 'mean', 'mode', 'probabilistic', 'beam'.")

    def evaluate(self, mode: DecodeMode, random_order: bool = False) -> pd.DataFrame:
        """
        Evaluate DLS for each (case, prefix length).

        Returns DataFrame columns:
            - case_id
            - prefix_len
            - dls
            - mode
        """
        if mode == "probabilistic":
            return self.evaluate_probabilistic(
                random_order=random_order,
                reduction="mean",
            )

        decoder = self._build_decoder(mode)

        rows: list[dict] = []
        for case_id, prefix_len, _prefix, target_suffix, decoded_suffixes in decoder.evaluate(
            random_order=random_order):
            prediction = decoded_suffixes[0] if len(decoded_suffixes) > 0 else []
            dls_value = float(dls_score(target_suffix, prediction))

            rows.append(
                {
                    "case_id": case_id,
                    "prefix_len": int(prefix_len),
                    "dls": dls_value,
                    "mode": mode,
                }
            )

        return pd.DataFrame(rows)

    def evaluate_probabilistic(
        self,
        random_order: bool = False,
        reduction: ProbReduction = "mean",
        parallel_scoring: bool = False,
        num_processes: Optional[int] = None,
        inference_cache_path: Optional[str] = None,
        reuse_inference_cache: bool = False,
    ) -> pd.DataFrame:
        """
        Evaluate probabilistic DLS using T sampled suffixes per prefix.

        reduction:
            - "mean": average DLS across samples
            - "best": best DLS across samples

        Returns DataFrame columns:
            - case_id
            - prefix_len
            - dls
            - dls_mean
            - dls_best
            - mode
        """
        if reduction not in ("mean", "best"):
            raise ValueError("Unsupported reduction. Use one of: 'mean', 'best'.")

        samples = self.collect_probabilistic_inference_samples(
            random_order=random_order,
            cache_path=inference_cache_path,
            reuse_cache=reuse_inference_cache,
        )

        return self.evaluate_probabilistic_from_samples(
            samples=samples,
            reduction=reduction,
            parallel_scoring=parallel_scoring,
            num_processes=num_processes,
        )

    def collect_probabilistic_inference_samples(
        self,
        random_order: bool = False,
        cache_path: Optional[str] = None,
        reuse_cache: bool = False) -> list[dict]:
        """
        Run probabilistic inference and return raw sampled suffixes per prefix.

        If cache_path is provided, samples are persisted as .pkl.
        If reuse_cache=True and cache exists, cached samples are loaded.
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
        samples: list[dict] = []
        for case_id, prefix_len, _prefix, target_suffix, decoded_suffixes in decoder.evaluate(
            random_order=random_order
        ):
            samples.append(
                {
                    "case_id": case_id,
                    "prefix_len": int(prefix_len),
                    "target_suffix": target_suffix,
                    "decoded_suffixes": decoded_suffixes,
                }
            )

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
        dls_best = float(max(sample_dls))
        dls_value = dls_mean if reduction == "mean" else dls_best

        return {
            "case_id": row.get("case_id"),
            "prefix_len": int(row.get("prefix_len", 0)),
            "dls": dls_value,
            "dls_mean": dls_mean,
            "dls_best": dls_best,
            "mode": "probabilistic",
        }

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
            futures = [
                executor.submit(self._score_probabilistic_row, row, reduction)
                for row in samples
            ]
            rows = [future.result() for future in concurrent.futures.as_completed(futures)]

        return pd.DataFrame(rows)

    @staticmethod
    def dls_per_prefix_length(results_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate mean DLS by prefix length."""
        if len(results_df) == 0:
            return pd.DataFrame(columns=["prefix_len", "dls"])
        return (
            results_df.groupby("prefix_len", as_index=False)["dls"]
            .mean()
            .sort_values("prefix_len")
            .reset_index(drop=True))

    @staticmethod
    def average_dls(results_df: pd.DataFrame) -> float:
        """Compute overall average DLS."""
        if len(results_df) == 0:
            return 0.0
        return float(results_df["dls"].mean())
