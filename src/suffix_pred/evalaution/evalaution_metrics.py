import pickle
import concurrent.futures
from typing import Literal, Optional

import pandas as pd

ProbReduction = Literal["mean", "best"]


def load_decoded_suffixes(cache_path: str) -> list[dict]:
    """Load decoded suffix outputs from a pickle file.

    Parameters
    ----------
    cache_path : str
        Path to a ``.pkl`` file produced by
        :class:`~decode_test_set_suffixes.TestSetSuffixDecoder`.

    Returns
    -------
    list[dict]
        Each dict contains: ``case_id``, ``prefix_len``, ``prefix``,
        ``target_suffix``, ``decoded_suffixes``, ``mode``.
    """
    with open(cache_path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a list of dicts in the pickle file.")
    return data


# ---------------------------------------------------------------------------
# Damerau-Levenshtein distance & similarity
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _score_deterministic_row(row: dict) -> dict:
    """Score a single deterministic (mode / mean / beam) row."""
    decoded_suffixes = row.get("decoded_suffixes", [])
    prediction = decoded_suffixes[0] if len(decoded_suffixes) > 0 else []
    dls_value = float(dls_score(row.get("target_suffix", []), prediction))
    return {
        "case_id": row.get("case_id"),
        "prefix_len": int(row.get("prefix_len", 0)),
        "dls": dls_value,
        "mode": row.get("mode", ""),
    }


def _score_probabilistic_row(row: dict, reduction: ProbReduction) -> dict:
    """Score a single probabilistic row over its sampled suffixes."""
    target_suffix = row.get("target_suffix", [])
    decoded_suffixes = row.get("decoded_suffixes", [])
    if len(decoded_suffixes) == 0:
        decoded_suffixes = [[]]

    sample_dls = [dls_score(target_suffix, seq) for seq in decoded_suffixes]
    dls_mean = float(sum(sample_dls) / len(sample_dls))
    dls_min = float(min(sample_dls))
    dls_best = float(max(sample_dls))
    dls_value = dls_mean if reduction == "mean" else dls_best

    return {
        "case_id": row.get("case_id"),
        "prefix_len": int(row.get("prefix_len", 0)),
        "dls": dls_value,
        "dls_mean": dls_mean,
        "dls_min": dls_min,
        "dls_best": dls_best,
        "dls_max": dls_best,
        "mode": "probabilistic",
    }


# ---------------------------------------------------------------------------
# Main evaluation entry points
# ---------------------------------------------------------------------------


def evaluate_dls(
    outputs: list[dict],
    probabilistic_reduction: ProbReduction = "mean",
    parallel_scoring: bool = False,
    num_processes: Optional[int] = None,
) -> pd.DataFrame:
    """Evaluate Damerau-Levenshtein Similarity from decoded suffix outputs.

    Parameters
    ----------
    outputs : list[dict]
        Decoded suffix dicts as produced by
        :meth:`~decode_test_set_suffixes.TestSetSuffixDecoder.decode`
        (or loaded via :func:`load_decoded_suffixes`).
    probabilistic_reduction : ProbReduction
        For probabilistic outputs, aggregate the per-sample DLS values
        using ``"mean"`` or ``"best"`` (max).
    parallel_scoring : bool
        Use multi-process scoring (useful for large probabilistic sets).
    num_processes : int, optional
        Number of workers when *parallel_scoring* is enabled.

    Returns
    -------
    pd.DataFrame
        Columns: ``case_id``, ``prefix_len``, ``dls``, ``mode``.
        Probabilistic rows additionally include ``dls_mean``, ``dls_min``,
        ``dls_best``, ``dls_max``.
    """
    if len(outputs) == 0:
        return pd.DataFrame(columns=["case_id", "prefix_len", "dls", "mode"])

    mode = str(outputs[0].get("mode", "")).strip().lower()

    if mode == "probabilistic":
        return _evaluate_probabilistic(
            outputs,
            reduction=probabilistic_reduction,
            parallel_scoring=parallel_scoring,
            num_processes=num_processes,
        )

    rows = [_score_deterministic_row(row) for row in outputs]
    return pd.DataFrame(rows)


def _evaluate_probabilistic(
    samples: list[dict],
    reduction: ProbReduction = "mean",
    parallel_scoring: bool = False,
    num_processes: Optional[int] = None,
) -> pd.DataFrame:
    """Score probabilistic decoded outputs."""
    if reduction not in ("mean", "best"):
        raise ValueError("Unsupported reduction. Use one of: 'mean', 'best'.")

    if not parallel_scoring:
        rows = [_score_probabilistic_row(row, reduction) for row in samples]
        return pd.DataFrame(rows)

    max_workers = max(1, int(num_processes or 1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_score_probabilistic_row, row, reduction)
            for row in samples
        ]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Aggregation utilities
# ---------------------------------------------------------------------------


def dls_per_prefix_length(results_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean DLS by prefix length."""
    if len(results_df) == 0:
        return pd.DataFrame(columns=["prefix_len", "dls"])

    has_probabilistic_bounds = {"dls_min", "dls_max"}.issubset(results_df.columns)
    if has_probabilistic_bounds:
        grouped = results_df.groupby("prefix_len", as_index=False).agg(
            dls=("dls", "mean"),
            dls_min=("dls_min", "mean"),
            dls_max=("dls_max", "mean"),
        )
    else:
        grouped = results_df.groupby("prefix_len", as_index=False)["dls"].mean()

    return grouped.sort_values("prefix_len").reset_index(drop=True)


def average_dls(results_df: pd.DataFrame) -> float:
    """Compute overall average DLS."""
    if len(results_df) == 0:
        return 0.0
    return float(results_df["dls"].mean())
