import pickle
import concurrent.futures
from typing import Literal, Optional
import pandas as pd

ProbReduction = Literal["mean", "best"]

def load_decoded_suffixes(cache_path: str) -> list[dict]:
    """
    Load decoded suffix outputs from a pickle file.

    Inputs:
    - cache_path : str
    - Path to a .pkl file produced by :class: decode_test_set_suffixes.TestSetSuffixDecoder.

    Outputs:
    - list[dict]: Each dict contains: case_id, prefix_len, prefix, target_suffix, decoded_suffixes, mode.
    """
    with open(cache_path, "rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, list):
        raise ValueError("Expected a list of dicts in the pickle file.")
    return data


# Damerau-Levenshtein distance & similarity
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
            dp[i][j] = min( dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
            
            if (i > 1 and j > 1 and seq_a[i - 1] == seq_b[j - 2] and seq_a[i - 2] == seq_b[j - 1]):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[n][m]


def dls_score(target_activity_seq: list, predicted_activity_seq: list) -> float:
    """
    Compute Damerau-Levenshtein Similarity in [0, 1].
    """
    denominator = max(len(target_activity_seq), len(predicted_activity_seq), 1)
    distance = damerau_levenshtein_distance(target_activity_seq, predicted_activity_seq)
    return 1.0 - (distance / denominator)

# Scoring helpers
def _score_deterministic_row(row: dict) -> dict:
    """
    Score a single deterministic (mode or beam) row.
    """
    decoded_suffixes = row.get("decoded_suffixes", [])
    prediction = decoded_suffixes[0] if len(decoded_suffixes) > 0 else []
    dls_value = float(dls_score(row.get("target_suffix", []), prediction))
    
    return {"case_id": row.get("case_id"),
            "prefix_len": int(row.get("prefix_len", 0)),
            "dls": dls_value,
            "mode": row.get("mode", "")}


def _score_probabilistic_row(row: dict, reduction: ProbReduction) -> dict:
    """
    Score a single probabilistic row over its sampled suffixes.
    """
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


# Main evaluation entry points
def evaluate_dls(outputs: list[dict], 
                 probabilistic_reduction: ProbReduction = "mean",
                 parallel_scoring: bool = False,
                 num_processes: Optional[int] = None,) -> pd.DataFrame:
    """
    Evaluate Damerau-Levenshtein Similarity from decoded suffix outputs.

    Inputs:
    - outputs : list[dict] Decoded suffix dicts as produced by :meth:`decode_test_set_suffixes.TestSetSuffixDecoder.decode (or loaded via :func:load_decoded_suffixes).
    - probabilistic_reduction: mean
    - parallel_scoring : bool
    - Use multi-process scoring (useful for large probabilistic sets).
    - num_processes : int, optional
    - Number of workers when parallel_scoring is enabled.

    Outputs:
    pd.DataFrame: Columns: case_id, prefix_len, dls, mode. Probabilistic rows additionally include: dls_mean, dls_min, dls_best, dls_max.)
    """
    if len(outputs) == 0:
        return pd.DataFrame(columns=["case_id", "prefix_len", "dls", "mode"])

    mode = str(outputs[0].get("mode", "")).strip().lower()

    if "probabilistic" in mode:
        return _evaluate_probabilistic(outputs,
                                       reduction=probabilistic_reduction,
                                       parallel_scoring=parallel_scoring,
                                       num_processes=num_processes,
                                      )

    rows = [_score_deterministic_row(row) for row in outputs]
    return pd.DataFrame(rows)

def _evaluate_probabilistic(samples: list[dict],
                            reduction: ProbReduction = "mean",
                            parallel_scoring: bool = False,
                            num_processes: Optional[int] = None,) -> pd.DataFrame:
    """
    Score probabilistic decoded outputs.
    """
    if reduction not in ("mean", "best"):
        raise ValueError("Unsupported reduction. Use one of: 'mean', 'best'.")

    if not parallel_scoring:
        rows = [_score_probabilistic_row(row, reduction) for row in samples]
        return pd.DataFrame(rows)

    max_workers = max(1, int(num_processes or 1))
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_score_probabilistic_row, row, reduction)for row in samples]
        rows = [future.result() for future in concurrent.futures.as_completed(futures)]

    return pd.DataFrame(rows)


# Aggregation utilities
def dls_per_prefix_length(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate mean DLS by prefix length.
    """
    if len(results_df) == 0:
        return pd.DataFrame(columns=["prefix_len", "dls"])

    has_probabilistic_bounds = {"dls_min", "dls_max"}.issubset(results_df.columns)
    if has_probabilistic_bounds:
        grouped = results_df.groupby("prefix_len", as_index=False).agg(dls=("dls", "mean"),
                                                                       dls_min=("dls_min", "mean"),
                                                                       dls_max=("dls_max", "mean"),
                                                                       )
    else:
        grouped = results_df.groupby("prefix_len", as_index=False)["dls"].mean()

    return grouped.sort_values("prefix_len").reset_index(drop=True)


def average_dls(results_df: pd.DataFrame) -> float:
    """
    Compute overall average DLS.
    """
    if len(results_df) == 0:
        return 0.0
    return float(results_df["dls"].mean())


def exact_match_score(target_seq: list, predicted_seq: list) -> float:
    """1.0 iff the predicted activity suffix equals the target suffix."""
    return 1.0 if list(target_seq) == list(predicted_seq) else 0.0


def token_level_accuracy(target_seq: list, predicted_seq: list) -> float:
    """Positional token agreement; denominator = max length (penalises length errors)."""
    denom = max(len(target_seq), len(predicted_seq), 1)
    correct = sum(1 for a, b in zip(target_seq, predicted_seq) if a == b)
    return correct / denom


def _coherence_row(row: dict, reduction: ProbReduction = "mean") -> dict:
    target = row.get("target_suffix", [])
    decoded = row.get("decoded_suffixes", []) or [[]]

    if "probabilistic" in str(row.get("mode", "")).lower():
        ems = [exact_match_score(target, seq) for seq in decoded]
        tas = [token_level_accuracy(target, seq) for seq in decoded]
        if reduction == "best":
            em, ta = max(ems), max(tas)
        else:
            em, ta = sum(ems) / len(ems), sum(tas) / len(tas)
    else:
        pred = decoded[0] if len(decoded) > 0 else []
        em = exact_match_score(target, pred)
        ta = token_level_accuracy(target, pred)

    return {"case_id": row.get("case_id"),
            "prefix_len": int(row.get("prefix_len", 0)),
            "exact_match": float(em),
            "token_acc": float(ta),
            "mode": row.get("mode", "")}


def evaluate_coherence(outputs: list[dict],
                       probabilistic_reduction: ProbReduction = "mean") -> pd.DataFrame:
    """
    Coherent (exact-match) and incoherent (token-level) accuracy per row.

    Columns: case_id, prefix_len, exact_match, token_acc, mode.
    """
    if len(outputs) == 0:
        return pd.DataFrame(columns=["case_id", "prefix_len", "exact_match", "token_acc", "mode"])
    return pd.DataFrame([_coherence_row(row, probabilistic_reduction) for row in outputs])


def average_coherence(coherence_df: pd.DataFrame) -> dict:
    """Overall coherent (exact_match) and incoherent (token_acc) accuracy."""
    if len(coherence_df) == 0:
        return {"exact_match": 0.0, "token_acc": 0.0}
    return {"exact_match": float(coherence_df["exact_match"].mean()),
            "token_acc": float(coherence_df["token_acc"].mean())}


def decision_conformance_rate(decision_steps: int, conflicts: int) -> float:
    """conformance = 1 - conflicts / decision_steps; NaN when no decision steps."""
    ds = int(decision_steps)
    if ds <= 0:
        return float("nan")
    return 1.0 - float(conflicts) / float(ds)


def aggregate_decision_conformance(reasoning_rows: list[dict]) -> dict:
    """
    Aggregate decision conformance over per-case reasoning rows.

    Each row may carry a ``reasonings`` list of per-decode dicts (the format
    emitted by experiments.evaluation), or be a single reasoning dict. Both
    cases must expose ``decision_steps`` and ``conflicts`` counters.

    Returns: decision_steps, conflicts, conflict_rate, decision_conformance.
    """
    decision_steps = 0
    conflicts = 0
    for row in reasoning_rows:
        reasonings = row.get("reasonings") if isinstance(row, dict) else None
        if reasonings is None:
            reasonings = [row]
        for r in reasonings:
            decision_steps += int(r.get("decision_steps", 0))
            conflicts += int(r.get("conflicts", 0))

    conflict_rate = (conflicts / decision_steps) if decision_steps else 0.0
    return {"decision_steps": decision_steps,
            "conflicts": conflicts,
            "conflict_rate": float(conflict_rate),
            "decision_conformance": decision_conformance_rate(decision_steps, conflicts)}
