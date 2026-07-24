from __future__ import annotations

import numpy as np


def log_score(probabilities: np.ndarray, choices: np.ndarray) -> np.ndarray:
    selected = probabilities[np.arange(len(choices)), choices]
    return np.log(np.clip(selected, 1e-12, 1.0))


def cross_entropy(probabilities: np.ndarray, choices: np.ndarray) -> float:
    return float(-log_score(probabilities, choices).mean())


def clustered_elpd_bootstrap(
    candidate_scores: np.ndarray,
    baseline_scores: np.ndarray,
    clusters: np.ndarray,
    *,
    repetitions: int = 200,
    seed: int = 1103,
    ci: float = 0.95,
) -> dict[str, float]:
    difference = candidate_scores - baseline_scores
    unique = np.unique(clusters)
    groups = {cluster: np.where(clusters == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([groups[x] for x in selected])
        samples.append(float(difference[indices].mean()))
    alpha = (1 - ci) / 2
    return {
        "mean": float(difference.mean()),
        "lower": float(np.quantile(samples, alpha)),
        "upper": float(np.quantile(samples, 1 - alpha)),
    }


def empirical_probabilities(choices: np.ndarray) -> np.ndarray:
    counts = np.bincount(choices, minlength=3).astype(float) + 1.0
    return counts / counts.sum()


def ranking_reversal_fraction(utilities: np.ndarray, max_pairs: int = 200000) -> float:
    if utilities.shape[0] < 2 or utilities.shape[1] < 2:
        return 0.0
    rng = np.random.default_rng(1103)
    n = utilities.shape[1]
    total_pairs = n * (n - 1) // 2
    count = min(total_pairs, max_pairs)
    left = rng.integers(0, n, count)
    right = rng.integers(0, n, count)
    valid = left != right
    left, right = left[valid], right[valid]
    signs = np.sign(utilities[:, left] - utilities[:, right])
    reversals = np.any(signs != signs[0:1], axis=0)
    return float(reversals.mean()) if len(reversals) else 0.0

