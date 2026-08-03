from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


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
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_sums = np.bincount(inverse, weights=difference)
    cluster_counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = rng.choice(unique, len(unique), replace=True)
        selected_indices = np.searchsorted(unique, selected)
        samples.append(
            float(
                cluster_sums[selected_indices].sum()
                / cluster_counts[selected_indices].sum()
            )
        )
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


def bootstrap_reversal_evidence(
    bootstrap_utilities: list[np.ndarray],
    image_counts: np.ndarray,
    *,
    min_image_votes: int = 20,
    min_probability: float = 0.90,
    min_standardized_gap: float = 0.5,
    max_pairs: int = 200000,
    seed: int = 1103,
) -> dict[str, float | int]:
    """Estimate reliable class ranking reversals across voter-bootstrap refits."""
    if len(bootstrap_utilities) < 2:
        return {"fraction": 0.0, "eligible_pairs": 0, "reliable_reversals": 0}
    reference = bootstrap_utilities[0]
    aligned = []
    for current in bootstrap_utilities:
        correlation = np.nan_to_num(
            np.corrcoef(current, reference)[: current.shape[0], current.shape[0] :]
        )
        rows, cols = linear_sum_assignment(-correlation)
        value = np.empty_like(current)
        value[cols] = current[rows]
        aligned.append(value)
    draws = np.stack(aligned)
    eligible_images = np.flatnonzero(image_counts >= min_image_votes)
    if len(eligible_images) < 2:
        return {"fraction": 0.0, "eligible_pairs": 0, "reliable_reversals": 0}
    rng = np.random.default_rng(seed)
    total = len(eligible_images) * (len(eligible_images) - 1) // 2
    count = min(total, max_pairs)
    if total <= max_pairs:
        pair_index = np.triu_indices(len(eligible_images), 1)
        left = eligible_images[pair_index[0]]
        right = eligible_images[pair_index[1]]
    else:
        left = rng.choice(eligible_images, count)
        right = rng.choice(eligible_images, count)
        equal = left == right
        while equal.any():
            right[equal] = rng.choice(eligible_images, int(equal.sum()))
            equal = left == right
    deltas = draws[:, :, left] - draws[:, :, right]
    shared_delta = deltas.mean(axis=1)
    direction_probability = np.maximum(
        (shared_delta > 0).mean(axis=0), (shared_delta < 0).mean(axis=0)
    )
    scale = max(float(draws.mean(axis=1).std()), 1e-12)
    gap = np.abs(shared_delta.mean(axis=0)) / scale
    informative = (direction_probability >= min_probability) & (
        gap >= min_standardized_gap
    )
    if not informative.any():
        return {"fraction": 0.0, "eligible_pairs": 0, "reliable_reversals": 0}
    signs = np.sign(deltas)
    reversed_draw = np.any(signs != signs[:, 0:1, :], axis=1)
    reversal_probability = reversed_draw.mean(axis=0)
    reliable = informative & (reversal_probability >= min_probability)
    eligible_count = int(informative.sum())
    reliable_count = int(reliable.sum())
    return {
        "fraction": reliable_count / eligible_count,
        "eligible_pairs": eligible_count,
        "reliable_reversals": reliable_count,
    }

