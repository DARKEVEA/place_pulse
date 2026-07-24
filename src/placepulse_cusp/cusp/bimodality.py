from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from diptest import diptest
from sklearn.mixture import GaussianMixture
from statsmodels.stats.multitest import multipletests


@dataclass
class WindowResult:
    beta_bin: int
    n: int
    dip: float
    p_value: float
    q_value: float
    component_weights: list[float]
    ashman_d: float
    passes: bool


def conditional_bimodality(
    y: np.ndarray,
    beta: np.ndarray,
    *,
    bins: int = 10,
    min_neighbors: int = 200,
    fdr: float = 0.05,
    min_component_weight: float = 0.10,
    min_ashman_d: float = 2.0,
    min_adjacent_windows: int = 3,
    seed: int = 1103,
) -> tuple[list[WindowResult], bool]:
    edges = np.unique(np.quantile(beta, np.linspace(0, 1, bins + 1)))
    provisional = []
    for index in range(max(len(edges) - 1, 0)):
        upper_closed = index == len(edges) - 2
        mask = (beta >= edges[index]) & (
            (beta <= edges[index + 1]) if upper_closed else (beta < edges[index + 1])
        )
        values = y[mask]
        if len(values) < min_neighbors or np.std(values) < 1e-8:
            continue
        dip, p_value = diptest(values)
        mixture = GaussianMixture(2, random_state=seed, n_init=10).fit(values[:, None])
        means = mixture.means_.ravel()
        variances = mixture.covariances_.reshape(-1)
        ashman = float(
            np.sqrt(2) * abs(means[0] - means[1]) / np.sqrt(variances[0] + variances[1])
        )
        provisional.append(
            {
                "beta_bin": index,
                "n": len(values),
                "dip": float(dip),
                "p_value": float(p_value),
                "weights": mixture.weights_.tolist(),
                "ashman": ashman,
            }
        )
    if not provisional:
        return [], False
    q_values = multipletests([x["p_value"] for x in provisional], alpha=fdr, method="fdr_bh")[1]
    results = []
    for item, q_value in zip(provisional, q_values, strict=True):
        passes = (
            q_value < fdr
            and min(item["weights"]) >= min_component_weight
            and item["ashman"] >= min_ashman_d
        )
        results.append(
            WindowResult(
                item["beta_bin"],
                item["n"],
                item["dip"],
                item["p_value"],
                float(q_value),
                item["weights"],
                item["ashman"],
                passes,
            )
        )
    passing = sorted(result.beta_bin for result in results if result.passes)
    longest = current = 0
    previous = None
    for value in passing:
        current = current + 1 if previous is not None and value == previous + 1 else 1
        longest = max(longest, current)
        previous = value
    return results, longest >= min_adjacent_windows

