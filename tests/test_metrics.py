import numpy as np

from placepulse_cusp.evaluation.metrics import (
    bootstrap_reversal_evidence,
    clustered_elpd_bootstrap,
)


def test_clustered_elpd_bootstrap_matches_explicit_cluster_resampling():
    candidate = np.asarray([1.0, 2.0, 4.0, 8.0, 16.0])
    baseline = np.asarray([0.5, 1.5, 2.0, 4.0, 8.0])
    clusters = np.asarray(["a", "a", "b", "c", "c"])
    repetitions = 20
    seed = 17
    unique = np.unique(clusters)
    difference = candidate - baseline
    groups = {cluster: np.where(clusters == cluster)[0] for cluster in unique}
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(repetitions):
        selected = rng.choice(unique, len(unique), replace=True)
        indices = np.concatenate([groups[value] for value in selected])
        samples.append(float(difference[indices].mean()))
    expected = {
        "mean": float(difference.mean()),
        "lower": float(np.quantile(samples, 0.025)),
        "upper": float(np.quantile(samples, 0.975)),
    }

    result = clustered_elpd_bootstrap(
        candidate,
        baseline,
        clusters,
        repetitions=repetitions,
        seed=seed,
    )

    assert result == expected


def test_bootstrap_reversal_requires_replicated_high_information_evidence():
    utilities = [
        np.asarray([[2.0, -2.0], [-1.0, 1.0]]) + index * 1e-3
        for index in range(10)
    ]
    result = bootstrap_reversal_evidence(
        utilities,
        np.asarray([100, 100]),
        min_probability=0.9,
        min_standardized_gap=0.0,
        max_pairs=100,
    )
    assert result["eligible_pairs"] > 0
    assert result["fraction"] == 1.0
