import numpy as np

from placepulse_cusp.evaluation.metrics import bootstrap_reversal_evidence


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
