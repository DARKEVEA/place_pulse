from __future__ import annotations

from typing import Any

import numpy as np

from placepulse_cusp.cusp.density import (
    CuspDensity,
    LinearGaussianDensity,
    MixtureExpertDensity,
    SplineGaussianDensity,
)


def compare_density_models(
    config: dict[str, Any],
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    train_weight: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, object]]:
    cusp_cfg = config["cusp"]
    models: dict[str, object] = {
        "linear": LinearGaussianDensity(),
        "gam": SplineGaussianDensity(),
        "mixture_expert": MixtureExpertDensity(),
        "cusp": CuspDensity(
            quadrature_points=cusp_cfg["quadrature_points"],
            domain=tuple(cusp_cfg["domain"]),
            max_iterations=cusp_cfg["max_iterations"],
        ),
    }
    metrics = {}
    for name, model in models.items():
        model.fit(train_x, train_y, train_weight)
        scores = model.logpdf(test_x, test_y)
        metrics[name] = {
            "mean_log_density": float(np.mean(scores)),
            "cross_entropy": float(-np.mean(scores)),
            "scores": scores.tolist(),
            "parameters": model.parameters(),
        }
    cusp = models["cusp"]
    competitors = [metrics[name]["mean_log_density"] for name in ("linear", "gam", "mixture_expert")]
    best_other = max(competitors)
    improvement = metrics["cusp"]["mean_log_density"] - best_other
    metrics["comparison"] = {
        "best_non_cusp_log_density": best_other,
        "cusp_improvement": improvement,
        "relative_cross_entropy_reduction": improvement
        / max(abs(best_other), 1e-12),
        "fold_fraction": float(cusp.fold_mask(test_x).mean()),
        "integration_error": cusp.integration_error_,
    }
    return metrics, models

