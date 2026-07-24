from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from placepulse_cusp.cusp.density import CuspDensity, MixtureExpertDensity
from placepulse_cusp.provenance import metadata, write_json


def _cusp_sample(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, 2))
    alpha = x[:, 0]
    beta = 1.5 + 1.2 * x[:, 1]
    grid = np.linspace(-4, 4, 1001)
    y = np.empty(n)
    for index in range(n):
        logp = -(grid**4 / 4 - beta[index] * grid**2 / 2 - alpha[index] * grid) / 0.5
        probability = np.exp(logp - logp.max())
        probability /= probability.sum()
        y[index] = rng.choice(grid, p=probability)
    return x, y, np.ones(n)


def validate_density_recovery(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(config["project"]["seed"])
    repetitions = config["simulation"]["repetitions"]
    cusp_wins, mixture_false_wins = 0, 0
    details = []
    n = min(max(config["simulation"]["votes"] // 10, 300), 2500)
    for repetition in range(repetitions):
        x, y, weight = _cusp_sample(n, rng)
        split = int(n * 0.8)
        cusp = CuspDensity(
            quadrature_points=config["cusp"]["quadrature_points"],
            domain=tuple(config["cusp"]["domain"]),
            max_iterations=config["cusp"]["max_iterations"],
        ).fit(x[:split], y[:split], weight[:split])
        mixture = MixtureExpertDensity().fit(x[:split], y[:split], weight[:split])
        cusp_score = float(cusp.logpdf(x[split:], y[split:]).mean())
        mixture_score = float(mixture.logpdf(x[split:], y[split:]).mean())
        cusp_wins += cusp_score > mixture_score

        means = np.where(rng.random(n) < 0.5, -1.5 + x[:, 0], 1.5 + x[:, 0])
        mixture_y = means + rng.normal(0, 0.5, n)
        cusp_m = CuspDensity(
            quadrature_points=config["cusp"]["quadrature_points"],
            domain=tuple(config["cusp"]["domain"]),
            max_iterations=config["cusp"]["max_iterations"],
        ).fit(x[:split], mixture_y[:split])
        moe_m = MixtureExpertDensity().fit(x[:split], mixture_y[:split])
        false_win = cusp_m.logpdf(x[split:], mixture_y[split:]).mean() > moe_m.logpdf(
            x[split:], mixture_y[split:]
        ).mean()
        mixture_false_wins += false_win
        details.append(
            {
                "repetition": repetition,
                "cusp_score": cusp_score,
                "mixture_score": mixture_score,
                "mixture_false_cusp_win": bool(false_win),
            }
        )
    recovery = cusp_wins / repetitions
    false_positive = mixture_false_wins / repetitions
    # Smoke configurations are diagnostic and intentionally too small for confirmatory thresholds.
    result = {
        "status": "ok"
        if recovery >= config["simulation"]["recovery_min_rate"]
        and false_positive <= config["simulation"]["cusp_max_mixture_false_positive"]
        else "failed",
        "cusp_recovery_rate": recovery,
        "mixture_false_cusp_rate": false_positive,
        "repetitions": repetitions,
        "details": details,
        "provenance": metadata(config),
    }
    target = Path(config["reporting"]["artifacts_dir"]) / "metrics" / "simulation_recovery.json"
    write_json(target, result)
    return result

