from __future__ import annotations

from typing import Any


def heterogeneity_verdict(
    config: dict[str, Any],
    scalar_ce: float,
    continuous_ce: float,
    mixture_ce: float,
    continuous_ci: dict[str, float],
    mixture_ci: dict[str, float],
    class_weights: list[float],
    reversal_fraction: float,
    stability_ari: float = 1.0,
    time_direction_ok: bool = True,
    simulation_ok: bool = True,
) -> tuple[str, dict[str, Any]]:
    threshold = config["gates"]["min_cross_entropy_reduction"]

    def qualifies(ce: float, ci: dict[str, float]) -> bool:
        reduction = (scalar_ce - ce) / max(scalar_ce, 1e-12)
        return (
            reduction >= threshold
            and ci["lower"] > 0
            and time_direction_ok
            and simulation_ok
        )

    continuous_ok = qualifies(continuous_ce, continuous_ci)
    mixture_predictive = qualifies(mixture_ce, mixture_ci)
    stable_classes = (
        sum(weight >= config["gates"]["min_class_weight"] for weight in class_weights) >= 2
        and stability_ari >= config["gates"]["min_ari"]
        and reversal_fraction >= config["gates"]["min_reversal_fraction"]
    )
    if mixture_predictive and stable_classes:
        verdict = "SCALAR_REJECTED_MIXTURE"
    elif continuous_ok:
        verdict = "SCALAR_REJECTED_CONTINUOUS"
    else:
        verdict = "SCALAR_NOT_REJECTED"
    return verdict, {
        "continuous_predictive_gate": continuous_ok,
        "mixture_predictive_gate": mixture_predictive,
        "stable_class_gate": stable_classes,
        "continuous_reduction": (scalar_ce - continuous_ce) / max(scalar_ce, 1e-12),
        "mixture_reduction": (scalar_ce - mixture_ce) / max(scalar_ce, 1e-12),
    }

