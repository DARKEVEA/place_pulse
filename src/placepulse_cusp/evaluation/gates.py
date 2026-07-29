from __future__ import annotations

from typing import Any


def edge_predictive_gates(
    config: dict[str, Any],
    m0_ce: float,
    scalar_ce: float,
    continuous_ce: float,
    mixture_ce: float,
    continuous_ci: dict[str, float],
    mixture_ci: dict[str, float],
    scalar_vs_m0_ci: dict[str, float],
    *,
    simulation_ok: bool = True,
    selection_boundary: bool = False,
) -> dict[str, bool | float]:
    """Evaluate gates that only depend on the primary edge holdout.

    Keeping these checks separate lets the pipeline avoid expensive stability
    and auxiliary-holdout refits for candidates that have already failed the
    primary predictive test.
    """
    threshold = config["gates"]["min_cross_entropy_reduction"]
    baseline_reduction = (m0_ce - scalar_ce) / max(m0_ce, 1e-12)
    baseline_ok = (
        baseline_reduction >= threshold
        and scalar_vs_m0_ci["lower"] > 0
        and not selection_boundary
        and simulation_ok
    )

    def qualifies(ce: float, ci: dict[str, float]) -> bool:
        reduction = (scalar_ce - ce) / max(scalar_ce, 1e-12)
        return reduction >= threshold and ci["lower"] > 0 and simulation_ok

    return {
        "baseline": baseline_ok,
        "continuous": qualifies(continuous_ce, continuous_ci),
        "mixture": qualifies(mixture_ce, mixture_ci),
        "baseline_reduction": baseline_reduction,
        "continuous_reduction": (scalar_ce - continuous_ce)
        / max(scalar_ce, 1e-12),
        "mixture_reduction": (scalar_ce - mixture_ce) / max(scalar_ce, 1e-12),
    }


def heterogeneity_verdict(
    config: dict[str, Any],
    m0_ce: float,
    scalar_ce: float,
    continuous_ce: float,
    mixture_ce: float,
    continuous_ci: dict[str, float],
    mixture_ci: dict[str, float],
    scalar_vs_m0_ci: dict[str, float],
    class_weights: list[float],
    reversal_fraction: float,
    stability_ari: float = 1.0,
    continuous_auxiliary_ok: bool = True,
    mixture_auxiliary_ok: bool = True,
    simulation_ok: bool = True,
    selection_boundary: bool = False,
) -> tuple[str, dict[str, Any]]:
    edge = edge_predictive_gates(
        config,
        m0_ce,
        scalar_ce,
        continuous_ce,
        mixture_ce,
        continuous_ci,
        mixture_ci,
        scalar_vs_m0_ci,
        simulation_ok=simulation_ok,
        selection_boundary=selection_boundary,
    )
    baseline_ok = bool(edge["baseline"])
    continuous_ok = bool(edge["continuous"]) and continuous_auxiliary_ok
    mixture_predictive = bool(edge["mixture"]) and mixture_auxiliary_ok
    stable_classes = (
        sum(weight >= config["gates"]["min_class_weight"] for weight in class_weights) >= 2
        and stability_ari >= config["gates"]["min_ari"]
        and reversal_fraction >= config["gates"]["min_reversal_fraction"]
    )
    mixture_preferred = mixture_predictive and stable_classes and (
        not continuous_ok or mixture_ce < continuous_ce
    )
    if not simulation_ok or selection_boundary:
        verdict = "MODEL_CALIBRATION_FAILED"
    elif not baseline_ok:
        verdict = "SCALAR_SIGNAL_NOT_ESTABLISHED"
    elif mixture_preferred:
        verdict = "SCALAR_REJECTED_MIXTURE"
    elif continuous_ok:
        verdict = "SCALAR_REJECTED_CONTINUOUS"
    else:
        verdict = "SCALAR_NOT_REJECTED"
    return verdict, {
        "continuous_predictive_gate": continuous_ok,
        "continuous_edge_predictive_gate": edge["continuous"],
        "baseline_predictive_gate": baseline_ok,
        "simulation_gate": simulation_ok,
        "selection_boundary_gate": not selection_boundary,
        "baseline_reduction": edge["baseline_reduction"],
        "mixture_predictive_gate": mixture_predictive,
        "mixture_edge_predictive_gate": edge["mixture"],
        "continuous_auxiliary_gate": continuous_auxiliary_ok,
        "mixture_auxiliary_gate": mixture_auxiliary_ok,
        "stable_class_gate": stable_classes,
        "mixture_preferred_to_continuous": mixture_preferred,
        "continuous_reduction": edge["continuous_reduction"],
        "mixture_reduction": edge["mixture_reduction"],
    }
