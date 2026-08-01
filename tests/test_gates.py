from placepulse_cusp.config import load_config
from placepulse_cusp.evaluation.gates import (
    edge_predictive_gates,
    heterogeneity_verdict,
    selection_boundary_is_relevant,
)


def _verdict(*, m0=1.0, scalar=0.9, simulation=True, boundary=False):
    config = load_config("configs/smoke.yaml")
    return heterogeneity_verdict(
        config,
        m0,
        scalar,
        1.1,
        1.1,
        {"mean": -0.2, "lower": -0.3, "upper": -0.1},
        {"mean": -0.2, "lower": -0.3, "upper": -0.1},
        {"mean": m0 - scalar, "lower": m0 - scalar, "upper": m0 - scalar},
        [0.5, 0.5],
        0.2,
        0.8,
        True,
        True,
        simulation_ok=simulation,
        selection_boundary=boundary,
    )


def test_baseline_failure_defers_scalar_claim():
    verdict, gates = _verdict(m0=0.9, scalar=1.0)
    assert verdict == "SCALAR_SIGNAL_NOT_ESTABLISHED"
    assert not gates["baseline_predictive_gate"]


def test_simulation_or_search_boundary_fails_calibration():
    assert _verdict(simulation=False)[0] == "MODEL_CALIBRATION_FAILED"
    assert _verdict(boundary=True)[0] == "MODEL_CALIBRATION_FAILED"


def test_edge_gates_can_be_checked_before_auxiliary_refits():
    config = load_config("configs/smoke.yaml")
    gates = edge_predictive_gates(
        config,
        1.0,
        0.9,
        0.8,
        1.0,
        {"mean": 0.1, "lower": 0.01, "upper": 0.2},
        {"mean": -0.1, "lower": -0.2, "upper": 0.0},
        {"mean": 0.1, "lower": 0.01, "upper": 0.2},
    )
    assert gates["baseline"]
    assert gates["baseline_edge"]
    assert gates["continuous"]
    assert not gates["mixture"]


def test_baseline_edge_gate_survives_unrelated_search_boundary():
    verdict, gates = _verdict(boundary=True)

    assert verdict == "MODEL_CALIBRATION_FAILED"
    assert not gates["baseline_predictive_gate"]
    assert gates["baseline_edge_predictive_gate"]


def _boundary_selection(value):
    return {
        "baseline": {
            "utility_l2": value,
            "style_l2": value,
            "selection_boundary": True,
            "selection_boundary_parameters": ["utility_l2"],
        },
        "continuous_l2": value,
        "continuous_selection_boundary": True,
        "mixture_l2": value,
        "mixture_selection_boundary": True,
    }


def test_unsupported_upper_boundaries_are_safe_shrinkage():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]

    relevant = selection_boundary_is_relevant(
        config,
        [_boundary_selection(10.0)],
        {"baseline_edge": False, "continuous": False, "mixture": False},
    )

    assert not relevant


def test_supported_upper_boundary_remains_relevant():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]

    relevant = selection_boundary_is_relevant(
        config,
        [_boundary_selection(10.0)],
        {"baseline_edge": False, "continuous": True, "mixture": False},
    )

    assert relevant


def test_lower_boundary_is_always_relevant():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]

    relevant = selection_boundary_is_relevant(
        config,
        [_boundary_selection(0.1)],
        {"baseline_edge": False, "continuous": False, "mixture": False},
    )

    assert relevant
