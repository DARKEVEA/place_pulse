from __future__ import annotations

import numpy as np

from placepulse_cusp.config import load_config
from placepulse_cusp.simulation import recovery


def _checkpoint_config(tmp_path):
    config = load_config("configs/smoke.yaml")
    config["reporting"]["artifacts_dir"] = str(tmp_path / "artifacts")
    config["simulation"]["model_repetitions"] = 2
    config["simulation"]["repetitions"] = 2
    return config


def test_null_pilot_configuration():
    config = load_config("configs/calibration_pilot_cuda.yaml")

    assert config["simulation"]["model_mechanisms"] == ["null"]
    assert config["simulation"]["model_repetitions"] == 1
    assert config["models"]["l2_candidates"][-1] == 100.0


def test_scalar_pilot_configuration():
    config = load_config("configs/calibration_scalar_pilot_cuda.yaml")

    assert config["simulation"]["model_mechanisms"] == ["scalar"]
    assert config["simulation"]["model_repetitions"] == 1
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_005_scalar_calibration_pilot"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_005_SCALAR_CALIBRATION_PILOT"
    )


def test_continuous_pilot_configuration():
    config = load_config("configs/calibration_continuous_pilot_cuda.yaml")

    assert config["simulation"]["model_mechanisms"] == ["continuous"]
    assert config["simulation"]["model_repetitions"] == 1
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_006_continuous_calibration_pilot"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_006_CONTINUOUS_CALIBRATION_PILOT"
    )


def test_continuous_rule_pilot_configuration():
    config = load_config(
        "configs/calibration_continuous_rule_pilot_cuda.yaml"
    )

    assert config["simulation"]["model_mechanisms"] == ["continuous"]
    assert config["simulation"]["model_repetitions"] == 1
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_007_continuous_rule_pilot"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_007_CONTINUOUS_RULE_PILOT"
    )


def test_model_recovery_resumes_completed_repetitions(tmp_path, monkeypatch):
    config = _checkpoint_config(tmp_path)
    calls = []
    expected = {
        "null": "SCALAR_SIGNAL_NOT_ESTABLISHED",
        "scalar": "SCALAR_NOT_REJECTED",
        "continuous": "SCALAR_REJECTED_CONTINUOUS",
        "mixture": "SCALAR_REJECTED_MIXTURE",
    }

    def fake_once(config, mechanism, seed):
        calls.append((mechanism, seed))
        return {
            "mechanism": mechanism,
            "verdict": expected[mechanism],
            "selected_rank": 2,
            "selected_classes": 3,
            "truth_ari": 1.0,
        }

    monkeypatch.setattr(recovery, "_model_recovery_once", fake_once)
    first = recovery.validate_model_recovery(config)
    assert len(calls) == 8
    assert first["status"] == "ok"

    monkeypatch.setattr(
        recovery,
        "_model_recovery_once",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed repetition was recomputed")
        ),
    )
    second = recovery.validate_model_recovery(config, resume=True)
    assert second["recovery_rates"] == first["recovery_rates"]
    progress = (
        tmp_path / "artifacts" / "metrics" / "model_recovery_progress.json"
    ).read_text("utf-8")
    assert '"status": "complete"' in progress
    assert '"completed": 8' in progress


def test_density_recovery_resumes_completed_repetitions(tmp_path, monkeypatch):
    config = _checkpoint_config(tmp_path)
    sample_calls = []

    def fake_sample(n, rng):
        sample_calls.append(n)
        x = np.zeros((n, 2))
        return x, np.zeros(n), np.ones(n)

    class FakeDensity:
        def fit(self, *args, **kwargs):
            return self

        def logpdf(self, x, y):
            return np.zeros(len(y))

    monkeypatch.setattr(recovery, "_cusp_sample", fake_sample)
    monkeypatch.setattr(recovery, "CuspDensity", lambda **kwargs: FakeDensity())
    monkeypatch.setattr(recovery, "MixtureExpertDensity", FakeDensity)
    first = recovery.validate_density_recovery(config)
    assert len(sample_calls) == 2

    monkeypatch.setattr(
        recovery,
        "_cusp_sample",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("completed density repetition was recomputed")
        ),
    )
    second = recovery.validate_density_recovery(config, resume=True)
    assert second["details"] == first["details"]


def test_model_recovery_can_run_selected_mechanisms(tmp_path, monkeypatch):
    config = _checkpoint_config(tmp_path)
    config["simulation"]["model_repetitions"] = 1
    config["simulation"]["model_mechanisms"] = ["null"]
    calls = []

    def fake_once(config, mechanism, seed):
        calls.append((mechanism, seed))
        return {
            "mechanism": mechanism,
            "verdict": "SCALAR_SIGNAL_NOT_ESTABLISHED",
            "selected_rank": 2,
            "selected_classes": 3,
            "truth_ari": 1.0,
        }

    monkeypatch.setattr(recovery, "_model_recovery_once", fake_once)
    result = recovery.validate_model_recovery(config)

    assert [mechanism for mechanism, _ in calls] == ["null"]
    assert result["status"] == "ok"
    assert result["mechanisms"] == ["null"]
    assert result["recovery_rates"] == {"null": 1.0}


def test_null_high_regularisation_boundary_is_recovered():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]
    upper = max(config["models"]["l2_candidates"])
    item = {
        "verdict": "MODEL_CALIBRATION_FAILED",
        "gates": {
            "baseline_predictive_gate": False,
            "continuous_edge_predictive_gate": False,
            "mixture_edge_predictive_gate": False,
            "baseline_reduction": 0.0,
            "continuous_reduction": 0.0,
            "mixture_reduction": -0.001,
        },
        "baseline_selection": {
            "utility_l2": upper,
            "style_l2": upper,
            "response_styles": False,
            "selection_boundary": True,
            "selection_boundary_parameters": ["utility_l2"],
        },
    }

    assessment = recovery._model_recovery_assessment(
        config, "null", "SCALAR_SIGNAL_NOT_ESTABLISHED", item
    )

    assert assessment["recovered"]
    assert assessment["reason"] == "null_high_regularisation_boundary"
    assert assessment["raw_verdict"] == "MODEL_CALIBRATION_FAILED"


def test_null_low_regularisation_boundary_is_not_recovered():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]
    lower = min(config["models"]["l2_candidates"])
    item = {
        "verdict": "MODEL_CALIBRATION_FAILED",
        "gates": {
            "baseline_predictive_gate": False,
            "continuous_edge_predictive_gate": False,
            "mixture_edge_predictive_gate": False,
            "baseline_reduction": 0.0,
            "continuous_reduction": 0.0,
            "mixture_reduction": 0.0,
        },
        "baseline_selection": {
            "utility_l2": lower,
            "style_l2": lower,
            "response_styles": False,
            "selection_boundary": True,
            "selection_boundary_parameters": ["utility_l2"],
        },
    }

    assessment = recovery._model_recovery_assessment(
        config, "null", "SCALAR_SIGNAL_NOT_ESTABLISHED", item
    )

    assert not assessment["recovered"]
    assert assessment["reason"] == "target_verdict_not_recovered"


def test_null_boundary_with_predictive_gain_is_not_recovered():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]
    upper = max(config["models"]["l2_candidates"])
    threshold = config["gates"]["min_cross_entropy_reduction"]
    item = {
        "verdict": "MODEL_CALIBRATION_FAILED",
        "gates": {
            "baseline_predictive_gate": False,
            "continuous_edge_predictive_gate": False,
            "mixture_edge_predictive_gate": False,
            "baseline_reduction": threshold + 0.001,
            "continuous_reduction": 0.0,
            "mixture_reduction": 0.0,
        },
        "baseline_selection": {
            "utility_l2": upper,
            "style_l2": upper,
            "response_styles": False,
            "selection_boundary": True,
            "selection_boundary_parameters": ["utility_l2"],
        },
    }

    assessment = recovery._model_recovery_assessment(
        config, "null", "SCALAR_SIGNAL_NOT_ESTABLISHED", item
    )

    assert not assessment["recovered"]
