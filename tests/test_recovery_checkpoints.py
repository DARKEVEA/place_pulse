from __future__ import annotations

import json

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


def test_mixture_pilot_configuration():
    config = load_config("configs/calibration_mixture_pilot_cuda.yaml")

    assert config["simulation"]["model_mechanisms"] == ["mixture"]
    assert config["simulation"]["model_repetitions"] == 1
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_008_mixture_calibration_pilot"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_008_MIXTURE_CALIBRATION_PILOT"
    )


def test_mixture_aggregation_pilot_configuration():
    config = load_config(
        "configs/calibration_mixture_aggregation_pilot_cuda.yaml"
    )

    assert config["simulation"]["model_mechanisms"] == ["mixture"]
    assert config["simulation"]["model_repetitions"] == 1
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_009_mixture_aggregation_pilot"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_009_MIXTURE_AGGREGATION_PILOT"
    )


def test_multiseed_model_screening_configuration():
    config = load_config(
        "configs/calibration_multiseed_screening_cuda.yaml"
    )

    assert config["simulation"]["model_mechanisms"] == [
        "null",
        "scalar",
        "continuous",
        "mixture",
    ]
    assert config["simulation"]["model_repetitions"] == 5
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_010_multiseed_model_screening"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_010_MULTISEED_MODEL_SCREENING"
    )


def test_density_pilot_configuration():
    config = load_config("configs/calibration_density_pilot_cuda.yaml")

    assert config["simulation"]["repetitions"] == 1
    assert config["cusp"]["quadrature_points"] == 160
    assert config["cusp"]["max_iterations"] == 1000
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_011_density_calibration_pilot"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_011_DENSITY_CALIBRATION_PILOT"
    )


def test_density_multiseed_configuration():
    config = load_config("configs/calibration_density_multiseed_cuda.yaml")

    assert config["simulation"]["repetitions"] == 5
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_012_density_multiseed_screening"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_012_DENSITY_MULTISEED_SCREENING"
    )


def test_density_diagnostics_configuration():
    config = load_config("configs/calibration_density_diagnostics_cuda.yaml")

    assert config["simulation"]["repetitions"] == 1
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_013_density_diagnostics"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_013_DENSITY_DIAGNOSTICS"
    )


def test_density_confirmatory_configuration():
    config = load_config("configs/calibration_density_confirmatory_cuda.yaml")

    assert config["simulation"]["repetitions"] == 100
    assert config["simulation"]["recovery_min_rate"] == 0.80
    assert config["simulation"]["cusp_max_mixture_false_positive"] == 0.10
    assert config["reporting"]["artifacts_dir"] == (
        "artifacts/run_014_density_confirmatory"
    )
    assert config["reporting"]["run_label"] == (
        "RUN_014_DENSITY_CONFIRMATORY"
    )


def test_recovery_selection_aggregation_uses_outer_fold_modes():
    def selection(fold, classes, mixture_l2, rank, continuous_l2):
        return {
            "fold": fold,
            "baseline": {
                "name": "m1a",
                "utility_l2": 0.1,
                "style_l2": 100.0,
                "response_styles": False,
                "selection_boundary": False,
                "selection_boundary_parameters": [],
            },
            "continuous_rank": rank,
            "continuous_l2": continuous_l2,
            "mixture_classes": classes,
            "mixture_l2": mixture_l2,
            "continuous_selection_boundary": False,
            "mixture_selection_boundary": False,
        }

    aggregated = recovery._aggregate_recovery_selections(
        [
            selection(0, 3, 0.1, 2, 1.0),
            selection(1, 4, 1.0, 3, 10.0),
            selection(2, 3, 0.1, 2, 1.0),
        ]
    )

    assert aggregated["baseline"]["name"] == "m1a"
    assert aggregated["continuous_rank"] == 2
    assert aggregated["continuous_l2"] == 1.0
    assert aggregated["mixture_classes"] == 3
    assert aggregated["mixture_l2"] == 0.1


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
    assert all(
        item["mixture_cusp_score"] == 0.0
        and item["mixture_reference_score"] == 0.0
        and item["mixture_cusp_margin"] == 0.0
        for item in first["details"]
    )

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


def test_mixture_effective_recovery_does_not_replace_strict_recovery():
    config = load_config("configs/smoke.yaml")
    item = {
        "verdict": "SCALAR_REJECTED_MIXTURE",
        "selected_classes": 4,
        "effective_classes": 3,
        "truth_ari": 0.99,
        "stability_ari": 0.99,
    }

    assessment = recovery._model_recovery_assessment(
        config, "mixture", "SCALAR_REJECTED_MIXTURE", item
    )

    assert not assessment["recovered"]
    assert not assessment["strict_recovered"]
    assert assessment["reason"] == "mixture_structure_not_recovered"
    assert assessment["effective_recovered"]
    assert assessment["effective_reason"] == (
        "effective_mixture_structure_recovered"
    )


def test_mixture_effective_recovery_still_requires_stability():
    config = load_config("configs/smoke.yaml")
    item = {
        "verdict": "SCALAR_REJECTED_MIXTURE",
        "selected_classes": 4,
        "effective_classes": 3,
        "truth_ari": 0.99,
        "stability_ari": 0.0,
    }

    assessment = recovery._model_recovery_assessment(
        config, "mixture", "SCALAR_REJECTED_MIXTURE", item
    )

    assert not assessment["strict_recovered"]
    assert not assessment["effective_recovered"]


def test_model_recovery_reports_strict_and_effective_rates(
    tmp_path, monkeypatch
):
    config = _checkpoint_config(tmp_path)
    config["simulation"]["model_repetitions"] = 1
    config["simulation"]["model_mechanisms"] = ["mixture"]

    monkeypatch.setattr(
        recovery,
        "_model_recovery_once",
        lambda *args, **kwargs: {
            "mechanism": "mixture",
            "verdict": "SCALAR_REJECTED_MIXTURE",
            "selected_classes": 4,
            "effective_classes": 3,
            "truth_ari": 0.99,
            "stability_ari": 0.99,
        },
    )

    result = recovery.validate_model_recovery(config)

    assert result["status"] == "failed"
    assert result["effective_status"] == "ok"
    assert result["recovery_rates"] == {"mixture": 0.0}
    assert result["effective_recovery_rates"] == {"mixture": 1.0}


def _complex_boundary_item(mechanism, *, boundary_value, baseline_reduction):
    return {
        "mechanism": mechanism,
        "verdict": "MODEL_CALIBRATION_FAILED",
        "gates": {
            "simulation_gate": True,
            "baseline_predictive_gate": False,
            "continuous_edge_predictive_gate": False,
            "mixture_edge_predictive_gate": False,
            "baseline_reduction": baseline_reduction,
            "continuous_reduction": 0.0,
            "mixture_reduction": -0.001,
        },
        "baseline_selection": {
            "utility_l2": 100.0 if mechanism == "null" else 0.1,
            "style_l2": 100.0,
            "response_styles": False,
            "selection_boundary": True,
            "selection_boundary_parameters": (
                ["utility_l2", "continuous_l2", "mixture_l2"]
                if mechanism == "null"
                else ["continuous_l2", "mixture_l2"]
            ),
        },
        "selected_continuous_l2": boundary_value,
        "selected_mixture_l2": boundary_value,
        "outer_fold_selections": [
            {
                "baseline": {
                    "utility_l2": 100.0 if mechanism == "null" else 0.1,
                    "selection_boundary_parameters": (
                        ["utility_l2"] if mechanism == "null" else []
                    ),
                },
                "continuous_l2": boundary_value,
                "continuous_selection_boundary": True,
                "mixture_l2": boundary_value,
                "mixture_selection_boundary": True,
            }
        ],
    }


def test_null_effective_recovery_accepts_complex_upper_shrinkage():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 10.0, 100.0]
    item = _complex_boundary_item("null", boundary_value=100.0, baseline_reduction=0.0)

    assessment = recovery._model_recovery_assessment(
        config, "null", "SCALAR_SIGNAL_NOT_ESTABLISHED", item
    )

    assert not assessment["strict_recovered"]
    assert assessment["effective_recovered"]
    assert assessment["effective_reason"] == "null_high_regularisation_shrinkage"


def test_scalar_effective_recovery_accepts_complex_upper_shrinkage():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 10.0, 100.0]
    item = _complex_boundary_item(
        "scalar", boundary_value=100.0, baseline_reduction=0.1
    )

    assessment = recovery._model_recovery_assessment(
        config, "scalar", "SCALAR_NOT_REJECTED", item
    )

    assert not assessment["strict_recovered"]
    assert assessment["effective_recovered"]
    assert assessment["assessment_evidence"] == (
        "legacy_baseline_reduction_without_stored_ci"
    )


def test_effective_recovery_rejects_complex_lower_boundary():
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 10.0, 100.0]
    item = _complex_boundary_item("scalar", boundary_value=0.1, baseline_reduction=0.1)

    assessment = recovery._model_recovery_assessment(
        config, "scalar", "SCALAR_NOT_REJECTED", item
    )

    assert not assessment["effective_recovered"]


def test_reassessment_preserves_raw_result_and_writes_separate_file(tmp_path):
    config = _checkpoint_config(tmp_path)
    config["models"]["l2_candidates"] = [0.1, 10.0, 100.0]
    source = tmp_path / "artifacts" / "metrics" / "model_recovery.json"
    source.parent.mkdir(parents=True)
    item = _complex_boundary_item("null", boundary_value=100.0, baseline_reduction=0.0)
    item["repetition"] = 0
    source.write_text(
        json.dumps(
            {
                "mechanisms": ["null"],
                "repetitions": 1,
                "details": [item],
            }
        ),
        encoding="utf-8",
    )
    before = source.read_bytes()

    result = recovery.reassess_model_recovery(config)

    assert source.read_bytes() == before
    assert result["status"] == "failed"
    assert result["effective_status"] == "ok"
    assert result["effective_recovery_rates"] == {"null": 1.0}
    assert result["assessment_provenance"]["raw_result_preserved"]
    assert (
        source.parent / "model_recovery_reassessed.json"
    ).exists()
