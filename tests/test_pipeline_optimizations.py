from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from placepulse_cusp import pipeline
from placepulse_cusp.config import load_config


def test_selection_disables_lbfgs_but_final_training_keeps_it():
    config = load_config("configs/smoke.yaml")
    assert pipeline._selection_training_kwargs(config)["lbfgs_steps"] == 0
    assert (
        pipeline._screening_training_kwargs(config)["epochs"]
        < pipeline._selection_training_kwargs(config)["epochs"]
    )
    assert (
        pipeline._training_kwargs(config)["lbfgs_steps"]
        == config["models"]["lbfgs_steps"]
    )


def test_scalar_selection_reports_boundary_parameters(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]
    encoded = SimpleNamespace(choice=torch.tensor([0]))
    monkeypatch.setattr(
        pipeline,
        "_encoded_inner_edge_splits",
        lambda *args, **kwargs: [(encoded, encoded)],
    )
    monkeypatch.setattr(pipeline, "_choice_array", lambda value: np.asarray([0]))
    monkeypatch.setattr(
        pipeline, "cross_entropy", lambda probabilities, choices: float(probabilities[0])
    )

    def fake_fit(*args, **kwargs):
        score = -float(kwargs["utility_l2"])
        if kwargs["response_styles"]:
            score -= float(kwargs["style_l2"]) + 1.0
        return SimpleNamespace(
            predict_proba=lambda value: np.asarray([score])
        )

    monkeypatch.setattr(pipeline.DavidsonModel, "fit", fake_fit)
    selected = pipeline._select_scalar_baseline(
        object(), config, torch.device("cpu")
    )

    assert selected["utility_l2"] == 10.0
    assert selected["style_l2"] == 10.0
    assert selected["selection_boundary"]
    assert selected["selection_boundary_parameters"] == [
        "utility_l2",
        "style_l2",
    ]


def test_scalar_selection_collapses_trivial_high_l2_styles(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]
    config["gates"]["min_cross_entropy_reduction"] = 0.005
    encoded = SimpleNamespace(choice=torch.tensor([0]))
    monkeypatch.setattr(
        pipeline,
        "_encoded_inner_edge_splits",
        lambda *args, **kwargs: [(encoded, encoded), (encoded, encoded)],
    )
    monkeypatch.setattr(pipeline, "_choice_array", lambda value: np.asarray([0]))
    monkeypatch.setattr(
        pipeline, "cross_entropy", lambda probabilities, choices: float(probabilities[0])
    )

    def fake_fit(*args, **kwargs):
        utility_l2 = float(kwargs["utility_l2"])
        base_score = 1.0 + abs(utility_l2 - 1.0) * 0.01
        if kwargs["response_styles"]:
            style_l2 = float(kwargs["style_l2"])
            improvement = 0.001 if style_l2 == 10.0 else 0.0005
            base_score -= improvement
        return SimpleNamespace(
            predict_proba=lambda value: np.asarray([base_score])
        )

    monkeypatch.setattr(pipeline.DavidsonModel, "fit", fake_fit)
    selected = pipeline._select_scalar_baseline(
        object(), config, torch.device("cpu")
    )

    assert selected["name"] == "m1a"
    assert not selected["response_styles"]
    assert selected["m1b_statistical_gate"]
    assert not selected["m1b_practical_gate"]
    assert selected["m1b_high_regularisation_collapse"]
    assert not selected["selection_boundary"]
    assert selected["selection_boundary_parameters"] == []


def test_scalar_selection_keeps_material_high_l2_style_boundary(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"]["l2_candidates"] = [0.1, 1.0, 10.0]
    config["gates"]["min_cross_entropy_reduction"] = 0.005
    encoded = SimpleNamespace(choice=torch.tensor([0]))
    monkeypatch.setattr(
        pipeline,
        "_encoded_inner_edge_splits",
        lambda *args, **kwargs: [(encoded, encoded), (encoded, encoded)],
    )
    monkeypatch.setattr(pipeline, "_choice_array", lambda value: np.asarray([0]))
    monkeypatch.setattr(
        pipeline, "cross_entropy", lambda probabilities, choices: float(probabilities[0])
    )

    def fake_fit(*args, **kwargs):
        utility_l2 = float(kwargs["utility_l2"])
        base_score = 1.0 + abs(utility_l2 - 1.0) * 0.01
        if kwargs["response_styles"]:
            style_l2 = float(kwargs["style_l2"])
            improvement = 0.01 if style_l2 == 10.0 else 0.005
            base_score -= improvement
        return SimpleNamespace(
            predict_proba=lambda value: np.asarray([base_score])
        )

    monkeypatch.setattr(pipeline.DavidsonModel, "fit", fake_fit)
    selected = pipeline._select_scalar_baseline(
        object(), config, torch.device("cpu")
    )

    assert selected["name"] == "m1b"
    assert selected["response_styles"]
    assert selected["m1b_statistical_gate"]
    assert selected["m1b_practical_gate"]
    assert not selected["m1b_high_regularisation_collapse"]
    assert selected["selection_boundary"]
    assert selected["selection_boundary_parameters"] == ["style_l2"]


def test_continuous_grid_uses_full_starts_only_for_shortlist(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"].update(
        {
            "continuous_dimensions": [1, 2],
            "l2_candidates": [0.1, 1.0, 10.0],
            "random_starts": 2,
        }
    )
    encoded = SimpleNamespace(choice=torch.tensor([0]))
    monkeypatch.setattr(
        pipeline,
        "_encoded_inner_edge_splits",
        lambda *args, **kwargs: [(encoded, encoded)],
    )
    monkeypatch.setattr(pipeline, "_choice_array", lambda value: np.asarray([0]))
    monkeypatch.setattr(
        pipeline, "cross_entropy", lambda probabilities, choices: float(probabilities[0])
    )
    calls = []

    def fake_fit(
        encoded_train,
        config,
        baseline,
        rank,
        l2,
        seed,
        *,
        selection=True,
        starts=None,
        screening=False,
    ):
        calls.append((rank, l2, starts, screening))
        return SimpleNamespace(
            predict_proba=lambda value: np.asarray([rank * 100.0 + l2])
        )

    monkeypatch.setattr(pipeline, "_best_continuous_fit", fake_fit)
    selected = pipeline._select_continuous(
        object(), config, torch.device("cpu"), {"utility_l2": 0.1}
    )

    screening = [call for call in calls if call[2] == 1 and call[3]]
    refinement = [call for call in calls if call[2] == 2]
    assert len(screening) == 6
    assert len(refinement) == 2
    assert selected == (1, 0.1)


def test_mixture_grid_uses_full_starts_only_for_shortlist(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"].update(
        {
            "mixture_classes": [2, 3],
            "l2_candidates": [0.1, 1.0, 10.0],
            "random_starts": 2,
        }
    )
    encoded = SimpleNamespace(choice=torch.tensor([0]))
    monkeypatch.setattr(
        pipeline,
        "_encoded_inner_edge_splits",
        lambda *args, **kwargs: [(encoded, encoded)],
    )
    monkeypatch.setattr(pipeline, "_choice_array", lambda value: np.asarray([0]))
    monkeypatch.setattr(
        pipeline, "cross_entropy", lambda probabilities, choices: float(probabilities[0])
    )
    calls = []

    def fake_fit(
        encoded_train,
        config,
        baseline,
        classes,
        l2,
        seed,
        *,
        selection=True,
        starts=None,
        screening=False,
    ):
        calls.append((classes, l2, starts, screening))
        return SimpleNamespace(
            predict_proba=lambda value: np.asarray([classes * 100.0 + l2])
        )

    monkeypatch.setattr(pipeline, "_best_mixture_fit", fake_fit)
    selected = pipeline._select_mixture(
        object(), config, torch.device("cpu"), {"utility_l2": 0.1}
    )

    screening = [call for call in calls if call[2] == 1 and call[3]]
    refinement = [call for call in calls if call[2] == 2]
    assert len(screening) == 6
    assert len(refinement) == 2
    assert selected == (2, 0.1)


def test_mixture_selection_can_return_candidate_diagnostics(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"].update(
        {
            "mixture_classes": [2, 3],
            "l2_candidates": [0.1, 1.0],
            "random_starts": 2,
        }
    )
    encoded = SimpleNamespace(choice=torch.tensor([0]))
    monkeypatch.setattr(
        pipeline,
        "_encoded_inner_edge_splits",
        lambda *args, **kwargs: [(encoded, encoded), (encoded, encoded)],
    )
    monkeypatch.setattr(pipeline, "_choice_array", lambda value: np.asarray([0]))
    monkeypatch.setattr(
        pipeline, "cross_entropy", lambda probabilities, choices: float(probabilities[0])
    )

    def fake_fit(
        encoded_train,
        config,
        baseline,
        classes,
        l2,
        seed,
        **kwargs,
    ):
        return SimpleNamespace(
            predict_proba=lambda value: np.asarray([classes + l2])
        )

    monkeypatch.setattr(pipeline, "_best_mixture_fit", fake_fit)
    classes, l2, diagnostics = pipeline._select_mixture(
        object(),
        config,
        torch.device("cpu"),
        {"utility_l2": 0.1},
        return_diagnostics=True,
    )

    assert (classes, l2) == (2, 0.1)
    assert diagnostics["selected_classes"] == 2
    assert diagnostics["selected_l2"] == 0.1
    assert len(diagnostics["screening"]) == 4
    assert len(diagnostics["refinement"]) == 2
    assert all(
        len(item["fold_cross_entropy"]) == 2
        for item in diagnostics["refinement"]
    )
    assert all(
        item["standard_error"] == 0.0
        for item in diagnostics["refinement"]
    )


def test_final_continuous_fit_polishes_only_best_start(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"].update({"random_starts": 3, "lbfgs_steps": 4})
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            history=[float(len(calls))],
            module=torch.nn.Linear(1, 1),
        )

    monkeypatch.setattr(pipeline.ContinuousPreferenceModel, "fit", fake_fit)
    pipeline._best_continuous_fit(
        object(),
        config,
        {
            "utility_l2": 0.1,
            "style_l2": 0.1,
            "response_styles": False,
        },
        1,
        0.1,
        1103,
        selection=False,
    )
    assert len(calls) == 4
    assert all(call["lbfgs_steps"] == 0 for call in calls[:3])
    assert calls[-1]["lbfgs_steps"] == 4
    assert calls[-1]["epochs"] == 0
    assert calls[-1]["initial_state"] is not None


def test_final_mixture_fit_polishes_only_best_start(monkeypatch):
    config = load_config("configs/smoke.yaml")
    config["models"].update({"random_starts": 3, "lbfgs_steps": 4})
    calls = []

    def fake_fit(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            history=[float(len(calls))],
            module=torch.nn.Linear(1, 1),
        )

    monkeypatch.setattr(pipeline.MixtureDavidsonModel, "fit", fake_fit)
    pipeline._best_mixture_fit(
        object(),
        config,
        {
            "utility_l2": 0.1,
            "style_l2": 0.1,
            "response_styles": False,
        },
        2,
        0.1,
        1103,
        selection=False,
    )
    assert len(calls) == 4
    assert all(call["lbfgs_steps"] == 0 for call in calls[:3])
    assert calls[-1]["lbfgs_steps"] == 4
    assert calls[-1]["epochs"] == 0
    assert calls[-1]["initial_state"] is not None
