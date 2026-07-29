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
