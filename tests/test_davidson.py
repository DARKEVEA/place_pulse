import numpy as np
import torch

from placepulse_cusp.models.base import EncodedVotes, davidson_logits
from placepulse_cusp.models.continuous import ContinuousPreferenceModel
from placepulse_cusp.models.davidson import DavidsonModel, _DavidsonModule
from placepulse_cusp.models.mixture import MixtureDavidsonModel


def test_probabilities_sum_to_one():
    delta = torch.tensor([-2.0, 0.0, 2.0])
    probability = torch.softmax(davidson_logits(delta, torch.tensor(-1.0)), -1)
    np.testing.assert_allclose(probability.sum(-1).numpy(), 1.0, atol=1e-7)


def test_left_right_swap_is_symmetric():
    delta = torch.tensor([1.4])
    original = torch.softmax(davidson_logits(delta, torch.tensor(-0.8)), -1)
    swapped = torch.softmax(davidson_logits(-delta, torch.tensor(-0.8)), -1)
    np.testing.assert_allclose(original[:, 0], swapped[:, 1], atol=1e-7)
    np.testing.assert_allclose(original[:, 2], swapped[:, 2], atol=1e-7)


def test_unknown_voter_prediction_integrates_response_styles():
    module = _DavidsonModule(2, 2)
    with torch.no_grad():
        module.utility[:] = torch.tensor([0.5, -0.5])
        module.left_bias[:] = torch.tensor([-2.0, 2.0])
        module.voter_tie[:] = torch.tensor([-1.0, 1.0])
    model = DavidsonModel(module, 0.0, 0.0, [])
    data = EncodedVotes(
        left=torch.tensor([0]),
        right=torch.tensor([1]),
        voter=torch.tensor([-1]),
        choice=torch.tensor([0]),
        image_ids=["a", "b"],
        voter_ids=[],
        vote_ids=["vote"],
    )
    probability = model.predict_proba(data, new_user_samples=2)
    assert probability.shape == (1, 3)
    np.testing.assert_allclose(probability.sum(), 1.0, atol=1e-7)


def test_all_models_honor_batched_training_and_lbfgs():
    data = EncodedVotes(
        left=torch.tensor([0, 1, 0, 1]),
        right=torch.tensor([1, 0, 1, 0]),
        voter=torch.tensor([0, 0, 1, 1]),
        choice=torch.tensor([0, 1, 2, 0]),
        image_ids=["a", "b"],
        voter_ids=["v0", "v1"],
        vote_ids=["0", "1", "2", "3"],
    )
    scalar = DavidsonModel.fit(
        data,
        response_styles=False,
        epochs=2,
        batch_size=2,
        lbfgs_steps=1,
    )
    continuous = ContinuousPreferenceModel.fit(
        data,
        rank=1,
        epochs=2,
        batch_size=2,
        lbfgs_steps=1,
    )
    mixture = MixtureDavidsonModel.fit(
        data,
        classes=2,
        epochs=2,
        batch_size=2,
        lbfgs_steps=1,
    )
    for model in (scalar, continuous, mixture):
        probability = model.predict_proba(data)
        np.testing.assert_allclose(probability.sum(1), 1.0, atol=1e-6)
