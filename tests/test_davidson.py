import numpy as np
import torch

from placepulse_cusp.models.base import davidson_logits


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

