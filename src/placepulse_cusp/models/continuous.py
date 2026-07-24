from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from torch import nn
from torch.nn import functional as F

from placepulse_cusp.models.base import EncodedVotes, davidson_logits


class _ContinuousModule(nn.Module):
    def __init__(self, n_images: int, n_voters: int, rank: int):
        super().__init__()
        self.utility = nn.Parameter(torch.zeros(n_images))
        self.image_factor = nn.Parameter(torch.randn(n_images, rank) * 0.03)
        self.voter_factor = nn.Parameter(torch.randn(n_voters, rank) * 0.03)
        self.left_bias = nn.Parameter(torch.zeros(n_voters))
        self.voter_tie = nn.Parameter(torch.zeros(n_voters))
        self.log_tie = nn.Parameter(torch.tensor(-1.0))

    def logits(self, data: EncodedVotes) -> torch.Tensor:
        known = data.voter >= 0
        delta = (self.utility - self.utility.mean())[data.left] - (
            self.utility - self.utility.mean()
        )[data.right]
        tie = torch.zeros_like(delta)
        if known.any():
            voter_factor = self.voter_factor[data.voter[known]]
            contrast = self.image_factor[data.left[known]] - self.image_factor[data.right[known]]
            delta = delta.clone()
            delta[known] += (voter_factor * contrast).sum(-1) + self.left_bias[data.voter[known]]
            tie[known] = self.voter_tie[data.voter[known]]
        return davidson_logits(delta, self.log_tie, tie)


@dataclass
class ContinuousPreferenceModel:
    module: _ContinuousModule
    rank: int
    l2: float
    history: list[float]

    @classmethod
    def fit(
        cls,
        train: EncodedVotes,
        *,
        rank: int = 2,
        l2: float = 1e-3,
        epochs: int = 250,
        learning_rate: float = 0.03,
        patience: int = 30,
    ) -> "ContinuousPreferenceModel":
        module = _ContinuousModule(train.n_images, train.n_voters, rank).to(train.left.device)
        optimiser = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history, best, stale, best_state = [], float("inf"), 0, None
        for _ in range(epochs):
            optimiser.zero_grad()
            loss = F.cross_entropy(module.logits(train), train.choice)
            penalty = sum(
                parameter.square().mean()
                for name, parameter in module.named_parameters()
                if name != "log_tie"
            )
            objective = loss + l2 * penalty
            objective.backward()
            optimiser.step()
            value = float(objective.detach().cpu())
            history.append(value)
            if value < best - 1e-7:
                best, stale = value, 0
                best_state = {k: v.detach().clone() for k, v in module.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        if best_state:
            module.load_state_dict(best_state)
        return cls(module, rank, l2, history)

    def predict_proba(self, data: EncodedVotes) -> np.ndarray:
        with torch.no_grad():
            return torch.softmax(self.module.logits(data), -1).cpu().numpy()

    def utilities(self) -> np.ndarray:
        values = self.module.utility.detach().cpu().numpy()
        return values - values.mean()

    def aligned_image_factors(self, reference: np.ndarray) -> np.ndarray:
        current = self.module.image_factor.detach().cpu().numpy()
        rotation, _ = orthogonal_procrustes(current, reference)
        return current @ rotation

