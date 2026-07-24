from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from placepulse_cusp.models.base import EncodedVotes, davidson_logits


class _DavidsonModule(nn.Module):
    def __init__(self, n_images: int, n_voters: int):
        super().__init__()
        self.utility = nn.Parameter(torch.zeros(n_images))
        self.left_bias = nn.Parameter(torch.zeros(n_voters))
        self.voter_tie = nn.Parameter(torch.zeros(n_voters))
        self.log_tie = nn.Parameter(torch.tensor(-1.0))

    def logits(self, data: EncodedVotes) -> torch.Tensor:
        known = data.voter >= 0
        bias = torch.zeros_like(data.left, dtype=self.utility.dtype)
        tie = torch.zeros_like(bias)
        if self.left_bias.numel():
            bias[known] = self.left_bias[data.voter[known]]
            tie[known] = self.voter_tie[data.voter[known]]
        centered = self.utility - self.utility.mean()
        delta = centered[data.left] - centered[data.right] + bias
        return davidson_logits(delta, self.log_tie, tie)


@dataclass
class DavidsonModel:
    module: _DavidsonModule
    l2: float
    history: list[float]

    @classmethod
    def fit(
        cls,
        train: EncodedVotes,
        *,
        l2: float = 1e-3,
        epochs: int = 200,
        learning_rate: float = 0.03,
        patience: int = 30,
    ) -> "DavidsonModel":
        module = _DavidsonModule(train.n_images, train.n_voters).to(train.left.device)
        optimiser = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history, best, stale = [], float("inf"), 0
        best_state = None
        for _ in range(epochs):
            optimiser.zero_grad()
            logits = module.logits(train)
            penalty = (
                module.utility.square().mean()
                + module.left_bias.square().mean()
                + module.voter_tie.square().mean()
            )
            loss = F.cross_entropy(logits, train.choice) + l2 * penalty
            loss.backward()
            optimiser.step()
            value = float(loss.detach().cpu())
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
        return cls(module, l2, history)

    def predict_proba(self, data: EncodedVotes, new_user_samples: int = 32) -> np.ndarray:
        with torch.no_grad():
            probabilities = torch.softmax(self.module.logits(data), -1)
            unknown = data.voter < 0
            n_voters = self.module.left_bias.numel()
            if unknown.any() and n_voters:
                sample_count = min(new_user_samples, n_voters)
                sampled = torch.linspace(
                    0, n_voters - 1, sample_count, device=data.left.device
                ).long()
                utility = self.module.utility - self.module.utility.mean()
                base_delta = utility[data.left[unknown]] - utility[data.right[unknown]]
                delta = base_delta[:, None] + self.module.left_bias[sampled][None, :]
                tie = self.module.voter_tie[sampled][None, :].expand_as(delta)
                logits = davidson_logits(
                    delta.reshape(-1), self.module.log_tie, tie.reshape(-1)
                ).reshape(len(base_delta), sample_count, 3)
                probabilities[unknown] = torch.softmax(logits, -1).mean(1)
            return probabilities.cpu().numpy()

    def utilities(self) -> np.ndarray:
        values = self.module.utility.detach().cpu().numpy()
        return values - values.mean()
