from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from torch import nn
from torch.nn import functional as F

from placepulse_cusp.models.base import EncodedVotes, davidson_logits


class _ContinuousModule(nn.Module):
    def __init__(
        self, n_images: int, n_voters: int, rank: int, response_styles: bool = True
    ):
        super().__init__()
        self.utility = nn.Parameter(torch.zeros(n_images))
        self.image_factor = nn.Parameter(torch.randn(n_images, rank) * 0.03)
        self.voter_factor = nn.Parameter(torch.randn(n_voters, rank) * 0.03)
        style_voters = n_voters if response_styles else 0
        self.left_bias = nn.Parameter(torch.zeros(style_voters))
        self.voter_tie = nn.Parameter(torch.zeros(style_voters))
        self.log_tie = nn.Parameter(torch.tensor(-1.0))
        self.response_styles = response_styles

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
            delta[known] += (voter_factor * contrast).sum(-1)
            if self.response_styles:
                delta[known] += self.left_bias[data.voter[known]]
                tie[known] = self.voter_tie[data.voter[known]]
        return davidson_logits(delta, self.log_tie, tie)


@dataclass
class ContinuousPreferenceModel:
    module: _ContinuousModule
    rank: int
    l2: float
    history: list[float]
    utility_l2: float = 1e-3
    style_l2: float = 1e-3

    @classmethod
    def fit(
        cls,
        train: EncodedVotes,
        *,
        rank: int = 2,
        l2: float = 1e-3,
        utility_l2: float | None = None,
        style_l2: float | None = None,
        response_styles: bool = True,
        epochs: int = 250,
        learning_rate: float = 0.03,
        patience: int = 30,
        batch_size: int | None = None,
        lbfgs_steps: int = 0,
    ) -> "ContinuousPreferenceModel":
        utility_l2 = l2 if utility_l2 is None else utility_l2
        style_l2 = utility_l2 if style_l2 is None else style_l2
        module = _ContinuousModule(
            train.n_images, train.n_voters, rank, response_styles=response_styles
        ).to(train.left.device)
        optimiser = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history, best, stale, best_state = [], float("inf"), 0, None
        for _ in range(epochs):
            optimiser.zero_grad()
            data_loss = torch.zeros((), device=train.left.device)
            for batch in train.batches(batch_size):
                data_loss = data_loss + F.cross_entropy(
                    module.logits(batch), batch.choice, reduction="sum"
                )
            penalty = utility_l2 * module.utility.square().sum()
            penalty = penalty + l2 * (
                module.image_factor.square().sum() + module.voter_factor.square().sum()
            )
            if response_styles:
                penalty = penalty + style_l2 * (
                    module.left_bias.square().sum() + module.voter_tie.square().sum()
                )
            objective = (data_loss + penalty) / max(train.n_votes, 1)
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
        if lbfgs_steps > 0:
            lbfgs = torch.optim.LBFGS(
                module.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe"
            )

            def closure():
                lbfgs.zero_grad()
                data_loss = torch.zeros((), device=train.left.device)
                for batch in train.batches(batch_size):
                    data_loss = data_loss + F.cross_entropy(
                        module.logits(batch), batch.choice, reduction="sum"
                    )
                penalty = utility_l2 * module.utility.square().sum()
                penalty = penalty + l2 * (
                    module.image_factor.square().sum()
                    + module.voter_factor.square().sum()
                )
                if response_styles:
                    penalty = penalty + style_l2 * (
                        module.left_bias.square().sum()
                        + module.voter_tie.square().sum()
                    )
                objective = (data_loss + penalty) / max(train.n_votes, 1)
                objective.backward()
                return objective

            lbfgs.step(closure)
            history.append(float(closure().detach().cpu()))
        return cls(module, rank, l2, history, utility_l2, style_l2)

    def predict_proba(
        self,
        data: EncodedVotes,
        new_user_samples: int = 32,
        *,
        population: bool = False,
    ) -> np.ndarray:
        with torch.no_grad():
            probabilities = torch.softmax(self.module.logits(data), -1)
            unknown = (
                torch.ones_like(data.voter, dtype=torch.bool)
                if population
                else data.voter < 0
            )
            n_voters = self.module.voter_factor.shape[0]
            if unknown.any() and n_voters:
                sample_count = min(new_user_samples, n_voters)
                sampled = torch.linspace(
                    0, n_voters - 1, sample_count, device=data.left.device
                ).long()
                utility = self.module.utility - self.module.utility.mean()
                base_delta = utility[data.left[unknown]] - utility[data.right[unknown]]
                contrast = (
                    self.module.image_factor[data.left[unknown]]
                    - self.module.image_factor[data.right[unknown]]
                )
                preference = contrast @ self.module.voter_factor[sampled].T
                delta = base_delta[:, None] + preference
                if self.module.response_styles:
                    delta = delta + self.module.left_bias[sampled][None, :]
                    tie = self.module.voter_tie[sampled][None, :].expand_as(delta)
                else:
                    tie = torch.zeros_like(delta)
                logits = davidson_logits(
                    delta.reshape(-1), self.module.log_tie, tie.reshape(-1)
                ).reshape(len(base_delta), sample_count, 3)
                probabilities[unknown] = torch.softmax(logits, -1).mean(1)
            return probabilities.cpu().numpy()

    def utilities(self) -> np.ndarray:
        values = self.module.utility.detach().cpu().numpy()
        return values - values.mean()

    def aligned_image_factors(self, reference: np.ndarray) -> np.ndarray:
        current = self.module.image_factor.detach().cpu().numpy()
        rotation, _ = orthogonal_procrustes(current, reference)
        return current @ rotation
