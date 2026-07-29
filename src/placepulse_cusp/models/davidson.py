from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from placepulse_cusp.models.base import EncodedVotes, davidson_logits


class _DavidsonModule(nn.Module):
    def __init__(self, n_images: int, n_voters: int, response_styles: bool = True):
        super().__init__()
        self.utility = nn.Parameter(torch.zeros(n_images))
        style_voters = n_voters if response_styles else 0
        self.left_bias = nn.Parameter(torch.zeros(style_voters))
        self.voter_tie = nn.Parameter(torch.zeros(style_voters))
        self.log_tie = nn.Parameter(torch.tensor(-1.0))
        self.response_styles = response_styles

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
    utility_l2: float
    style_l2: float
    history: list[float]

    @property
    def l2(self) -> float:
        """Backward-compatible alias for older callers and serialized tables."""
        return self.utility_l2

    @classmethod
    def fit(
        cls,
        train: EncodedVotes,
        *,
        l2: float | None = None,
        utility_l2: float = 1e-3,
        style_l2: float | None = None,
        response_styles: bool = True,
        epochs: int = 200,
        learning_rate: float = 0.03,
        patience: int = 30,
        batch_size: int | None = None,
        lbfgs_steps: int = 0,
    ) -> "DavidsonModel":
        if l2 is not None:
            utility_l2 = l2
            if style_l2 is None:
                style_l2 = l2
        style_l2 = utility_l2 if style_l2 is None else style_l2
        module = _DavidsonModule(
            train.n_images, train.n_voters, response_styles=response_styles
        ).to(train.left.device)
        optimiser = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history, best, stale = [], float("inf"), 0
        best_state = None
        for _ in range(epochs):
            optimiser.zero_grad()
            data_loss = torch.zeros((), device=train.left.device)
            for batch in train.batches(batch_size):
                data_loss = data_loss + F.cross_entropy(
                    module.logits(batch), batch.choice, reduction="sum"
                )
            penalty = utility_l2 * module.utility.square().sum()
            if response_styles:
                penalty = penalty + style_l2 * (
                    module.left_bias.square().sum() + module.voter_tie.square().sum()
                )
            loss = (data_loss + penalty) / max(train.n_votes, 1)
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
                if response_styles:
                    penalty = penalty + style_l2 * (
                        module.left_bias.square().sum() + module.voter_tie.square().sum()
                    )
                objective = (data_loss + penalty) / max(train.n_votes, 1)
                objective.backward()
                return objective

            lbfgs.step(closure)
            history.append(float(closure().detach().cpu()))
        return cls(module, utility_l2, style_l2, history)

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
