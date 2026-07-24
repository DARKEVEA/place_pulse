from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from placepulse_cusp.models.base import EncodedVotes


class _MixtureModule(nn.Module):
    def __init__(self, n_images: int, classes: int):
        super().__init__()
        self.utility = nn.Parameter(torch.randn(classes, n_images) * 0.02)
        self.log_tie = nn.Parameter(torch.full((classes,), -1.0))
        self.class_logits = nn.Parameter(torch.zeros(classes))

    def class_log_probs(self, data: EncodedVotes) -> torch.Tensor:
        centered = self.utility - self.utility.mean(dim=1, keepdim=True)
        delta = centered[:, data.left] - centered[:, data.right]
        logits = torch.stack(
            (
                delta / 2.0,
                -delta / 2.0,
                torch.log(torch.tensor(2.0, device=delta.device))
                + self.log_tie[:, None].expand_as(delta),
            ),
            dim=-1,
        )
        return torch.log_softmax(logits, -1)


@dataclass
class MixtureDavidsonModel:
    module: _MixtureModule
    classes: int
    l2: float
    posterior: np.ndarray
    history: list[float]

    @classmethod
    def fit(
        cls,
        train: EncodedVotes,
        *,
        classes: int = 2,
        l2: float = 1e-3,
        dirichlet_alpha: float = 1.2,
        epochs: int = 200,
        learning_rate: float = 0.03,
        patience: int = 30,
    ) -> "MixtureDavidsonModel":
        module = _MixtureModule(train.n_images, classes).to(train.left.device)
        optimiser = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history, best, stale, best_state = [], float("inf"), 0, None
        n_voters = train.n_voters
        posterior = torch.softmax(module.class_logits, 0).expand(max(n_voters, 1), -1).clone()
        known = train.voter >= 0
        for _ in range(epochs):
            with torch.no_grad():
                logp = module.class_log_probs(train)
                selected = logp.gather(
                    2, train.choice[None, :, None].expand(classes, -1, 1)
                ).squeeze(-1).T
                totals = torch.zeros(max(n_voters, 1), classes, device=train.left.device)
                if known.any() and n_voters:
                    totals.index_add_(0, train.voter[known], selected[known])
                totals += torch.log_softmax(module.class_logits, 0)
                posterior = torch.softmax(totals, -1)
            optimiser.zero_grad()
            logp = module.class_log_probs(train)
            selected = logp.gather(
                2, train.choice[None, :, None].expand(classes, -1, 1)
            ).squeeze(-1).T
            weights = torch.softmax(module.class_logits, 0).expand(train.n_votes, -1).clone()
            if known.any() and n_voters:
                weights[known] = posterior[train.voter[known]]
            expected_nll = -(weights.detach() * selected).sum(-1).mean()
            class_prob = torch.softmax(module.class_logits, 0)
            prior = -(dirichlet_alpha - 1.0) * torch.log(class_prob + 1e-12).mean()
            mixing_loss = -(
                posterior.mean(0).detach() * torch.log_softmax(module.class_logits, 0)
            ).sum()
            penalty = module.utility.square().mean()
            loss = expected_nll + 0.05 * mixing_loss + l2 * penalty + prior / max(
                train.n_votes, 1
            )
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
        with torch.no_grad():
            logp = module.class_log_probs(train)
            selected = logp.gather(
                2, train.choice[None, :, None].expand(classes, -1, 1)
            ).squeeze(-1).T
            totals = torch.zeros(max(n_voters, 1), classes, device=train.left.device)
            if known.any() and n_voters:
                totals.index_add_(0, train.voter[known], selected[known])
            totals += torch.log_softmax(module.class_logits, 0)
            posterior_np = torch.softmax(totals, -1)[:n_voters].cpu().numpy()
        return cls(module, classes, l2, posterior_np, history)

    def predict_proba(self, data: EncodedVotes) -> np.ndarray:
        with torch.no_grad():
            class_probs = self.module.class_log_probs(data).exp().permute(1, 0, 2)
            weights = torch.softmax(self.module.class_logits, 0).expand(data.n_votes, -1).clone()
            known = data.voter >= 0
            if known.any() and len(self.posterior):
                posterior = torch.as_tensor(
                    self.posterior, dtype=weights.dtype, device=weights.device
                )
                valid = known & (data.voter < posterior.shape[0])
                weights[valid] = posterior[data.voter[valid]]
            return (class_probs * weights[:, :, None]).sum(1).cpu().numpy()

    def class_weights(self) -> np.ndarray:
        return torch.softmax(self.module.class_logits, 0).detach().cpu().numpy()

    def infer_posterior(self, data: EncodedVotes, n_voters: int | None = None) -> np.ndarray:
        """Infer class probabilities from a voter's observed history without refitting utilities."""
        size = n_voters if n_voters is not None else data.n_voters
        with torch.no_grad():
            logp = self.module.class_log_probs(data)
            selected = logp.gather(
                2, data.choice[None, :, None].expand(self.classes, -1, 1)
            ).squeeze(-1).T
            totals = torch.zeros(size, self.classes, device=data.left.device)
            known = (data.voter >= 0) & (data.voter < size)
            if known.any():
                totals.index_add_(0, data.voter[known], selected[known])
            totals += torch.log_softmax(self.module.class_logits, 0)
            return torch.softmax(totals, -1).cpu().numpy()

    def utilities(self) -> np.ndarray:
        values = self.module.utility.detach().cpu().numpy()
        return values - values.mean(axis=1, keepdims=True)

    def align_to(self, reference: np.ndarray) -> np.ndarray:
        current = self.utilities()
        correlation = np.corrcoef(current, reference)[: self.classes, self.classes :]
        rows, cols = linear_sum_assignment(-correlation)
        aligned = np.empty_like(current)
        aligned[cols] = current[rows]
        return aligned
