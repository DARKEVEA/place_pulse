from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from placepulse_cusp.models.base import EncodedVotes


class _MixtureModule(nn.Module):
    def __init__(
        self, n_images: int, n_voters: int, classes: int, response_styles: bool = True
    ):
        super().__init__()
        self.base_utility = nn.Parameter(torch.zeros(n_images))
        self.class_deviation = nn.Parameter(torch.randn(classes, n_images) * 0.02)
        self.log_tie = nn.Parameter(torch.full((classes,), -1.0))
        self.class_logits = nn.Parameter(torch.zeros(classes))
        style_voters = n_voters if response_styles else 0
        self.left_bias = nn.Parameter(torch.zeros(style_voters))
        self.voter_tie = nn.Parameter(torch.zeros(style_voters))
        self.response_styles = response_styles

    def class_log_probs(
        self, data: EncodedVotes, *, use_response_styles: bool = True
    ) -> torch.Tensor:
        utility = self.base_utility[None, :] + (
            self.class_deviation - self.class_deviation.mean(dim=0, keepdim=True)
        )
        centered = utility - utility.mean(dim=1, keepdim=True)
        delta = centered[:, data.left] - centered[:, data.right]
        bias = torch.zeros(data.n_votes, device=delta.device)
        tie_style = torch.zeros_like(bias)
        known = data.voter >= 0
        if self.response_styles and use_response_styles and known.any():
            bias[known] = self.left_bias[data.voter[known]]
            tie_style[known] = self.voter_tie[data.voter[known]]
        delta = delta + bias[None, :]
        logits = torch.stack(
            (
                delta / 2.0,
                -delta / 2.0,
                torch.log(torch.tensor(2.0, device=delta.device))
                + self.log_tie[:, None].expand_as(delta)
                + tie_style[None, :],
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
    utility_l2: float = 1e-3
    style_l2: float = 1e-3

    @classmethod
    def fit(
        cls,
        train: EncodedVotes,
        *,
        classes: int = 2,
        l2: float = 1e-3,
        utility_l2: float | None = None,
        style_l2: float | None = None,
        response_styles: bool = True,
        dirichlet_alpha: float = 1.2,
        epochs: int = 200,
        learning_rate: float = 0.03,
        patience: int = 30,
        batch_size: int | None = None,
        lbfgs_steps: int = 0,
    ) -> "MixtureDavidsonModel":
        utility_l2 = l2 if utility_l2 is None else utility_l2
        style_l2 = utility_l2 if style_l2 is None else style_l2
        module = _MixtureModule(
            train.n_images, train.n_voters, classes, response_styles=response_styles
        ).to(train.left.device)
        optimiser = torch.optim.Adam(module.parameters(), lr=learning_rate)
        history, best, stale, best_state = [], float("inf"), 0, None
        n_voters = train.n_voters

        def e_step() -> torch.Tensor:
            with torch.no_grad():
                totals = torch.zeros(
                    max(n_voters, 1), classes, device=train.left.device
                )
                for batch in train.batches(batch_size):
                    logp = module.class_log_probs(batch)
                    selected = logp.gather(
                        2,
                        batch.choice[None, :, None].expand(classes, -1, 1),
                    ).squeeze(-1).T
                    known = batch.voter >= 0
                    if known.any() and n_voters:
                        totals.index_add_(0, batch.voter[known], selected[known])
                totals += torch.log_softmax(module.class_logits, 0)
                return torch.softmax(totals, -1)

        def objective(posterior: torch.Tensor) -> torch.Tensor:
            expected_nll = torch.zeros((), device=train.left.device)
            for batch in train.batches(batch_size):
                logp = module.class_log_probs(batch)
                selected = logp.gather(
                    2,
                    batch.choice[None, :, None].expand(classes, -1, 1),
                ).squeeze(-1).T
                weights = torch.softmax(module.class_logits, 0).expand(
                    batch.n_votes, -1
                ).clone()
                known = batch.voter >= 0
                if known.any() and n_voters:
                    weights[known] = posterior[batch.voter[known]]
                expected_nll = expected_nll - (
                    weights.detach() * selected
                ).sum()
            class_prob = torch.softmax(module.class_logits, 0)
            prior = -(dirichlet_alpha - 1.0) * torch.log(
                class_prob + 1e-12
            ).mean()
            posterior_mean = (
                posterior[:n_voters].mean(0).detach()
                if n_voters
                else class_prob.detach()
            )
            mixing_loss = -(
                posterior_mean * torch.log_softmax(module.class_logits, 0)
            ).sum()
            penalty = utility_l2 * module.base_utility.square().sum()
            penalty = penalty + l2 * module.class_deviation.square().sum()
            if response_styles:
                penalty = penalty + style_l2 * (
                    module.left_bias.square().sum()
                    + module.voter_tie.square().sum()
                )
            return (
                expected_nll / max(train.n_votes, 1)
                + 0.05 * mixing_loss
                + (penalty + prior) / max(train.n_votes, 1)
            )

        posterior = e_step()
        for _ in range(epochs):
            posterior = e_step()
            optimiser.zero_grad()
            loss = objective(posterior)
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
            posterior = e_step()
            lbfgs = torch.optim.LBFGS(
                module.parameters(),
                max_iter=lbfgs_steps,
                line_search_fn="strong_wolfe",
            )

            def closure():
                lbfgs.zero_grad()
                value = objective(posterior)
                value.backward()
                return value

            lbfgs.step(closure)
            history.append(float(closure().detach().cpu()))
        posterior_np = e_step()[:n_voters].cpu().numpy()
        return cls(module, classes, l2, posterior_np, history, utility_l2, style_l2)

    def predict_proba(
        self, data: EncodedVotes, *, population: bool = False
    ) -> np.ndarray:
        with torch.no_grad():
            class_probs = self.module.class_log_probs(data).exp().permute(1, 0, 2)
            weights = torch.softmax(self.module.class_logits, 0).expand(data.n_votes, -1).clone()
            known = (data.voter >= 0) & (not population)
            if known.any() and len(self.posterior):
                posterior = torch.as_tensor(
                    self.posterior, dtype=weights.dtype, device=weights.device
                )
                valid = known & (data.voter < posterior.shape[0])
                weights[valid] = posterior[data.voter[valid]]
            return (class_probs * weights[:, :, None]).sum(1).cpu().numpy()

    def class_weights(self) -> np.ndarray:
        return torch.softmax(self.module.class_logits, 0).detach().cpu().numpy()

    def infer_posterior(
        self,
        data: EncodedVotes,
        n_voters: int | None = None,
        *,
        use_response_styles: bool = False,
    ) -> np.ndarray:
        """Infer class probabilities from a voter's observed history without refitting utilities."""
        size = n_voters if n_voters is not None else data.n_voters
        with torch.no_grad():
            logp = self.module.class_log_probs(
                data, use_response_styles=use_response_styles
            )
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
        values = (
            self.module.base_utility[None, :]
            + (
                self.module.class_deviation
                - self.module.class_deviation.mean(dim=0, keepdim=True)
            )
        ).detach().cpu().numpy()
        return values - values.mean(axis=1, keepdims=True)

    def align_to(self, reference: np.ndarray) -> np.ndarray:
        current = self.utilities()
        correlation = np.corrcoef(current, reference)[: self.classes, self.classes :]
        rows, cols = linear_sum_assignment(-correlation)
        aligned = np.empty_like(current)
        aligned[cols] = current[rows]
        return aligned
