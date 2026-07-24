from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import minimize
from scipy.special import expit, logsumexp
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

LOG_2PI = np.log(2.0 * np.pi)


def _normal_logpdf(y: np.ndarray, mean: np.ndarray, sigma: np.ndarray | float) -> np.ndarray:
    sigma = np.maximum(np.asarray(sigma), 1e-6)
    return -0.5 * (LOG_2PI + 2 * np.log(sigma) + ((y - mean) / sigma) ** 2)


@dataclass
class LinearGaussianDensity:
    coef_: np.ndarray | None = None
    sigma_: float = 1.0

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "LinearGaussianDensity":
        design = np.column_stack([np.ones(len(x)), x])
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        root = np.sqrt(weights)[:, None]
        self.coef_ = np.linalg.lstsq(design * root, y * root[:, 0], rcond=None)[0]
        residual = y - design @ self.coef_
        self.sigma_ = float(np.sqrt(np.average(residual**2, weights=weights) + 1e-8))
        return self

    def logpdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x])
        return _normal_logpdf(y, design @ self.coef_, self.sigma_)

    def parameters(self) -> dict[str, object]:
        return {"coef": self.coef_.tolist(), "sigma": self.sigma_}


@dataclass
class SplineGaussianDensity:
    knots: int = 5
    model_: object | None = None
    sigma_: float = 1.0

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "SplineGaussianDensity":
        self.model_ = make_pipeline(
            SplineTransformer(n_knots=self.knots, degree=3, include_bias=False),
            Ridge(alpha=1e-3),
        )
        self.model_.fit(x, y, ridge__sample_weight=sample_weight)
        residual = y - self.model_.predict(x)
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        self.sigma_ = float(np.sqrt(np.average(residual**2, weights=weights) + 1e-8))
        return self

    def logpdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _normal_logpdf(y, self.model_.predict(x), self.sigma_)

    def parameters(self) -> dict[str, object]:
        return {"knots": self.knots, "sigma": self.sigma_}


@dataclass
class MixtureExpertDensity:
    params_: np.ndarray | None = None
    success_: bool = False

    @staticmethod
    def _unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Two affine means, an affine gate, and two log standard deviations.
        mean1 = params[0:3]
        mean2 = params[3:6]
        gate = params[6:9]
        sigma = np.exp(np.clip(params[9:11], -6, 4))
        return mean1, mean2, gate, sigma

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "MixtureExpertDensity":
        design = np.column_stack([np.ones(len(x)), x])
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        base = np.linalg.lstsq(design, y, rcond=None)[0]
        spread = np.std(y) or 1.0
        initial = np.r_[
            base + np.array([-0.5 * spread, 0, 0]),
            base + np.array([0.5 * spread, 0, 0]),
            np.zeros(3),
            np.log([spread * 0.7, spread * 0.7]),
        ]

        def objective(params: np.ndarray) -> float:
            mean1, mean2, gate, sigma = self._unpack(params)
            gate_p = np.clip(expit(design @ gate), 1e-8, 1 - 1e-8)
            terms = np.column_stack(
                [
                    np.log(gate_p) + _normal_logpdf(y, design @ mean1, sigma[0]),
                    np.log1p(-gate_p) + _normal_logpdf(y, design @ mean2, sigma[1]),
                ]
            )
            return float(-np.sum(weights * logsumexp(terms, axis=1)) / weights.sum())

        result = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 1000})
        self.params_, self.success_ = result.x, bool(result.success)
        return self

    def logpdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        design = np.column_stack([np.ones(len(x)), x])
        mean1, mean2, gate, sigma = self._unpack(self.params_)
        gate_p = np.clip(expit(design @ gate), 1e-8, 1 - 1e-8)
        return logsumexp(
            np.column_stack(
                [
                    np.log(gate_p) + _normal_logpdf(y, design @ mean1, sigma[0]),
                    np.log1p(-gate_p) + _normal_logpdf(y, design @ mean2, sigma[1]),
                ]
            ),
            axis=1,
        )

    def parameters(self) -> dict[str, object]:
        return {"params": self.params_.tolist(), "success": self.success_}


@dataclass
class CuspDensity:
    quadrature_points: int = 160
    domain: tuple[float, float] = (-8.0, 8.0)
    max_iterations: int = 1000
    params_: np.ndarray | None = None
    success_: bool = False
    integration_error_: float = float("nan")

    def _nodes(self, points: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        raw_nodes, raw_weights = leggauss(points or self.quadrature_points)
        lower, upper = self.domain
        nodes = (raw_nodes + 1) * (upper - lower) / 2 + lower
        weights = raw_weights * (upper - lower) / 2
        return nodes, weights

    @staticmethod
    def _controls(params: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        design = np.column_stack([np.ones(len(x)), x])
        alpha = design @ params[:3]
        beta = design @ params[3:6]
        sigma2 = np.exp(np.clip(2 * params[6], -10, 8))
        return alpha, beta, float(sigma2)

    def _log_normalizer(
        self, alpha: np.ndarray, beta: np.ndarray, sigma2: float, points: int | None = None
    ) -> np.ndarray:
        nodes, weights = self._nodes(points)
        potential = (
            nodes[None, :] ** 4 / 4
            - beta[:, None] * nodes[None, :] ** 2 / 2
            - alpha[:, None] * nodes[None, :]
        )
        return logsumexp(-potential / sigma2 + np.log(weights)[None, :], axis=1)

    def fit(
        self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> "CuspDensity":
        weights = np.ones(len(y)) if sample_weight is None else sample_weight
        initial = np.array([0.0, 1.0, 0.0, -0.5, 0.0, 1.0, np.log(np.std(y) or 1.0)])

        def objective(params: np.ndarray) -> float:
            alpha, beta, sigma2 = self._controls(params, x)
            potential = y**4 / 4 - beta * y**2 / 2 - alpha * y
            logp = -potential / sigma2 - self._log_normalizer(alpha, beta, sigma2)
            return float(-np.sum(weights * logp) / weights.sum())

        result = minimize(
            objective,
            initial,
            method="L-BFGS-B",
            bounds=[(-10, 10)] * 6 + [(-4, 3)],
            options={"maxiter": self.max_iterations},
        )
        self.params_, self.success_ = result.x, bool(result.success)
        alpha, beta, sigma2 = self._controls(self.params_, x)
        coarse = self._log_normalizer(alpha, beta, sigma2)
        fine = self._log_normalizer(alpha, beta, sigma2, self.quadrature_points * 2)
        self.integration_error_ = float(np.mean(np.abs(coarse - fine)))
        return self

    def logpdf(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        alpha, beta, sigma2 = self._controls(self.params_, x)
        potential = y**4 / 4 - beta * y**2 / 2 - alpha * y
        return -potential / sigma2 - self._log_normalizer(alpha, beta, sigma2)

    def fold_mask(self, x: np.ndarray) -> np.ndarray:
        alpha, beta, _ = self._controls(self.params_, x)
        # Cubic y^3 - beta*y - alpha has three real roots when discriminant > 0.
        return 4 * beta**3 - 27 * alpha**2 > 0

    def parameters(self) -> dict[str, object]:
        return {
            "params": self.params_.tolist(),
            "success": self.success_,
            "integration_error": self.integration_error_,
            "domain": list(self.domain),
            "quadrature_points": self.quadrature_points,
        }

