from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

from placepulse_cusp.cusp.density import CuspDensity, MixtureExpertDensity
from placepulse_cusp.data.splits import grouped_edge_folds
from placepulse_cusp.evaluation.gates import heterogeneity_verdict
from placepulse_cusp.evaluation.metrics import (
    clustered_elpd_bootstrap,
    empirical_probabilities,
    log_score,
    ranking_reversal_fraction,
)
from placepulse_cusp.models.base import VoteEncoder, select_device
from placepulse_cusp.provenance import metadata, write_json


def _cusp_sample(
    n: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = rng.normal(size=(n, 2))
    alpha = x[:, 0]
    beta = 1.5 + 1.2 * x[:, 1]
    grid = np.linspace(-4, 4, 1001)
    y = np.empty(n)
    for index in range(n):
        logp = -(grid**4 / 4 - beta[index] * grid**2 / 2 - alpha[index] * grid) / 0.5
        probability = np.exp(logp - logp.max())
        probability /= probability.sum()
        y[index] = rng.choice(grid, p=probability)
    return x, y, np.ones(n)


def validate_density_recovery(config: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(config["project"]["seed"])
    repetitions = config["simulation"]["repetitions"]
    cusp_wins, mixture_false_wins = 0, 0
    details = []
    n = min(max(config["simulation"]["votes"] // 10, 300), 2500)
    for repetition in range(repetitions):
        x, y, weight = _cusp_sample(n, rng)
        split = int(n * 0.8)
        cusp = CuspDensity(
            quadrature_points=config["cusp"]["quadrature_points"],
            domain=tuple(config["cusp"]["domain"]),
            max_iterations=config["cusp"]["max_iterations"],
        ).fit(x[:split], y[:split], weight[:split])
        mixture = MixtureExpertDensity().fit(x[:split], y[:split], weight[:split])
        cusp_score = float(cusp.logpdf(x[split:], y[split:]).mean())
        mixture_score = float(mixture.logpdf(x[split:], y[split:]).mean())
        cusp_wins += cusp_score > mixture_score

        means = np.where(rng.random(n) < 0.5, -1.5 + x[:, 0], 1.5 + x[:, 0])
        mixture_y = means + rng.normal(0, 0.5, n)
        cusp_m = CuspDensity(
            quadrature_points=config["cusp"]["quadrature_points"],
            domain=tuple(config["cusp"]["domain"]),
            max_iterations=config["cusp"]["max_iterations"],
        ).fit(x[:split], mixture_y[:split])
        moe_m = MixtureExpertDensity().fit(x[:split], mixture_y[:split])
        false_win = cusp_m.logpdf(x[split:], mixture_y[split:]).mean() > moe_m.logpdf(
            x[split:], mixture_y[split:]
        ).mean()
        mixture_false_wins += false_win
        details.append(
            {
                "repetition": repetition,
                "cusp_score": cusp_score,
                "mixture_score": mixture_score,
                "mixture_false_cusp_win": bool(false_win),
            }
        )
    recovery = cusp_wins / repetitions
    false_positive = mixture_false_wins / repetitions
    # Smoke configurations are diagnostic and intentionally too small for confirmatory thresholds.
    result = {
        "status": "ok"
        if recovery >= config["simulation"]["recovery_min_rate"]
        and false_positive <= config["simulation"]["cusp_max_mixture_false_positive"]
        else "failed",
        "cusp_recovery_rate": recovery,
        "mixture_false_cusp_rate": false_positive,
        "repetitions": repetitions,
        "details": details,
        "provenance": metadata(config),
    }
    target = Path(config["reporting"]["artifacts_dir"]) / "metrics" / "simulation_recovery.json"
    write_json(target, result)
    return result


def _synthetic_frame(
    config: dict[str, Any], mechanism: str, seed: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    sim = config["simulation"]
    rng = np.random.default_rng(seed)
    n_voters = int(sim["voters"])
    n_images = int(sim["images"])
    n_votes = int(sim["votes"])
    voter = rng.integers(0, n_voters, n_votes)
    left = rng.integers(0, n_images, n_votes)
    right = rng.integers(0, n_images, n_votes)
    same = left == right
    right[same] = (right[same] + 1) % n_images
    base = np.zeros(n_images) if mechanism == "null" else rng.normal(size=n_images)
    delta = base[left] - base[right]
    true_class = np.zeros(n_voters, dtype=int)
    if mechanism == "continuous":
        voter_factor = rng.normal(scale=1.0, size=(n_voters, 2))
        image_factor = rng.normal(scale=1.0, size=(n_images, 2))
        delta += (
            voter_factor[voter] * (image_factor[left] - image_factor[right])
        ).sum(1)
    elif mechanism == "mixture":
        true_class = rng.choice(3, n_voters, p=[0.40, 0.35, 0.25])
        deviation = rng.normal(scale=2.0, size=(3, n_images))
        delta += (
            deviation[true_class[voter], left]
            - deviation[true_class[voter], right]
        )
    logits = np.column_stack(
        (delta / 2, -delta / 2, np.full(n_votes, np.log(2.0) - 1.0))
    )
    probability = np.exp(logits - logits.max(1, keepdims=True))
    probability /= probability.sum(1, keepdims=True)
    draw = rng.random(n_votes)
    choice_index = (draw[:, None] > probability.cumsum(1)).sum(1)
    labels = np.asarray(["left", "right", "equal"])
    voter_ids = [f"voter_{x:05d}" for x in voter]
    frame = pl.DataFrame(
        {
            "vote_id": [f"{seed}_{x}" for x in range(n_votes)],
            "voter_id": voter_ids,
            "dimension": ["safety"] * n_votes,
            "left_image_id": [f"image_{x:05d}" for x in left],
            "right_image_id": [f"image_{x:05d}" for x in right],
            "choice": labels[choice_index],
        }
    )
    truth = {f"voter_{index:05d}": int(value) for index, value in enumerate(true_class)}
    return frame, truth


def _model_recovery_once(
    config: dict[str, Any], mechanism: str, seed: int
) -> dict[str, Any]:
    # Import locally to avoid a module cycle: the production pipeline invokes
    # this calibration routine before it invokes run_dimension.
    from placepulse_cusp.models import DavidsonModel
    from placepulse_cusp.pipeline import (
        _best_continuous_fit,
        _best_mixture_fit,
        _choice_array,
        _cluster_array,
        _eligible_test,
        _mixture_stability,
        _select_continuous,
        _select_mixture,
        _select_scalar_baseline,
        _training_kwargs,
    )
    frame, truth = _synthetic_frame(config, mechanism, seed)
    device = select_device(config["project"]["device"])
    outer_folds = min(3, int(config["splits"]["outer_folds"]))
    assigned = grouped_edge_folds(frame, outer_folds, seed + 7)
    score_sets = {"m0": [], "scalar": [], "continuous": [], "mixture": [], "cluster": []}
    selections = []
    final_mixture = None
    final_encoder = None
    for fold in range(outer_folds):
        train = frame.filter(assigned != fold)
        test = _eligible_test(train, frame.filter(assigned == fold))
        if test.height < 20:
            continue
        baseline = _select_scalar_baseline(train, config, device)
        rank, continuous_l2 = _select_continuous(train, config, device, baseline)
        classes, mixture_l2 = _select_mixture(train, config, device, baseline)
        encoder = VoteEncoder().fit(train)
        encoded_train = encoder.transform(train, device)
        encoded_test = encoder.transform(test, device)
        scalar = DavidsonModel.fit(
            encoded_train,
            utility_l2=baseline["utility_l2"],
            style_l2=baseline["style_l2"],
            response_styles=baseline["response_styles"],
            **_training_kwargs(config),
        )
        continuous = _best_continuous_fit(
            encoded_train,
            config,
            baseline,
            rank,
            continuous_l2,
            seed + 1000 + fold * 100,
            selection=False,
        )
        mixture = _best_mixture_fit(
            encoded_train,
            config,
            baseline,
            classes,
            mixture_l2,
            seed + 2000 + fold * 100,
            selection=False,
        )
        choices = _choice_array(encoded_test)
        empirical = empirical_probabilities(_choice_array(encoded_train))
        m0 = np.repeat(empirical[None, :], len(choices), axis=0)
        score_sets["m0"].append(log_score(m0, choices))
        score_sets["scalar"].append(log_score(scalar.predict_proba(encoded_test), choices))
        score_sets["continuous"].append(
            log_score(continuous.predict_proba(encoded_test), choices)
        )
        score_sets["mixture"].append(
            log_score(mixture.predict_proba(encoded_test), choices)
        )
        score_sets["cluster"].append(_cluster_array(test))
        selections.append((baseline, rank, continuous_l2, classes, mixture_l2))
        final_mixture, final_encoder = mixture, encoder
    if not selections:
        return {"mechanism": mechanism, "verdict": "MODEL_CALIBRATION_FAILED"}
    values = {name: np.concatenate(items) for name, items in score_sets.items()}
    clusters = values["cluster"]
    ci_kwargs = {
        "repetitions": config["gates"]["elpd_bootstrap"],
        "seed": seed,
    }
    baseline, rank, continuous_l2, classes, mixture_l2 = selections[-1]
    scalar_ci = clustered_elpd_bootstrap(
        values["scalar"], values["m0"], clusters, **ci_kwargs
    )
    continuous_ci = clustered_elpd_bootstrap(
        values["continuous"], values["scalar"], clusters, **ci_kwargs
    )
    mixture_ci = clustered_elpd_bootstrap(
        values["mixture"], values["scalar"], clusters, **ci_kwargs
    )
    truth_ari = 0.0
    if final_mixture is not None and final_encoder is not None:
        predicted = final_mixture.posterior.argmax(1)
        actual = np.asarray([truth[x] for x in final_encoder.voter_ids])
        truth_ari = float(adjusted_rand_score(actual, predicted))
    bootstrap_stability, _ = _mixture_stability(
        frame,
        classes,
        mixture_l2,
        baseline,
        config,
        device,
        refits=config["simulation"].get("stability_refits", 5),
    )
    verdict, gates = heterogeneity_verdict(
        config,
        float(-values["m0"].mean()),
        float(-values["scalar"].mean()),
        float(-values["continuous"].mean()),
        float(-values["mixture"].mean()),
        continuous_ci,
        mixture_ci,
        scalar_ci,
        final_mixture.class_weights().tolist(),
        ranking_reversal_fraction(final_mixture.utilities()),
        bootstrap_stability,
        True,
        True,
        simulation_ok=True,
        selection_boundary=baseline.get("selection_boundary", False),
    )
    return {
        "mechanism": mechanism,
        "verdict": verdict,
        "gates": gates,
        "selected_rank": rank,
        "selected_classes": classes,
        "stability_ari": bootstrap_stability,
        "truth_ari": truth_ari,
    }


def validate_model_recovery(config: dict[str, Any]) -> dict[str, Any]:
    repetitions = int(
        config["simulation"].get("model_repetitions", config["simulation"]["repetitions"])
    )
    expected = {
        "null": "SCALAR_SIGNAL_NOT_ESTABLISHED",
        "scalar": "SCALAR_NOT_REJECTED",
        "continuous": "SCALAR_REJECTED_CONTINUOUS",
        "mixture": "SCALAR_REJECTED_MIXTURE",
    }
    details = []
    rates = {}
    for mechanism, target in expected.items():
        matches = 0
        for repetition in range(repetitions):
            print(
                f"[simulation] {mechanism}: repetition "
                f"{repetition + 1}/{repetitions}",
                file=sys.stderr,
                flush=True,
            )
            item = _model_recovery_once(
                config,
                mechanism,
                config["project"]["seed"] + repetition + 100000,
            )
            details.append(item)
            recovered = item["verdict"] == target
            if mechanism == "continuous":
                recovered = recovered and item.get("selected_rank") == 2
            elif mechanism == "mixture":
                recovered = (
                    recovered
                    and item.get("selected_classes") == 3
                    and item.get("truth_ari", 0.0) >= config["gates"]["min_ari"]
                )
            matches += recovered
        rates[mechanism] = matches / max(repetitions, 1)
    scalar_false_rejection = sum(
        item["mechanism"] == "scalar"
        and item["verdict"]
        in {"SCALAR_REJECTED_CONTINUOUS", "SCALAR_REJECTED_MIXTURE"}
        for item in details
    ) / max(repetitions, 1)
    recovery_min = config["simulation"]["recovery_min_rate"]
    mixture_truth_ari = [
        item.get("truth_ari", 0.0)
        for item in details
        if item["mechanism"] == "mixture"
    ]
    median_mixture_truth_ari = float(np.median(mixture_truth_ari))
    result = {
        "status": (
            "ok"
            if rates["null"] >= 0.95
            and rates["scalar"] >= recovery_min
            and rates["continuous"] >= recovery_min
            and rates["mixture"] >= recovery_min
            and median_mixture_truth_ari >= config["gates"]["min_ari"]
            and scalar_false_rejection
            <= config["simulation"]["scalar_max_false_positive"]
            else "failed"
        ),
        "recovery_rates": rates,
        "scalar_false_rejection_rate": scalar_false_rejection,
        "median_mixture_truth_ari": median_mixture_truth_ari,
        "repetitions": repetitions,
        "details": details,
        "provenance": metadata(config),
    }
    target = (
        Path(config["reporting"]["artifacts_dir"])
        / "metrics"
        / "model_recovery.json"
    )
    write_json(target, result)
    return result

