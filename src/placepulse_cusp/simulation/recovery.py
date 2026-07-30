from __future__ import annotations

import hashlib
import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score

from placepulse_cusp.cusp.density import CuspDensity, MixtureExpertDensity
from placepulse_cusp.data.splits import grouped_edge_folds
from placepulse_cusp.evaluation.gates import edge_predictive_gates, heterogeneity_verdict
from placepulse_cusp.evaluation.metrics import (
    clustered_elpd_bootstrap,
    empirical_probabilities,
    log_score,
    ranking_reversal_fraction,
)
from placepulse_cusp.models.base import VoteEncoder, select_device
from placepulse_cusp.provenance import metadata, write_json

RECOVERY_CHECKPOINT_SCHEMA_VERSION = 1


@lru_cache(maxsize=1)
def _recovery_code_hash() -> str:
    package_root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(package_root.rglob("*.py")):
        digest.update(str(path.relative_to(package_root)).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _checkpoint_path(
    config: dict[str, Any], family: str, mechanism: str, repetition: int
) -> Path:
    config_hash = config.get("_meta", {}).get("hash", "unknown")[:16]
    return (
        Path(config["reporting"]["artifacts_dir"])
        / "checkpoints"
        / family
        / (
            f"{mechanism}_repetition_{repetition:04d}_{config_hash}"
            f"_v{RECOVERY_CHECKPOINT_SCHEMA_VERSION}.json"
        )
    )


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write_json(temporary, payload)
    temporary.replace(path)


def _load_repetition_checkpoint(
    config: dict[str, Any], family: str, mechanism: str, repetition: int
) -> dict[str, Any] | None:
    path = _checkpoint_path(config, family, mechanism, repetition)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if (
        payload.get("schema_version") != RECOVERY_CHECKPOINT_SCHEMA_VERSION
        or payload.get("config_hash") != config.get("_meta", {}).get("hash")
        or payload.get("code_hash") != _recovery_code_hash()
        or payload.get("family") != family
        or payload.get("mechanism") != mechanism
        or payload.get("repetition") != repetition
    ):
        return None
    result = payload.get("result")
    return result if isinstance(result, dict) else None


def _save_repetition_checkpoint(
    config: dict[str, Any],
    family: str,
    mechanism: str,
    repetition: int,
    result: dict[str, Any],
) -> None:
    _atomic_write_json(
        _checkpoint_path(config, family, mechanism, repetition),
        {
            "schema_version": RECOVERY_CHECKPOINT_SCHEMA_VERSION,
            "config_hash": config.get("_meta", {}).get("hash"),
            "code_hash": _recovery_code_hash(),
            "family": family,
            "mechanism": mechanism,
            "repetition": repetition,
            "result": result,
        },
    )


def _write_recovery_progress(
    config: dict[str, Any],
    family: str,
    *,
    completed: int,
    total: int,
    status: str,
) -> None:
    _atomic_write_json(
        Path(config["reporting"]["artifacts_dir"])
        / "metrics"
        / f"{family}_progress.json",
        {
            "schema_version": RECOVERY_CHECKPOINT_SCHEMA_VERSION,
            "config_hash": config.get("_meta", {}).get("hash"),
            "code_hash": _recovery_code_hash(),
            "status": status,
            "completed": completed,
            "total": total,
        },
    )


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


def validate_density_recovery(
    config: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    repetitions = config["simulation"]["repetitions"]
    details = []
    n = min(max(config["simulation"]["votes"] // 10, 300), 2500)
    for repetition in range(repetitions):
        cached = (
            _load_repetition_checkpoint(
                config, "density_recovery", "density", repetition
            )
            if resume
            else None
        )
        if cached is not None:
            print(
                f"[resume] density: repetition {repetition + 1}/{repetitions}",
                file=sys.stderr,
                flush=True,
            )
            details.append(cached)
            _write_recovery_progress(
                config,
                "density_recovery",
                completed=len(details),
                total=repetitions,
                status="running",
            )
            continue
        rng = np.random.default_rng(
            config["project"]["seed"] + 200000 + repetition
        )
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
        item = {
            "repetition": repetition,
            "cusp_score": cusp_score,
            "mixture_score": mixture_score,
            "cusp_win": bool(cusp_score > mixture_score),
            "mixture_false_cusp_win": bool(false_win),
        }
        _save_repetition_checkpoint(
            config, "density_recovery", "density", repetition, item
        )
        details.append(item)
        _write_recovery_progress(
            config,
            "density_recovery",
            completed=len(details),
            total=repetitions,
            status="running",
        )
    cusp_wins = sum(bool(item["cusp_win"]) for item in details)
    mixture_false_wins = sum(
        bool(item["mixture_false_cusp_win"]) for item in details
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
    _write_recovery_progress(
        config,
        "density_recovery",
        completed=repetitions,
        total=repetitions,
        status="complete",
    )
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
    edge_gates = edge_predictive_gates(
        config,
        float(-values["m0"].mean()),
        float(-values["scalar"].mean()),
        float(-values["continuous"].mean()),
        float(-values["mixture"].mean()),
        continuous_ci,
        mixture_ci,
        scalar_ci,
        simulation_ok=True,
        selection_boundary=baseline.get("selection_boundary", False),
    )
    if edge_gates["baseline"] and edge_gates["mixture"]:
        bootstrap_stability, _ = _mixture_stability(
            frame,
            classes,
            mixture_l2,
            baseline,
            config,
            device,
            refits=config["simulation"].get("stability_refits", 5),
        )
    else:
        bootstrap_stability = 0.0
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
        "baseline_selection": {
            "name": baseline["name"],
            "utility_l2": float(baseline["utility_l2"]),
            "style_l2": float(baseline["style_l2"]),
            "response_styles": bool(baseline["response_styles"]),
            "selection_boundary": bool(
                baseline.get("selection_boundary", False)
            ),
            "selection_boundary_parameters": list(
                baseline.get("selection_boundary_parameters", [])
            ),
        },
        "selected_rank": rank,
        "selected_continuous_l2": float(continuous_l2),
        "selected_classes": classes,
        "selected_mixture_l2": float(mixture_l2),
        "stability_ari": bootstrap_stability,
        "truth_ari": truth_ari,
    }


def _model_recovery_assessment(
    config: dict[str, Any],
    mechanism: str,
    target_verdict: str,
    item: dict[str, Any],
) -> dict[str, Any]:
    """Assess synthetic recovery without weakening production verdicts.

    A null data-generating process can legitimately select the strongest
    utility shrinkage because its population utility is exactly zero. The
    production verdict remains conservative, but the negative-control
    recovery succeeds when the only problem is that high-regularisation
    boundary and no predictive improvement reaches the configured threshold.
    """
    if item.get("verdict") == target_verdict:
        recovered = True
        reason = "target_verdict"
    elif mechanism == "null":
        gates = item.get("gates", {})
        baseline = item.get("baseline_selection", {})
        boundary_parameters = baseline.get(
            "selection_boundary_parameters", []
        )
        candidates = [
            float(value) for value in config["models"]["l2_candidates"]
        ]
        upper = max(candidates)
        high_regularisation_boundary = (
            len(set(candidates)) > 1
            and bool(boundary_parameters)
            and set(boundary_parameters).issubset(
                {"utility_l2", "style_l2"}
            )
        )
        if "utility_l2" in boundary_parameters:
            high_regularisation_boundary = (
                high_regularisation_boundary
                and float(baseline.get("utility_l2", float("nan"))) == upper
            )
        if "style_l2" in boundary_parameters:
            high_regularisation_boundary = (
                high_regularisation_boundary
                and bool(baseline.get("response_styles", False))
                and float(baseline.get("style_l2", float("nan"))) == upper
            )
        threshold = float(config["gates"]["min_cross_entropy_reduction"])
        no_predictive_signal = (
            not bool(gates.get("baseline_predictive_gate", True))
            and not bool(gates.get("continuous_edge_predictive_gate", True))
            and not bool(gates.get("mixture_edge_predictive_gate", True))
            and float(gates.get("baseline_reduction", float("inf"))) < threshold
            and float(gates.get("continuous_reduction", float("inf"))) < threshold
            and float(gates.get("mixture_reduction", float("inf"))) < threshold
        )
        recovered = (
            item.get("verdict") == "MODEL_CALIBRATION_FAILED"
            and high_regularisation_boundary
            and no_predictive_signal
        )
        reason = (
            "null_high_regularisation_boundary"
            if recovered
            else "target_verdict_not_recovered"
        )
    else:
        recovered = False
        reason = "target_verdict_not_recovered"

    if recovered and mechanism == "continuous":
        recovered = item.get("selected_rank") == 2
        if not recovered:
            reason = "continuous_rank_not_recovered"
    elif recovered and mechanism == "mixture":
        recovered = (
            item.get("selected_classes") == 3
            and item.get("truth_ari", 0.0) >= config["gates"]["min_ari"]
        )
        if not recovered:
            reason = "mixture_structure_not_recovered"
    return {
        "recovered": bool(recovered),
        "reason": reason,
        "target_verdict": target_verdict,
        "raw_verdict": item.get("verdict"),
    }


def validate_model_recovery(
    config: dict[str, Any], *, resume: bool = False
) -> dict[str, Any]:
    repetitions = int(
        config["simulation"].get("model_repetitions", config["simulation"]["repetitions"])
    )
    all_expected = {
        "null": "SCALAR_SIGNAL_NOT_ESTABLISHED",
        "scalar": "SCALAR_NOT_REJECTED",
        "continuous": "SCALAR_REJECTED_CONTINUOUS",
        "mixture": "SCALAR_REJECTED_MIXTURE",
    }
    mechanisms = config["simulation"].get(
        "model_mechanisms", list(all_expected)
    )
    if (
        not isinstance(mechanisms, list)
        or not mechanisms
        or len(mechanisms) != len(set(mechanisms))
        or any(mechanism not in all_expected for mechanism in mechanisms)
    ):
        raise ValueError(
            "simulation.model_mechanisms must be a non-empty list containing "
            "unique values from: null, scalar, continuous, mixture"
        )
    expected = {
        mechanism: all_expected[mechanism] for mechanism in mechanisms
    }
    details = []
    rates = {}
    total = len(expected) * repetitions
    for mechanism, target in expected.items():
        matches = 0
        for repetition in range(repetitions):
            cached = (
                _load_repetition_checkpoint(
                    config, "model_recovery", mechanism, repetition
                )
                if resume
                else None
            )
            if cached is not None:
                print(
                    f"[resume] {mechanism}: repetition "
                    f"{repetition + 1}/{repetitions}",
                    file=sys.stderr,
                    flush=True,
                )
                item = cached
                computed = False
            else:
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
                item["repetition"] = repetition
                computed = True
            item["recovery_assessment"] = _model_recovery_assessment(
                config, mechanism, target, item
            )
            if computed:
                _save_repetition_checkpoint(
                    config, "model_recovery", mechanism, repetition, item
                )
            details.append(item)
            _write_recovery_progress(
                config,
                "model_recovery",
                completed=len(details),
                total=total,
                status="running",
            )
            matches += item["recovery_assessment"]["recovered"]
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
    median_mixture_truth_ari = (
        float(np.median(mixture_truth_ari)) if mixture_truth_ari else None
    )
    recovery_ok = all(
        rates[mechanism]
        >= (0.95 if mechanism == "null" else recovery_min)
        for mechanism in mechanisms
    )
    mixture_ok = (
        "mixture" not in mechanisms
        or (
            median_mixture_truth_ari is not None
            and median_mixture_truth_ari >= config["gates"]["min_ari"]
        )
    )
    scalar_ok = (
        "scalar" not in mechanisms
        or scalar_false_rejection
        <= config["simulation"]["scalar_max_false_positive"]
    )
    result = {
        "status": "ok" if recovery_ok and mixture_ok and scalar_ok else "failed",
        "mechanisms": mechanisms,
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
    _write_recovery_progress(
        config,
        "model_recovery",
        completed=total,
        total=total,
        status="complete",
    )
    return result
