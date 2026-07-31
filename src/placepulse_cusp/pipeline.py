from __future__ import annotations

import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mode
from typing import Any

import numpy as np
import polars as pl
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from placepulse_cusp.constants import DIMENSIONS
from placepulse_cusp.cusp.bimodality import conditional_bimodality
from placepulse_cusp.cusp.compare import compare_density_models
from placepulse_cusp.data.schema import standardise_votes
from placepulse_cusp.data.splits import grouped_edge_folds, prepare_data
from placepulse_cusp.data.validate import validate_votes
from placepulse_cusp.evaluation.gates import edge_predictive_gates, heterogeneity_verdict
from placepulse_cusp.evaluation.metrics import (
    bootstrap_reversal_evidence,
    clustered_elpd_bootstrap,
    cross_entropy,
    empirical_probabilities,
    log_score,
    ranking_reversal_fraction,
)
from placepulse_cusp.hardware import device_report
from placepulse_cusp.models import (
    ContinuousPreferenceModel,
    DavidsonModel,
    MixtureDavidsonModel,
)
from placepulse_cusp.models.base import VoteEncoder, select_device, set_deterministic
from placepulse_cusp.provenance import metadata, write_json, write_run_manifest
from placepulse_cusp.reporting.report import build_report, write_verdict
from placepulse_cusp.simulation.recovery import (
    validate_density_recovery,
    validate_model_recovery,
)

RESULT_SCHEMA_VERSION = 5


def _artifacts(config: dict[str, Any], kind: str) -> Path:
    path = Path(config["reporting"]["artifacts_dir"]) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fold_checkpoint_path(
    config: dict[str, Any], dimension: str, fold: int
) -> Path:
    config_hash = config.get("_meta", {}).get("hash", "unknown")[:16]
    input_hash = config.get("_runtime_input_hash", "unknown")[:12]
    return (
        _artifacts(config, "checkpoints")
        / (
            f"{dimension}_edge_fold_{fold}_{config_hash}_{input_hash}"
            f"_v{RESULT_SCHEMA_VERSION}.npz"
        )
    )


def _save_fold_checkpoint(
    path: Path,
    *,
    fold_metric: dict[str, Any],
    selection: dict[str, Any],
    scalar_scores: np.ndarray,
    m0_scores: np.ndarray,
    continuous_scores: np.ndarray,
    mixture_scores: np.ndarray,
    clusters: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        fold_metric=np.asarray(json.dumps(fold_metric)),
        selection=np.asarray(json.dumps(selection)),
        scalar_scores=scalar_scores,
        m0_scores=m0_scores,
        continuous_scores=continuous_scores,
        mixture_scores=mixture_scores,
        clusters=clusters.astype(str),
    )


def _load_fold_checkpoint(path: Path) -> dict[str, Any]:
    payload = np.load(path, allow_pickle=False)
    return {
        "fold_metric": json.loads(str(payload["fold_metric"])),
        "selection": json.loads(str(payload["selection"])),
        "scalar_scores": payload["scalar_scores"],
        "m0_scores": payload["m0_scores"],
        "continuous_scores": payload["continuous_scores"],
        "mixture_scores": payload["mixture_scores"],
        "clusters": payload["clusters"],
    }


def _eligible_test(train: pl.DataFrame, test: pl.DataFrame) -> pl.DataFrame:
    images = set(train["left_image_id"].to_list()) | set(train["right_image_id"].to_list())
    return test.filter(
        pl.col("left_image_id").is_in(images) & pl.col("right_image_id").is_in(images)
    )


def _choice_array(encoded) -> np.ndarray:
    return encoded.choice.detach().cpu().numpy()


def _cluster_array(frame: pl.DataFrame) -> np.ndarray:
    voter = frame["voter_id"].to_list()
    vote = frame["vote_id"].to_list()
    return np.asarray([v if v is not None else f"anonymous:{i}" for v, i in zip(voter, vote)])


def _training_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    model_cfg = config["models"]
    return {
        "epochs": model_cfg["epochs"],
        "learning_rate": model_cfg["learning_rate"],
        "patience": model_cfg["patience"],
        "batch_size": model_cfg.get("batch_size"),
        "lbfgs_steps": model_cfg.get("lbfgs_steps", 0),
    }


def _fit_scalar(
    train: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    *,
    baseline: dict[str, Any] | None = None,
):
    encoder = VoteEncoder().fit(train)
    encoded = encoder.transform(train, device)
    model_cfg = config["models"]
    baseline = baseline or {
        "name": "m1a",
        "utility_l2": model_cfg["l2_candidates"][0],
        "style_l2": model_cfg["l2_candidates"][0],
        "response_styles": False,
    }
    model = DavidsonModel.fit(
        encoded,
        utility_l2=baseline["utility_l2"],
        style_l2=baseline["style_l2"],
        response_styles=baseline["response_styles"],
        **_training_kwargs(config),
    )
    return encoder, model


def _fit_selected_models(
    train: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    *,
    baseline: dict[str, Any],
    rank: int,
    continuous_l2: float,
    classes: int,
    mixture_l2: float,
    seed_offset: int = 0,
) -> tuple[VoteEncoder, DavidsonModel, ContinuousPreferenceModel, MixtureDavidsonModel]:
    encoder = VoteEncoder().fit(train)
    encoded = encoder.transform(train, device)
    set_deterministic(config["project"]["seed"] + seed_offset)
    scalar = DavidsonModel.fit(
        encoded,
        utility_l2=baseline["utility_l2"],
        style_l2=baseline["style_l2"],
        response_styles=baseline["response_styles"],
        **_training_kwargs(config),
    )
    continuous = ContinuousPreferenceModel.fit(
        encoded,
        rank=rank,
        l2=continuous_l2,
        utility_l2=baseline["utility_l2"],
        style_l2=baseline["style_l2"],
        response_styles=baseline["response_styles"],
        **_training_kwargs(config),
    )
    mixture = MixtureDavidsonModel.fit(
        encoded,
        classes=classes,
        l2=mixture_l2,
        utility_l2=baseline["utility_l2"],
        style_l2=baseline["style_l2"],
        response_styles=baseline["response_styles"],
        dirichlet_alpha=config["models"]["dirichlet_alpha"],
        **_training_kwargs(config),
    )
    return encoder, scalar, continuous, mixture


def _evaluate_holdout(
    train: pl.DataFrame,
    test_all: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    *,
    baseline: dict[str, Any],
    rank: int,
    continuous_l2: float,
    classes: int,
    mixture_l2: float,
    seed_offset: int,
) -> dict[str, Any] | None:
    test = _eligible_test(train, test_all)
    if train.height < 20 or test.height < 5:
        return None
    encoder, scalar, continuous, mixture = _fit_selected_models(
        train,
        config,
        device,
        rank=rank,
        baseline=baseline,
        continuous_l2=continuous_l2,
        classes=classes,
        mixture_l2=mixture_l2,
        seed_offset=seed_offset,
    )
    encoded_test = encoder.transform(test, device)
    choices = _choice_array(encoded_test)
    scalar_scores = log_score(scalar.predict_proba(encoded_test), choices)
    continuous_scores = log_score(continuous.predict_proba(encoded_test), choices)
    mixture_scores = log_score(mixture.predict_proba(encoded_test), choices)
    return {
        "test_votes": test.height,
        "test_coverage": test.height / max(test_all.height, 1),
        "scalar_scores": scalar_scores,
        "continuous_scores": continuous_scores,
        "mixture_scores": mixture_scores,
        "clusters": _cluster_array(test),
    }


def _evaluate_auxiliary_holdouts(
    frame: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    *,
    baseline: dict[str, Any],
    rank: int,
    continuous_l2: float,
    classes: int,
    mixture_l2: float,
) -> dict[str, Any]:
    voter_results = []
    for fold in range(config["splits"]["outer_folds"]):
        result = _evaluate_holdout(
            frame.filter(pl.col("voter_fold") != fold),
            frame.filter(pl.col("voter_fold") == fold),
            config,
            device,
            rank=rank,
            baseline=baseline,
            continuous_l2=continuous_l2,
            classes=classes,
            mixture_l2=mixture_l2,
            seed_offset=100 + fold,
        )
        if result:
            voter_results.append(result)
    time_result = _evaluate_holdout(
        frame.filter(~pl.col("time_test")),
        frame.filter(pl.col("time_test")),
        config,
        device,
        rank=rank,
        baseline=baseline,
        continuous_l2=continuous_l2,
        classes=classes,
        mixture_l2=mixture_l2,
        seed_offset=200,
    )

    def summarise(results: list[dict[str, Any]], label: str) -> dict[str, Any]:
        if not results:
            return {"status": "not_evaluable"}
        scalar = np.concatenate([item["scalar_scores"] for item in results])
        continuous = np.concatenate([item["continuous_scores"] for item in results])
        mixture = np.concatenate([item["mixture_scores"] for item in results])
        clusters = np.concatenate([item["clusters"] for item in results])
        return {
            "status": "ok",
            "label": label,
            "test_votes": int(sum(item["test_votes"] for item in results)),
            "coverage": float(
                np.average(
                    [item["test_coverage"] for item in results],
                    weights=[item["test_votes"] for item in results],
                )
            ),
            "continuous_elpd_ci": clustered_elpd_bootstrap(
                continuous,
                scalar,
                clusters,
                repetitions=config["gates"]["elpd_bootstrap"],
                seed=config["project"]["seed"] + 301,
            ),
            "mixture_elpd_ci": clustered_elpd_bootstrap(
                mixture,
                scalar,
                clusters,
                repetitions=config["gates"]["elpd_bootstrap"],
                seed=config["project"]["seed"] + 302,
            ),
        }

    voter_summary = summarise(voter_results, "voter_holdout")
    time_summary = summarise([time_result] if time_result else [], "time_holdout")
    continuous_ok = (
        voter_summary.get("continuous_elpd_ci", {}).get("mean", float("-inf")) > 0
        and time_summary.get("continuous_elpd_ci", {}).get("upper", float("-inf")) >= 0
    )
    mixture_ok = (
        voter_summary.get("mixture_elpd_ci", {}).get("mean", float("-inf")) > 0
        and time_summary.get("mixture_elpd_ci", {}).get("upper", float("-inf")) >= 0
    )
    return {
        "voter_holdout": voter_summary,
        "time_holdout": time_summary,
        "continuous_auxiliary_gate": continuous_ok,
        "mixture_auxiliary_gate": mixture_ok,
    }


def _inner_edge_splits(
    train: pl.DataFrame, config: dict[str, Any], seed_offset: int
) -> list[tuple[pl.DataFrame, pl.DataFrame]]:
    folds = int(config["splits"]["inner_folds"])
    assigned = grouped_edge_folds(train, folds, config["project"]["seed"] + seed_offset)
    result = []
    for fold in range(folds):
        inner_train = train.filter(assigned != fold)
        validation = _eligible_test(inner_train, train.filter(assigned == fold))
        if inner_train.height >= 20 and validation.height >= 20:
            result.append((inner_train, validation))
    return result


def _selection_training_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    kwargs = _training_kwargs(config)
    kwargs["epochs"] = max(20, config["models"]["epochs"] // 2)
    # L-BFGS repeatedly evaluates a full-data closure and is intended for the
    # final selected specification. Adam is sufficient for coarse candidate
    # ranking; the winning model is still polished with the configured steps.
    kwargs["lbfgs_steps"] = 0
    return kwargs


def _screening_training_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Low-fidelity budget used only to form a candidate shortlist."""
    kwargs = _selection_training_kwargs(config)
    kwargs["epochs"] = max(5, kwargs["epochs"] // 3)
    return kwargs


def _encoded_inner_edge_splits(
    train: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    seed_offset: int,
) -> list[tuple[Any, Any]]:
    """Encode each inner split once and reuse it across all model candidates."""
    result = []
    for inner_train, validation in _inner_edge_splits(train, config, seed_offset):
        encoder = VoteEncoder().fit(inner_train)
        result.append(
            (encoder.transform(inner_train, device), encoder.transform(validation, device))
        )
    return result


def _select_scalar_baseline(
    train: pl.DataFrame, config: dict[str, Any], device: torch.device
) -> dict[str, Any]:
    encoded = _encoded_inner_edge_splits(train, config, device, 41)
    candidates = config["models"]["l2_candidates"]
    if not encoded:
        return {
            "name": "m1a",
            "utility_l2": candidates[0],
            "style_l2": candidates[0],
            "response_styles": False,
            "selection_boundary": True,
            "selection_boundary_parameters": ["inner_splits"],
        }
    m1a_scores: dict[float, list[float]] = {float(x): [] for x in candidates}
    for utility_l2 in candidates:
        for fold, (encoded_train, encoded_val) in enumerate(encoded):
            set_deterministic(config["project"]["seed"] + 410 + fold)
            model = DavidsonModel.fit(
                encoded_train,
                utility_l2=utility_l2,
                style_l2=utility_l2,
                response_styles=False,
                **_selection_training_kwargs(config),
            )
            m1a_scores[float(utility_l2)].append(
                cross_entropy(model.predict_proba(encoded_val), _choice_array(encoded_val))
            )
    utility_l2 = min(m1a_scores, key=lambda x: float(np.mean(m1a_scores[x])))
    a_scores = np.asarray(m1a_scores[utility_l2])
    m1b_scores: dict[float, list[float]] = {float(x): [] for x in candidates}
    for style_l2 in candidates:
        for fold, (encoded_train, encoded_val) in enumerate(encoded):
            set_deterministic(config["project"]["seed"] + 510 + fold)
            model = DavidsonModel.fit(
                encoded_train,
                utility_l2=utility_l2,
                style_l2=style_l2,
                response_styles=True,
                **_selection_training_kwargs(config),
            )
            m1b_scores[float(style_l2)].append(
                cross_entropy(model.predict_proba(encoded_val), _choice_array(encoded_val))
            )
    style_l2 = min(m1b_scores, key=lambda x: float(np.mean(m1b_scores[x])))
    b_scores = np.asarray(m1b_scores[style_l2])
    improvement = a_scores - b_scores
    standard_error = (
        float(improvement.std(ddof=1) / np.sqrt(len(improvement)))
        if len(improvement) > 1
        else 0.0
    )
    mean_improvement = float(improvement.mean())
    relative_improvement = mean_improvement / max(
        float(a_scores.mean()), 1e-12
    )
    statistical_gate = mean_improvement > standard_error
    practical_gate = (
        relative_improvement
        >= float(config["gates"]["min_cross_entropy_reduction"])
    )
    use_styles = statistical_gate and practical_gate
    values = [float(x) for x in candidates]
    high_regularisation_collapse = (
        len(values) > 1
        and statistical_gate
        and not practical_gate
        and style_l2 == max(values)
    )
    boundary_parameters = []
    if len(values) > 1:
        if utility_l2 in {min(values), max(values)}:
            boundary_parameters.append("utility_l2")
        if use_styles and style_l2 in {min(values), max(values)}:
            boundary_parameters.append("style_l2")
    boundary = bool(boundary_parameters)
    return {
        "name": "m1b" if use_styles else "m1a",
        "utility_l2": utility_l2,
        "style_l2": style_l2,
        "response_styles": use_styles,
        "m1b_improvement": mean_improvement,
        "m1b_improvement_se": standard_error,
        "m1b_relative_improvement": relative_improvement,
        "m1b_statistical_gate": statistical_gate,
        "m1b_practical_gate": practical_gate,
        "m1b_high_regularisation_collapse": high_regularisation_collapse,
        "selection_boundary": boundary,
        "selection_boundary_parameters": boundary_parameters,
    }


def _best_continuous_fit(
    encoded_train,
    config,
    baseline,
    rank,
    l2,
    seed,
    *,
    selection: bool = True,
    starts: int | None = None,
    screening: bool = False,
):
    best = None
    start_count = (
        config["models"].get("random_starts", 1) if starts is None else starts
    )
    start_count = max(1, int(start_count))
    training_kwargs = (
        _screening_training_kwargs(config)
        if screening
        else (
            _selection_training_kwargs(config)
            if selection
            else _training_kwargs(config)
        )
    )
    polish_best_only = (
        not selection
        and start_count > 1
        and int(training_kwargs.get("lbfgs_steps", 0)) > 0
    )
    fitting_kwargs = dict(training_kwargs)
    if polish_best_only:
        fitting_kwargs["lbfgs_steps"] = 0
    for start in range(start_count):
        set_deterministic(seed + start)
        model = ContinuousPreferenceModel.fit(
            encoded_train,
            rank=rank,
            l2=l2,
            utility_l2=baseline["utility_l2"],
            style_l2=baseline["style_l2"],
            response_styles=baseline["response_styles"],
            **fitting_kwargs,
        )
        if best is None or model.history[-1] < best.history[-1]:
            best = model
    if polish_best_only:
        polish_kwargs = dict(training_kwargs)
        polish_kwargs["epochs"] = 0
        set_deterministic(seed)
        best = ContinuousPreferenceModel.fit(
            encoded_train,
            rank=rank,
            l2=l2,
            utility_l2=baseline["utility_l2"],
            style_l2=baseline["style_l2"],
            response_styles=baseline["response_styles"],
            initial_state=best.module.state_dict(),
            **polish_kwargs,
        )
    return best


def _select_continuous(
    train: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    baseline: dict[str, Any],
) -> tuple[int, float]:
    encoded = _encoded_inner_edge_splits(train, config, device, 91)
    if not encoded:
        return config["models"]["continuous_dimensions"][0], config["models"]["l2_candidates"][0]
    candidates = [
        (rank, float(l2))
        for rank in config["models"]["continuous_dimensions"]
        for l2 in config["models"]["l2_candidates"]
    ]
    starts = max(1, int(config["models"].get("random_starts", 1)))

    def evaluate(candidate, candidate_starts: int, *, screening: bool = False) -> float:
        rank, l2 = candidate
        values = []
        for fold, (encoded_train, encoded_val) in enumerate(encoded):
            model = _best_continuous_fit(
                encoded_train,
                config,
                baseline,
                rank,
                l2,
                config["project"]["seed"] + 9100 + fold * 100,
                starts=candidate_starts,
                screening=screening,
            )
            values.append(
                cross_entropy(model.predict_proba(encoded_val), _choice_array(encoded_val))
            )
        return float(np.mean(values))

    # When the grid is larger than the requested number of starts, use one
    # deterministic start for every candidate, then spend the full multi-start
    # budget only on the best candidates. Small grids retain exhaustive search.
    if starts > 1 and len(candidates) > starts:
        screening_scores = {
            candidate: evaluate(candidate, 1, screening=True)
            for candidate in candidates
        }
        shortlist = sorted(screening_scores, key=screening_scores.get)[:starts]
        scores = {candidate: evaluate(candidate, starts) for candidate in shortlist}
    else:
        scores = {candidate: evaluate(candidate, starts) for candidate in candidates}
    return min(scores, key=scores.get)


def _best_mixture_fit(
    encoded_train,
    config,
    baseline,
    classes,
    l2,
    seed,
    *,
    selection: bool = True,
    starts: int | None = None,
    screening: bool = False,
):
    best = None
    start_count = (
        config["models"].get("random_starts", 1) if starts is None else starts
    )
    start_count = max(1, int(start_count))
    training_kwargs = (
        _screening_training_kwargs(config)
        if screening
        else (
            _selection_training_kwargs(config)
            if selection
            else _training_kwargs(config)
        )
    )
    polish_best_only = (
        not selection
        and start_count > 1
        and int(training_kwargs.get("lbfgs_steps", 0)) > 0
    )
    fitting_kwargs = dict(training_kwargs)
    if polish_best_only:
        fitting_kwargs["lbfgs_steps"] = 0
    for start in range(start_count):
        set_deterministic(seed + start)
        model = MixtureDavidsonModel.fit(
            encoded_train,
            classes=classes,
            l2=l2,
            utility_l2=baseline["utility_l2"],
            style_l2=baseline["style_l2"],
            response_styles=baseline["response_styles"],
            dirichlet_alpha=config["models"]["dirichlet_alpha"],
            **fitting_kwargs,
        )
        if best is None or model.history[-1] < best.history[-1]:
            best = model
    if polish_best_only:
        polish_kwargs = dict(training_kwargs)
        polish_kwargs["epochs"] = 0
        set_deterministic(seed)
        best = MixtureDavidsonModel.fit(
            encoded_train,
            classes=classes,
            l2=l2,
            utility_l2=baseline["utility_l2"],
            style_l2=baseline["style_l2"],
            response_styles=baseline["response_styles"],
            dirichlet_alpha=config["models"]["dirichlet_alpha"],
            initial_state=best.module.state_dict(),
            **polish_kwargs,
        )
    return best


def _select_mixture(
    train: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    baseline: dict[str, Any],
    *,
    return_diagnostics: bool = False,
) -> tuple[int, float] | tuple[int, float, dict[str, Any]]:
    encoded = _encoded_inner_edge_splits(train, config, device, 193)
    if not encoded:
        selected = (
            int(config["models"]["mixture_classes"][0]),
            float(config["models"]["l2_candidates"][0]),
        )
        if return_diagnostics:
            return *selected, {
                "selected_classes": selected[0],
                "selected_l2": selected[1],
                "screening": [],
                "refinement": [],
            }
        return selected
    candidates = [
        (classes, float(l2))
        for classes in config["models"]["mixture_classes"]
        for l2 in config["models"]["l2_candidates"]
    ]
    starts = max(1, int(config["models"].get("random_starts", 1)))

    def evaluate(
        candidate, candidate_starts: int, *, screening: bool = False
    ) -> dict[str, Any]:
        classes, l2 = candidate
        values = []
        for fold, (encoded_train, encoded_val) in enumerate(encoded):
            model = _best_mixture_fit(
                encoded_train,
                config,
                baseline,
                classes,
                l2,
                config["project"]["seed"] + 19300 + fold * 100,
                starts=candidate_starts,
                screening=screening,
            )
            values.append(
                cross_entropy(model.predict_proba(encoded_val), _choice_array(encoded_val))
            )
        scores = np.asarray(values, dtype=float)
        standard_error = (
            float(scores.std(ddof=1) / np.sqrt(len(scores)))
            if len(scores) > 1
            else 0.0
        )
        return {
            "classes": int(classes),
            "l2": float(l2),
            "mean_cross_entropy": float(scores.mean()),
            "standard_error": standard_error,
            "fold_cross_entropy": scores.tolist(),
        }

    if starts > 1 and len(candidates) > starts:
        screening_results = {
            candidate: evaluate(candidate, 1, screening=True)
            for candidate in candidates
        }
        shortlist = sorted(
            screening_results,
            key=lambda candidate: screening_results[candidate][
                "mean_cross_entropy"
            ],
        )[:starts]
        refinement_results = {
            candidate: evaluate(candidate, starts) for candidate in shortlist
        }
    else:
        screening_results = {}
        refinement_results = {
            candidate: evaluate(candidate, starts) for candidate in candidates
        }
    selected = min(
        refinement_results,
        key=lambda candidate: refinement_results[candidate][
            "mean_cross_entropy"
        ],
    )
    if return_diagnostics:
        return *selected, {
            "selected_classes": int(selected[0]),
            "selected_l2": float(selected[1]),
            "screening": list(screening_results.values()),
            "refinement": list(refinement_results.values()),
        }
    return selected


def run_dimension(
    config: dict[str, Any],
    dimension: str,
    *,
    detailed: bool = False,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completed_path = _artifacts(config, "metrics") / f"{dimension}_model_comparison.json"
    input_paths = [
        Path(config["data"]["processed_dir"]) / "votes.parquet",
        Path(config["data"]["processed_dir"]) / "splits.parquet",
        Path(config["data"]["processed_dir"]) / "data_validation.json",
    ]
    current_provenance = metadata(config, input_paths)
    config["_runtime_input_hash"] = hashlib.sha256(
        json.dumps(current_provenance["inputs"], sort_keys=True).encode("utf-8")
    ).hexdigest()
    if resume and completed_path.exists():
        completed = json.loads(completed_path.read_text("utf-8"))
        if (
            completed.get("result_schema_version") == RESULT_SCHEMA_VERSION
            and completed.get("provenance", {}).get("config_hash")
            == config.get("_meta", {}).get("hash")
            and completed.get("provenance", {}).get("inputs")
            == current_provenance["inputs"]
        ):
            print(
                f"[resume] {dimension}: loading completed result",
                file=sys.stderr,
                flush=True,
            )
            return completed, {}
    votes = pl.read_parquet(Path(config["data"]["processed_dir"]) / "votes.parquet")
    splits = pl.read_parquet(Path(config["data"]["processed_dir"]) / "splits.parquet")
    frame = votes.join(splits, on="vote_id").filter(pl.col("dimension") == dimension)
    device = select_device(config["project"]["device"])
    if (
        device.type == "cuda"
        and config["project"].get("deterministic", False)
        and os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}
    ):
        raise RuntimeError(
            "Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 "
            "(or :16:8) before Python starts."
        )
    compute = device_report(config["project"]["device"])
    print(
        f"[compute] {dimension}: requested={compute['requested']} "
        f"selected={compute['selected']} torch={compute['python_torch']}",
        file=sys.stderr,
        flush=True,
    )
    fold_metrics = []
    selections = []
    all_m0_scores, all_scalar_scores, all_cont_scores, all_mix_scores, all_clusters = (
        [],
        [],
        [],
        [],
        [],
    )
    for fold in range(config["splits"]["outer_folds"]):
        checkpoint = _fold_checkpoint_path(config, dimension, fold)
        if resume and checkpoint.exists():
            cached = _load_fold_checkpoint(checkpoint)
            fold_metrics.append(cached["fold_metric"])
            selections.append(cached["selection"])
            all_scalar_scores.append(cached["scalar_scores"])
            all_m0_scores.append(cached["m0_scores"])
            all_cont_scores.append(cached["continuous_scores"])
            all_mix_scores.append(cached["mixture_scores"])
            all_clusters.append(cached["clusters"])
            print(
                f"[resume] {dimension}: loaded edge fold {fold}",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"[fit] {dimension}: selecting and fitting edge fold {fold}",
            file=sys.stderr,
            flush=True,
        )
        train = frame.filter(pl.col("edge_fold") != fold)
        test_all = frame.filter(pl.col("edge_fold") == fold)
        test = _eligible_test(train, test_all)
        if train.height < 20 or test.height < 5:
            continue
        baseline = _select_scalar_baseline(train, config, device)
        selected_rank, selected_l2 = _select_continuous(
            train, config, device, baseline
        )
        selected_classes, mixture_l2 = _select_mixture(
            train, config, device, baseline
        )
        candidate_values = [float(x) for x in config["models"]["l2_candidates"]]
        continuous_boundary = len(candidate_values) > 1 and selected_l2 in {
            min(candidate_values),
            max(candidate_values),
        }
        mixture_boundary = len(candidate_values) > 1 and mixture_l2 in {
            min(candidate_values),
            max(candidate_values),
        }
        selections.append(
            {
                "fold": fold,
                "baseline": baseline,
                "continuous_rank": selected_rank,
                "continuous_l2": selected_l2,
                "mixture_classes": selected_classes,
                "mixture_l2": mixture_l2,
                "continuous_selection_boundary": continuous_boundary,
                "mixture_selection_boundary": mixture_boundary,
            }
        )
        encoder = VoteEncoder().fit(train)
        encoded_train = encoder.transform(train, device)
        encoded_test = encoder.transform(test, device)
        set_deterministic(config["project"]["seed"] + fold)
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
            selected_rank,
            selected_l2,
            config["project"]["seed"] + fold * 1000,
            selection=False,
        )
        mixture = _best_mixture_fit(
            encoded_train,
            config,
            baseline,
            selected_classes,
            mixture_l2,
            config["project"]["seed"] + 50000 + fold * 1000,
            selection=False,
        )
        choices = _choice_array(encoded_test)
        scalar_p = scalar.predict_proba(encoded_test)
        continuous_p = continuous.predict_proba(encoded_test)
        mixture_p = mixture.predict_proba(encoded_test)
        scalar_population_p = scalar.predict_proba(encoded_test, population=True)
        continuous_population_p = continuous.predict_proba(
            encoded_test, population=True
        )
        mixture_population_p = mixture.predict_proba(encoded_test, population=True)
        empirical = empirical_probabilities(_choice_array(encoded_train))
        history_counts = {
            voter: count
            for voter, count in train.filter(pl.col("voter_id").is_not_null())
            .group_by("voter_id")
            .len()
            .iter_rows()
        }
        test_history = np.asarray(
            [history_counts.get(voter, 0) for voter in test["voter_id"].to_list()]
        )
        history_diagnostics = {}
        for label, mask in {
            "0_4": test_history < 5,
            "5_19": (test_history >= 5) & (test_history < 20),
            "20_plus": test_history >= 20,
        }.items():
            if mask.any():
                history_diagnostics[label] = {
                    "votes": int(mask.sum()),
                    "continuous_elpd_vs_scalar": float(
                        (
                            log_score(continuous_p[mask], choices[mask])
                            - log_score(scalar_p[mask], choices[mask])
                        ).mean()
                    ),
                    "mixture_elpd_vs_scalar": float(
                        (
                            log_score(mixture_p[mask], choices[mask])
                            - log_score(scalar_p[mask], choices[mask])
                        ).mean()
                    ),
                }
        fold_metric = {
            "fold": fold,
            "train_votes": train.height,
            "test_votes": test.height,
            "test_coverage": test.height / max(test_all.height, 1),
            "selected_rank": selected_rank,
            "selected_classes": selected_classes,
            "baseline": baseline,
            "continuous_l2": selected_l2,
            "mixture_l2": mixture_l2,
            "m0_cross_entropy": cross_entropy(
                np.repeat(empirical[None, :], len(choices), axis=0), choices
            ),
            "scalar_cross_entropy": cross_entropy(scalar_p, choices),
            "continuous_cross_entropy": cross_entropy(continuous_p, choices),
            "mixture_cross_entropy": cross_entropy(mixture_p, choices),
            "population_marginalized_cross_entropy": {
                "scalar": cross_entropy(scalar_population_p, choices),
                "continuous": cross_entropy(continuous_population_p, choices),
                "mixture": cross_entropy(mixture_population_p, choices),
            },
            "history_strata": history_diagnostics,
        }
        scalar_scores = log_score(scalar_p, choices)
        m0_p = np.repeat(empirical[None, :], len(choices), axis=0)
        m0_scores = log_score(m0_p, choices)
        continuous_scores = log_score(continuous_p, choices)
        mixture_scores = log_score(mixture_p, choices)
        clusters = _cluster_array(test)
        fold_metrics.append(fold_metric)
        all_scalar_scores.append(scalar_scores)
        all_m0_scores.append(m0_scores)
        all_cont_scores.append(continuous_scores)
        all_mix_scores.append(mixture_scores)
        all_clusters.append(clusters)
        _save_fold_checkpoint(
            checkpoint,
            fold_metric=fold_metric,
            selection=selections[-1],
            scalar_scores=scalar_scores,
            m0_scores=m0_scores,
            continuous_scores=continuous_scores,
            mixture_scores=mixture_scores,
            clusters=clusters,
        )
        print(
            f"[checkpoint] {dimension}: saved edge fold {fold}",
            file=sys.stderr,
            flush=True,
        )
    if not fold_metrics:
        raise RuntimeError(f"No evaluable outer folds for {dimension}")
    selected_rank = mode(item["continuous_rank"] for item in selections)
    selected_classes = mode(item["mixture_classes"] for item in selections)
    baseline_name = mode(item["baseline"]["name"] for item in selections)
    matching_baselines = [
        item["baseline"] for item in selections if item["baseline"]["name"] == baseline_name
    ]
    baseline = {
        "name": baseline_name,
        "utility_l2": mode(item["utility_l2"] for item in matching_baselines),
        "style_l2": mode(item["style_l2"] for item in matching_baselines),
        "response_styles": baseline_name == "m1b",
        "selection_boundary": any(
            item.get("selection_boundary", False) for item in matching_baselines
        )
        or any(
            item.get("continuous_selection_boundary", False)
            or item.get("mixture_selection_boundary", False)
            for item in selections
        ),
    }
    selected_l2 = mode(
        item["continuous_l2"]
        for item in selections
        if item["continuous_rank"] == selected_rank
    )
    mixture_l2 = mode(
        item["mixture_l2"]
        for item in selections
        if item["mixture_classes"] == selected_classes
    )
    m0_scores = np.concatenate(all_m0_scores)
    scalar_scores = np.concatenate(all_scalar_scores)
    continuous_scores = np.concatenate(all_cont_scores)
    mixture_scores = np.concatenate(all_mix_scores)
    clusters = np.concatenate(all_clusters)
    scalar_ce = float(-scalar_scores.mean())
    m0_ce = float(-m0_scores.mean())
    continuous_ce = float(-continuous_scores.mean())
    mixture_ce = float(-mixture_scores.mean())
    continuous_ci = clustered_elpd_bootstrap(
        continuous_scores,
        scalar_scores,
        clusters,
        repetitions=config["gates"]["elpd_bootstrap"],
        seed=config["project"]["seed"],
    )
    mixture_ci = clustered_elpd_bootstrap(
        mixture_scores,
        scalar_scores,
        clusters,
        repetitions=config["gates"]["elpd_bootstrap"],
        seed=config["project"]["seed"] + 1,
    )
    scalar_vs_m0_ci = clustered_elpd_bootstrap(
        scalar_scores,
        m0_scores,
        clusters,
        repetitions=config["gates"]["elpd_bootstrap"],
        seed=config["project"]["seed"] + 2,
    )
    edge_gates = edge_predictive_gates(
        config,
        m0_ce,
        scalar_ce,
        continuous_ce,
        mixture_ce,
        continuous_ci,
        mixture_ci,
        scalar_vs_m0_ci,
        simulation_ok=config.get("_simulation_ok", True),
        selection_boundary=baseline["selection_boundary"],
    )
    # Refit the selected specification on all observations only after every
    # outer-fold score has been frozen. These models produce descriptive tables,
    # never outer-fold predictive metrics.
    final_encoder = VoteEncoder().fit(frame)
    final_encoded = final_encoder.transform(frame, device)
    set_deterministic(config["project"]["seed"])
    final_scalar = DavidsonModel.fit(
        final_encoded,
        utility_l2=baseline["utility_l2"],
        style_l2=baseline["style_l2"],
        response_styles=baseline["response_styles"],
        **_training_kwargs(config),
    )
    final_continuous = _best_continuous_fit(
        final_encoded,
        config,
        baseline,
        selected_rank,
        selected_l2,
        config["project"]["seed"] + 60000,
        selection=False,
    )
    mixture = _best_mixture_fit(
        final_encoded,
        config,
        baseline,
        selected_classes,
        mixture_l2,
        config["project"]["seed"] + 70000,
        selection=False,
    )
    final_models = {
        "encoder": final_encoder,
        "scalar": final_scalar,
        "continuous": final_continuous,
        "mixture": mixture,
        "train": frame,
    }
    raw_reversal = ranking_reversal_fraction(mixture.utilities())
    if edge_gates["baseline"] and edge_gates["mixture"]:
        stability, stability_models = _mixture_stability(
            frame, selected_classes, mixture_l2, baseline, config, device
        )
        image_counts_lookup = {}
        for column in ("left_image_id", "right_image_id"):
            for image, count in frame.group_by(column).len().iter_rows():
                image_counts_lookup[image] = image_counts_lookup.get(image, 0) + count
        image_counts = np.asarray(
            [image_counts_lookup.get(image, 0) for image in final_encoder.image_ids]
        )
        reversal_evidence = bootstrap_reversal_evidence(
            stability_models,
            image_counts,
            min_image_votes=config["gates"].get("min_image_votes", 20),
            min_probability=config["gates"]["min_reversal_probability"],
            min_standardized_gap=config["gates"].get(
                "min_standardized_utility_gap", 0.5
            ),
            seed=config["project"]["seed"],
        )
    else:
        stability, stability_models = 0.0, []
        reversal_evidence = {
            "fraction": 0.0,
            "eligible_pairs": 0,
            "reliable_reversals": 0,
            "status": "skipped",
            "reason": "mixture_edge_gate_failed",
        }
    reversal = float(reversal_evidence["fraction"])
    if edge_gates["baseline"] and (
        edge_gates["continuous"] or edge_gates["mixture"]
    ):
        auxiliary = _evaluate_auxiliary_holdouts(
            frame,
            config,
            device,
            baseline=baseline,
            rank=selected_rank,
            continuous_l2=selected_l2,
            classes=selected_classes,
            mixture_l2=mixture_l2,
        )
    else:
        auxiliary = {
            "status": "skipped",
            "reason": "heterogeneity_edge_gates_failed",
            "voter_holdout": {"status": "skipped"},
            "time_holdout": {"status": "skipped"},
            "continuous_auxiliary_gate": False,
            "mixture_auxiliary_gate": False,
        }
    verdict, gates = heterogeneity_verdict(
        config,
        m0_ce,
        scalar_ce,
        continuous_ce,
        mixture_ce,
        continuous_ci,
        mixture_ci,
        scalar_vs_m0_ci,
        mixture.class_weights().tolist(),
        reversal,
        stability,
        auxiliary["continuous_auxiliary_gate"],
        auxiliary["mixture_auxiliary_gate"],
        simulation_ok=config.get("_simulation_ok", True),
        selection_boundary=baseline["selection_boundary"],
    )
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dimension": dimension,
        "verdict": verdict,
        "selected_rank": selected_rank,
        "selected_classes": selected_classes,
        "selected_baseline": baseline,
        "outer_fold_selections": selections,
        "folds": fold_metrics,
        "m0_cross_entropy": m0_ce,
        "scalar_cross_entropy": scalar_ce,
        "scalar_vs_m0_elpd_ci": scalar_vs_m0_ci,
        "continuous_cross_entropy": continuous_ce,
        "mixture_cross_entropy": mixture_ce,
        "continuous_elpd_ci": continuous_ci,
        "mixture_elpd_ci": mixture_ci,
        "class_weights": mixture.class_weights().tolist(),
        "ranking_reversal_fraction": reversal,
        "raw_ranking_reversal_fraction": raw_reversal,
        "ranking_reversal_evidence": reversal_evidence,
        "stability_ari": stability,
        "stability_refits_completed": len(stability_models),
        "auxiliary_holdouts": auxiliary,
        "gates": gates,
        "provenance": {
            **current_provenance,
            "compute": compute,
        },
    }
    write_json(completed_path, result)
    if detailed:
        _save_model_tables(config, dimension, final_models)
    return result, final_models


def _mixture_stability(
    train: pl.DataFrame,
    classes: int,
    l2: float,
    baseline: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    *,
    refits: int | None = None,
) -> tuple[float, list[np.ndarray]]:
    identified = train.filter(pl.col("voter_id").is_not_null())
    voter_ids = identified["voter_id"].unique().sort().to_list()
    if len(voter_ids) < 2:
        return 0.0, []
    image_ids = sorted(
        set(train["left_image_id"].to_list()) | set(train["right_image_id"].to_list())
    )
    reference_encoder = VoteEncoder(image_ids=image_ids, voter_ids=voter_ids)
    reference = reference_encoder.transform(identified, device)
    rng = np.random.default_rng(config["project"]["seed"] + 88000)
    labels = []
    utilities = []
    refit_count = max(
        1,
        int(config["gates"]["stability_refits"] if refits is None else refits),
    )
    for index in range(refit_count):
        sampled = rng.choice(voter_ids, len(voter_ids), replace=True)
        unique, counts = np.unique(sampled, return_counts=True)
        multiplicity = pl.DataFrame(
            {"voter_id": unique.tolist(), "_bootstrap_count": counts.tolist()}
        )
        bootstrap = (
            identified.join(multiplicity, on="voter_id", how="inner")
            .with_columns(pl.int_ranges(0, pl.col("_bootstrap_count")).alias("_copy"))
            .explode("_copy")
            .with_columns(
                (
                    pl.col("voter_id").cast(pl.String)
                    + pl.lit("#")
                    + pl.col("_copy").cast(pl.String)
                ).alias("voter_id")
            )
            .drop("_bootstrap_count", "_copy")
        )
        bootstrap_voters = bootstrap["voter_id"].unique().sort().to_list()
        encoder = VoteEncoder(image_ids=image_ids, voter_ids=bootstrap_voters)
        encoded = encoder.transform(bootstrap, device)
        set_deterministic(
            config["project"]["seeds"][index % len(config["project"]["seeds"])]
        )
        model = MixtureDavidsonModel.fit(
            encoded,
            classes=classes,
            l2=l2,
            utility_l2=baseline["utility_l2"],
            style_l2=baseline["style_l2"],
            response_styles=baseline["response_styles"],
            dirichlet_alpha=config["models"]["dirichlet_alpha"],
            **_selection_training_kwargs(config),
        )
        posterior = model.infer_posterior(
            reference, n_voters=len(voter_ids), use_response_styles=False
        )
        labels.append(posterior.argmax(1))
        utilities.append(model.utilities())
    if len(labels) < 2:
        return 1.0, utilities
    values = [
        adjusted_rand_score(labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    return float(np.median(values)), utilities


def _save_model_tables(config: dict[str, Any], dimension: str, models: dict[str, Any]) -> None:
    tables = _artifacts(config, "tables")
    encoder = models["encoder"]
    scalar = models["scalar"]
    mixture = models["mixture"]
    pl.DataFrame(
        {"image_id": encoder.image_ids, "utility": scalar.utilities()}
    ).write_csv(tables / f"{dimension}_scalar_scores.csv")
    utility = mixture.utilities()
    payload = {"image_id": encoder.image_ids}
    for index in range(utility.shape[0]):
        payload[f"class_{index}_utility"] = utility[index]
    pl.DataFrame(payload).write_csv(tables / f"{dimension}_class_specific_scores.csv")
    posterior = mixture.posterior
    voter_payload = {"voter_id": encoder.voter_ids}
    for index in range(posterior.shape[1]):
        voter_payload[f"class_{index}_probability"] = posterior[:, index]
    pl.DataFrame(voter_payload).write_csv(tables / f"{dimension}_latent_classes.csv")


def _cross_fitted_mixture_utilities(
    frame: pl.DataFrame,
    classes: int,
    l2: float,
    baseline: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    fitted: list[tuple[list[str], np.ndarray]] = []
    reference = None
    for fold in range(config["splits"]["outer_folds"]):
        train = frame.filter(pl.col("edge_fold") != fold)
        if train.height < 20:
            continue
        encoder = VoteEncoder().fit(train)
        encoded = encoder.transform(train, device)
        model = _best_mixture_fit(
            encoded,
            config,
            baseline,
            classes,
            l2,
            config["project"]["seed"] + 90000 + fold * 100,
            selection=False,
        )
        values = model.utilities()
        if reference is None:
            reference = values
        else:
            common = sorted(set(encoder.image_ids) & set(fitted[0][0]))
            current_lookup = {image: i for i, image in enumerate(encoder.image_ids)}
            reference_lookup = {image: i for i, image in enumerate(fitted[0][0])}
            current_common = values[:, [current_lookup[x] for x in common]]
            reference_common = reference[:, [reference_lookup[x] for x in common]]
            correlation = np.nan_to_num(
                np.corrcoef(current_common, reference_common)[:classes, classes:]
            )
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-correlation)
            aligned = np.empty_like(values)
            aligned[cols] = values[rows]
            values = aligned
        fitted.append((encoder.image_ids, values))
    if not fitted:
        raise RuntimeError("No evaluable cross-fitted mixture folds")
    common_images = sorted(set.intersection(*(set(images) for images, _ in fitted)))
    fold_values = []
    for images, values in fitted:
        lookup = {image: i for i, image in enumerate(images)}
        fold_values.append(values[:, [lookup[image] for image in common_images]])
    return common_images, np.mean(fold_values, axis=0)


def run_cusp_stage(
    config: dict[str, Any],
    primary_result: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if primary_result["verdict"] != "SCALAR_REJECTED_MIXTURE":
        return primary_result["verdict"], {"status": "skipped", "reason": "mixture_gate_failed"}
    votes = pl.read_parquet(Path(config["data"]["processed_dir"]) / "votes.parquet")
    splits = pl.read_parquet(Path(config["data"]["processed_dir"]) / "splits.parquet")
    votes = votes.join(splits, on="vote_id")
    device = select_device(config["project"]["device"])
    fitted = {}
    for dimension in DIMENSIONS:
        frame = votes.filter(pl.col("dimension") == dimension)
        metrics_path = _artifacts(config, "metrics") / f"{dimension}_model_comparison.json"
        if not metrics_path.exists():
            return "SCALAR_REJECTED_MIXTURE", {
                "status": "skipped",
                "reason": f"missing_cross_fitted_metrics:{dimension}",
            }
        dimension_result = json.loads(metrics_path.read_text("utf-8"))
        classes = dimension_result["selected_classes"]
        selections = dimension_result["outer_fold_selections"]
        mixture_l2 = mode(
            item["mixture_l2"]
            for item in selections
            if item["mixture_classes"] == classes
        )
        baseline = dimension_result["selected_baseline"]
        fitted[dimension] = _cross_fitted_mixture_utilities(
            frame, classes, mixture_l2, baseline, config, device
        )
    common_images = set(fitted["safety"][0])
    for images, _ in fitted.values():
        common_images &= set(images)
    common = sorted(common_images)
    if len(common) < config["bimodality"]["min_neighbors"]:
        return "SCALAR_REJECTED_MIXTURE", {
            "status": "skipped",
            "reason": "insufficient_cross_dimension_image_overlap",
            "images": len(common),
        }
    utilities = {}
    for dimension, (images, values) in fitted.items():
        lookup = {image: i for i, image in enumerate(images)}
        utilities[dimension] = values[:, [lookup[image] for image in common]]
    other = [d for d in DIMENSIONS if d != "safety"]
    mean_features, variance_features = [], []
    for dimension in other:
        values = utilities[dimension]
        direction = -1.0 if dimension in {"boring", "depressing"} else 1.0
        mean_features.append(direction * values.mean(0))
        variance_features.append(values.var(0))
    mean_matrix = StandardScaler().fit_transform(np.column_stack(mean_features))
    variance_matrix = StandardScaler().fit_transform(np.column_stack(variance_features))
    alpha = PCA(1, random_state=config["project"]["seed"]).fit_transform(mean_matrix).ravel()
    beta = PCA(1, random_state=config["project"]["seed"]).fit_transform(variance_matrix).ravel()
    safety = utilities["safety"]
    classes = safety.shape[0]
    y = safety.ravel()
    alpha_rows = np.tile(alpha, classes)
    beta_rows = np.tile(beta, classes)
    y = (y - y.mean()) / max(y.std(), 1e-8)
    results, passes = conditional_bimodality(
        y,
        beta_rows,
        bins=config["bimodality"]["quantile_bins"],
        min_neighbors=config["bimodality"]["min_neighbors"],
        fdr=config["bimodality"]["fdr"],
        min_component_weight=config["bimodality"]["min_component_weight"],
        min_ashman_d=config["bimodality"]["min_ashman_d"],
        min_adjacent_windows=config["bimodality"]["min_adjacent_windows"],
        seed=config["project"]["seed"],
    )
    bimodality_payload = {
        "passes": passes,
        "windows": [asdict(item) for item in results],
        "images": len(common),
    }
    write_json(_artifacts(config, "metrics") / "bimodality.json", bimodality_payload)
    if not passes:
        return "SCALAR_REJECTED_MIXTURE", bimodality_payload
    rng = np.random.default_rng(config["project"]["seed"])
    image_test = rng.random(len(common)) >= 0.8
    row_test = np.tile(image_test, classes)
    x = np.column_stack([alpha_rows, beta_rows])
    density_metrics, _ = compare_density_models(
        config, x[~row_test], y[~row_test], x[row_test], y[row_test]
    )
    comparison = density_metrics["comparison"]
    cusp_cfg = config["cusp"]
    cusp_wins = (
        comparison["cusp_improvement"] > 0
        and comparison["relative_cross_entropy_reduction"]
        >= config["gates"]["min_cross_entropy_reduction"]
        and comparison["fold_fraction"] > 0
        and comparison["integration_error"] < cusp_cfg["integration_tolerance"]
    )
    # A confirmatory CUSP verdict requires city metadata for leave-one-city-out stability.
    city_available = "city_left" in votes.columns and "city_right" in votes.columns
    verdict = "CUSP_COMPATIBLE" if cusp_wins and city_available else "BIMODAL_NON_CUSP"
    density_metrics["comparison"]["city_robustness_available"] = city_available
    density_metrics["comparison"]["confirmatory_cusp_gate"] = cusp_wins and city_available
    write_json(_artifacts(config, "metrics") / "density_model_comparison.json", density_metrics)
    return verdict, {"bimodality": bimodality_payload, "density": density_metrics}


def run_all(config: dict[str, Any], *, resume: bool = False) -> dict[str, Any]:
    try:
        standardise_votes(config)
    except (FileNotFoundError, ValueError) as error:
        write_verdict(config, "DATA_INSUFFICIENT", reasons=[str(error)])
        build_report(config)
        return {"verdict": "DATA_INSUFFICIENT", "reasons": [str(error)]}
    validation = validate_votes(config)
    if validation["status"] != "ok":
        write_verdict(config, "DATA_INSUFFICIENT", reasons=validation["reasons"])
        build_report(config)
        return {"verdict": "DATA_INSUFFICIENT", "reasons": validation["reasons"]}
    prepare_data(config)
    write_run_manifest(
        config,
        [
            config["data"]["votes_file"],
            Path(config["data"]["processed_dir"]) / "votes.parquet",
            Path(config["data"]["processed_dir"]) / "splits.parquet",
            Path(config["data"]["processed_dir"]) / "data_validation.json",
        ],
    )
    calibration_root = config["simulation"].get("calibration_artifacts")
    calibration_metrics = (
        Path(calibration_root) if calibration_root else _artifacts(config, "metrics")
    )
    model_recovery_path = calibration_metrics / "model_recovery.json"
    density_recovery_path = calibration_metrics / "simulation_recovery.json"
    if calibration_root and (
        not model_recovery_path.exists() or not density_recovery_path.exists()
    ):
        raise RuntimeError(
            f"Frozen calibration artifacts are missing from {calibration_metrics}"
        )
    model_simulation = (
        json.loads(model_recovery_path.read_text("utf-8"))
        if (resume or calibration_root) and model_recovery_path.exists()
        else validate_model_recovery(config, resume=resume)
    )
    density_simulation = (
        json.loads(density_recovery_path.read_text("utf-8"))
        if (resume or calibration_root) and density_recovery_path.exists()
        else validate_density_recovery(config, resume=resume)
    )
    simulation = {
        "model_recovery": model_simulation,
        "density_recovery": density_simulation,
    }
    simulation_ok = (
        model_simulation["status"] == "ok" and density_simulation["status"] == "ok"
    )
    config["_simulation_ok"] = simulation_ok
    if not simulation_ok:
        write_verdict(
            config,
            "MODEL_CALIBRATION_FAILED",
            reasons=["simulation_calibration_failed"],
            metrics=simulation,
        )
        build_report(config)
        return {"verdict": "MODEL_CALIBRATION_FAILED", "simulation": simulation}
    primary, _ = run_dimension(
        config,
        config["data"]["primary_dimension"],
        detailed=True,
        resume=resume,
    )
    replication = {}
    for dimension in config["data"]["dimensions"]:
        if dimension == config["data"]["primary_dimension"]:
            continue
        try:
            replication[dimension] = run_dimension(
                config, dimension, detailed=False, resume=resume
            )[0]
        except RuntimeError as error:
            replication[dimension] = {"status": "not_evaluable", "reason": str(error)}
    write_json(_artifacts(config, "metrics") / "replication_dimensions.json", replication)
    verdict, cusp_metrics = run_cusp_stage(config, primary)
    write_verdict(
        config,
        verdict,
        gates=primary["gates"],
        metrics={"primary": primary, "cusp": cusp_metrics, "replication": replication},
    )
    build_report(config)
    return {"verdict": verdict, "primary": primary, "cusp": cusp_metrics}
