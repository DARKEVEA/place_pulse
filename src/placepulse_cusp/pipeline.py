from __future__ import annotations

import json
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
from placepulse_cusp.data.splits import prepare_data
from placepulse_cusp.data.validate import validate_votes
from placepulse_cusp.evaluation.gates import heterogeneity_verdict
from placepulse_cusp.evaluation.metrics import (
    clustered_elpd_bootstrap,
    cross_entropy,
    empirical_probabilities,
    log_score,
    ranking_reversal_fraction,
)
from placepulse_cusp.models import (
    ContinuousPreferenceModel,
    DavidsonModel,
    MixtureDavidsonModel,
)
from placepulse_cusp.models.base import VoteEncoder, select_device, set_deterministic
from placepulse_cusp.provenance import metadata, write_json
from placepulse_cusp.reporting.report import build_report, write_verdict
from placepulse_cusp.simulation.recovery import validate_density_recovery

RESULT_SCHEMA_VERSION = 2


def _artifacts(config: dict[str, Any], kind: str) -> Path:
    path = Path(config["reporting"]["artifacts_dir"]) / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def _fold_checkpoint_path(
    config: dict[str, Any], dimension: str, fold: int
) -> Path:
    config_hash = config.get("_meta", {}).get("hash", "unknown")[:16]
    return (
        _artifacts(config, "checkpoints")
        / f"{dimension}_edge_fold_{fold}_{config_hash}_v{RESULT_SCHEMA_VERSION}.npz"
    )


def _save_fold_checkpoint(
    path: Path,
    *,
    fold_metric: dict[str, Any],
    selection: dict[str, Any],
    scalar_scores: np.ndarray,
    continuous_scores: np.ndarray,
    mixture_scores: np.ndarray,
    clusters: np.ndarray,
) -> None:
    np.savez_compressed(
        path,
        fold_metric=np.asarray(json.dumps(fold_metric)),
        selection=np.asarray(json.dumps(selection)),
        scalar_scores=scalar_scores,
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


def _fit_scalar(train: pl.DataFrame, config: dict[str, Any], device: torch.device):
    encoder = VoteEncoder().fit(train)
    encoded = encoder.transform(train, device)
    model_cfg = config["models"]
    model = DavidsonModel.fit(
        encoded,
        l2=model_cfg["l2_candidates"][0],
        epochs=model_cfg["epochs"],
        learning_rate=model_cfg["learning_rate"],
        patience=model_cfg["patience"],
    )
    return encoder, model


def _fit_selected_models(
    train: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    *,
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
        l2=continuous_l2,
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    continuous = ContinuousPreferenceModel.fit(
        encoded,
        rank=rank,
        l2=continuous_l2,
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    mixture = MixtureDavidsonModel.fit(
        encoded,
        classes=classes,
        l2=mixture_l2,
        dirichlet_alpha=config["models"]["dirichlet_alpha"],
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    return encoder, scalar, continuous, mixture


def _evaluate_holdout(
    train: pl.DataFrame,
    test_all: pl.DataFrame,
    config: dict[str, Any],
    device: torch.device,
    *,
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


def _select_continuous(
    train: pl.DataFrame, config: dict[str, Any], device: torch.device
) -> tuple[int, float]:
    rng = np.random.default_rng(config["project"]["seed"] + 91)
    mask = rng.random(train.height) < 0.8
    inner_train, validation = train.filter(mask), train.filter(~mask)
    validation = _eligible_test(inner_train, validation)
    if validation.height < 20:
        return config["models"]["continuous_dimensions"][0], config["models"]["l2_candidates"][0]
    encoder = VoteEncoder().fit(inner_train)
    encoded_train = encoder.transform(inner_train, device)
    encoded_val = encoder.transform(validation, device)
    best = (float("inf"), 1, 1e-3)
    for rank in config["models"]["continuous_dimensions"]:
        for l2 in config["models"]["l2_candidates"]:
            set_deterministic(config["project"]["seed"] + rank)
            model = ContinuousPreferenceModel.fit(
                encoded_train,
                rank=rank,
                l2=l2,
                epochs=max(20, config["models"]["epochs"] // 2),
                learning_rate=config["models"]["learning_rate"],
                patience=config["models"]["patience"],
            )
            score = cross_entropy(model.predict_proba(encoded_val), _choice_array(encoded_val))
            if score < best[0]:
                best = score, rank, l2
    return best[1], best[2]


def _select_mixture(
    train: pl.DataFrame, config: dict[str, Any], device: torch.device
) -> tuple[int, float]:
    rng = np.random.default_rng(config["project"]["seed"] + 193)
    voters = np.asarray(train["voter_id"].fill_null("__anonymous__").to_list())
    unique = np.unique(voters)
    val_voters = set(rng.choice(unique, max(1, len(unique) // 5), replace=False))
    mask = np.asarray([v not in val_voters for v in voters])
    inner_train, validation = train.filter(mask), train.filter(~mask)
    validation = _eligible_test(inner_train, validation)
    if validation.height < 20:
        return config["models"]["mixture_classes"][0], config["models"]["l2_candidates"][0]
    encoder = VoteEncoder().fit(inner_train)
    encoded_train = encoder.transform(inner_train, device)
    encoded_val = encoder.transform(validation, device)
    best = (float("inf"), 2, 1e-3)
    for classes in config["models"]["mixture_classes"]:
        for l2 in config["models"]["l2_candidates"]:
            set_deterministic(config["project"]["seed"] + classes)
            model = MixtureDavidsonModel.fit(
                encoded_train,
                classes=classes,
                l2=l2,
                dirichlet_alpha=config["models"]["dirichlet_alpha"],
                epochs=max(20, config["models"]["epochs"] // 2),
                learning_rate=config["models"]["learning_rate"],
                patience=config["models"]["patience"],
            )
            score = cross_entropy(model.predict_proba(encoded_val), _choice_array(encoded_val))
            if score < best[0]:
                best = score, classes, l2
    return best[1], best[2]


def run_dimension(
    config: dict[str, Any],
    dimension: str,
    *,
    detailed: bool = False,
    resume: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    completed_path = _artifacts(config, "metrics") / f"{dimension}_model_comparison.json"
    if resume and completed_path.exists():
        completed = json.loads(completed_path.read_text("utf-8"))
        if (
            completed.get("result_schema_version") == RESULT_SCHEMA_VERSION
            and completed.get("provenance", {}).get("config_hash")
            == config.get("_meta", {}).get("hash")
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
    fold_metrics = []
    selections = []
    all_scalar_scores, all_cont_scores, all_mix_scores, all_clusters = [], [], [], []
    for fold in range(config["splits"]["outer_folds"]):
        checkpoint = _fold_checkpoint_path(config, dimension, fold)
        if resume and checkpoint.exists():
            cached = _load_fold_checkpoint(checkpoint)
            fold_metrics.append(cached["fold_metric"])
            selections.append(cached["selection"])
            all_scalar_scores.append(cached["scalar_scores"])
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
        selected_rank, selected_l2 = _select_continuous(train, config, device)
        selected_classes, mixture_l2 = _select_mixture(train, config, device)
        selections.append(
            {
                "fold": fold,
                "continuous_rank": selected_rank,
                "continuous_l2": selected_l2,
                "mixture_classes": selected_classes,
                "mixture_l2": mixture_l2,
            }
        )
        encoder = VoteEncoder().fit(train)
        encoded_train = encoder.transform(train, device)
        encoded_test = encoder.transform(test, device)
        set_deterministic(config["project"]["seed"] + fold)
        scalar = DavidsonModel.fit(
            encoded_train,
            l2=selected_l2,
            epochs=config["models"]["epochs"],
            learning_rate=config["models"]["learning_rate"],
            patience=config["models"]["patience"],
        )
        continuous = ContinuousPreferenceModel.fit(
            encoded_train,
            rank=selected_rank,
            l2=selected_l2,
            epochs=config["models"]["epochs"],
            learning_rate=config["models"]["learning_rate"],
            patience=config["models"]["patience"],
        )
        mixture = MixtureDavidsonModel.fit(
            encoded_train,
            classes=selected_classes,
            l2=mixture_l2,
            dirichlet_alpha=config["models"]["dirichlet_alpha"],
            epochs=config["models"]["epochs"],
            learning_rate=config["models"]["learning_rate"],
            patience=config["models"]["patience"],
        )
        choices = _choice_array(encoded_test)
        scalar_p = scalar.predict_proba(encoded_test)
        continuous_p = continuous.predict_proba(encoded_test)
        mixture_p = mixture.predict_proba(encoded_test)
        empirical = empirical_probabilities(_choice_array(encoded_train))
        fold_metric = {
            "fold": fold,
            "train_votes": train.height,
            "test_votes": test.height,
            "test_coverage": test.height / max(test_all.height, 1),
            "selected_rank": selected_rank,
            "selected_classes": selected_classes,
            "continuous_l2": selected_l2,
            "mixture_l2": mixture_l2,
            "m0_cross_entropy": cross_entropy(
                np.repeat(empirical[None, :], len(choices), axis=0), choices
            ),
            "scalar_cross_entropy": cross_entropy(scalar_p, choices),
            "continuous_cross_entropy": cross_entropy(continuous_p, choices),
            "mixture_cross_entropy": cross_entropy(mixture_p, choices),
        }
        scalar_scores = log_score(scalar_p, choices)
        continuous_scores = log_score(continuous_p, choices)
        mixture_scores = log_score(mixture_p, choices)
        clusters = _cluster_array(test)
        fold_metrics.append(fold_metric)
        all_scalar_scores.append(scalar_scores)
        all_cont_scores.append(continuous_scores)
        all_mix_scores.append(mixture_scores)
        all_clusters.append(clusters)
        _save_fold_checkpoint(
            checkpoint,
            fold_metric=fold_metric,
            selection=selections[-1],
            scalar_scores=scalar_scores,
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
    scalar_scores = np.concatenate(all_scalar_scores)
    continuous_scores = np.concatenate(all_cont_scores)
    mixture_scores = np.concatenate(all_mix_scores)
    clusters = np.concatenate(all_clusters)
    scalar_ce = float(-scalar_scores.mean())
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
    # Refit the selected specification on all observations only after every
    # outer-fold score has been frozen. These models produce descriptive tables,
    # never outer-fold predictive metrics.
    final_encoder = VoteEncoder().fit(frame)
    final_encoded = final_encoder.transform(frame, device)
    set_deterministic(config["project"]["seed"])
    final_scalar = DavidsonModel.fit(
        final_encoded,
        l2=selected_l2,
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    final_continuous = ContinuousPreferenceModel.fit(
        final_encoded,
        rank=selected_rank,
        l2=selected_l2,
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    mixture = MixtureDavidsonModel.fit(
        final_encoded,
        classes=selected_classes,
        l2=mixture_l2,
        dirichlet_alpha=config["models"]["dirichlet_alpha"],
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    final_models = {
        "encoder": final_encoder,
        "scalar": final_scalar,
        "continuous": final_continuous,
        "mixture": mixture,
        "train": frame,
    }
    reversal = ranking_reversal_fraction(mixture.utilities())
    stability = _mixture_stability(frame, selected_classes, mixture_l2, config, device)
    auxiliary = _evaluate_auxiliary_holdouts(
        frame,
        config,
        device,
        rank=selected_rank,
        continuous_l2=selected_l2,
        classes=selected_classes,
        mixture_l2=mixture_l2,
    )
    verdict, gates = heterogeneity_verdict(
        config,
        scalar_ce,
        continuous_ce,
        mixture_ce,
        continuous_ci,
        mixture_ci,
        mixture.class_weights().tolist(),
        reversal,
        stability,
        auxiliary["continuous_auxiliary_gate"],
        auxiliary["mixture_auxiliary_gate"],
    )
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "dimension": dimension,
        "verdict": verdict,
        "selected_rank": selected_rank,
        "selected_classes": selected_classes,
        "outer_fold_selections": selections,
        "folds": fold_metrics,
        "scalar_cross_entropy": scalar_ce,
        "continuous_cross_entropy": continuous_ce,
        "mixture_cross_entropy": mixture_ce,
        "continuous_elpd_ci": continuous_ci,
        "mixture_elpd_ci": mixture_ci,
        "class_weights": mixture.class_weights().tolist(),
        "ranking_reversal_fraction": reversal,
        "stability_ari": stability,
        "auxiliary_holdouts": auxiliary,
        "gates": gates,
        "provenance": metadata(config),
    }
    write_json(completed_path, result)
    if detailed:
        _save_model_tables(config, dimension, final_models)
    return result, final_models


def _mixture_stability(
    train: pl.DataFrame,
    classes: int,
    l2: float,
    config: dict[str, Any],
    device: torch.device,
) -> float:
    encoder = VoteEncoder().fit(train)
    encoded = encoder.transform(train, device)
    labels = []
    refits = max(1, config["gates"]["stability_refits"])
    for index in range(refits):
        set_deterministic(config["project"]["seeds"][index % len(config["project"]["seeds"])])
        model = MixtureDavidsonModel.fit(
            encoded,
            classes=classes,
            l2=l2,
            dirichlet_alpha=config["models"]["dirichlet_alpha"],
            epochs=max(20, config["models"]["epochs"] // 2),
            learning_rate=config["models"]["learning_rate"],
            patience=config["models"]["patience"],
        )
        labels.append(model.posterior.argmax(1))
    if len(labels) < 2:
        return 1.0
    values = [
        adjusted_rand_score(labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    return float(np.median(values))


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


def _full_mixture(
    frame: pl.DataFrame,
    classes: int,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[VoteEncoder, MixtureDavidsonModel]:
    encoder = VoteEncoder().fit(frame)
    encoded = encoder.transform(frame, device)
    set_deterministic(config["project"]["seed"])
    model = MixtureDavidsonModel.fit(
        encoded,
        classes=classes,
        l2=config["models"]["l2_candidates"][0],
        dirichlet_alpha=config["models"]["dirichlet_alpha"],
        epochs=config["models"]["epochs"],
        learning_rate=config["models"]["learning_rate"],
        patience=config["models"]["patience"],
    )
    return encoder, model


def run_cusp_stage(
    config: dict[str, Any],
    primary_result: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    if primary_result["verdict"] != "SCALAR_REJECTED_MIXTURE":
        return primary_result["verdict"], {"status": "skipped", "reason": "mixture_gate_failed"}
    votes = pl.read_parquet(Path(config["data"]["processed_dir"]) / "votes.parquet")
    device = select_device(config["project"]["device"])
    classes = primary_result["selected_classes"]
    fitted = {}
    for dimension in DIMENSIONS:
        frame = votes.filter(pl.col("dimension") == dimension)
        fitted[dimension] = _full_mixture(frame, classes, config, device)
    common_images = set(fitted["safety"][0].image_ids)
    for encoder, _ in fitted.values():
        common_images &= set(encoder.image_ids)
    common = sorted(common_images)
    if len(common) < config["bimodality"]["min_neighbors"]:
        return "SCALAR_REJECTED_MIXTURE", {
            "status": "skipped",
            "reason": "insufficient_cross_dimension_image_overlap",
            "images": len(common),
        }
    utilities = {}
    for dimension, (encoder, model) in fitted.items():
        lookup = {image: i for i, image in enumerate(encoder.image_ids)}
        utilities[dimension] = model.utilities()[:, [lookup[image] for image in common]]
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
    simulation = validate_density_recovery(config)
    if simulation["status"] != "ok":
        write_verdict(
            config, "SCALAR_NOT_REJECTED", reasons=["simulation_calibration_failed"], metrics=simulation
        )
        build_report(config)
        return {"verdict": "SCALAR_NOT_REJECTED", "simulation": simulation}
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
