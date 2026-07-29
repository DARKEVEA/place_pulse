from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from placepulse_cusp.provenance import write_json


def _balanced_folds(values: np.ndarray, folds: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    unique = np.unique(values)
    rng.shuffle(unique)
    lookup = {value: index % folds for index, value in enumerate(unique)}
    return np.asarray([lookup[value] for value in values], dtype=np.int16)


def edge_keys(frame: pl.DataFrame) -> np.ndarray:
    """Return an order-invariant comparison-edge key for every vote."""
    dimensions = frame["dimension"].to_list()
    left = frame["left_image_id"].to_list()
    right = frame["right_image_id"].to_list()
    return np.asarray(
        [
            f"{dimension}\x1f{min(a, b)}\x1f{max(a, b)}"
            for dimension, a, b in zip(dimensions, left, right)
        ],
        dtype=object,
    )


def grouped_edge_folds(frame: pl.DataFrame, folds: int, seed: int) -> np.ndarray:
    """Assign all observations of the same unordered image pair to one fold."""
    return _balanced_folds(edge_keys(frame), folds, seed)


def create_splits(config: dict[str, Any], votes_path: str | Path | None = None) -> Path:
    path = Path(votes_path or Path(config["data"]["interim_dir"]) / "votes.parquet")
    votes = pl.read_parquet(path)
    folds = int(config["splits"]["outer_folds"])
    seed = int(config["project"]["seed"])
    edge_fold = grouped_edge_folds(votes, folds, seed)
    dimensions = votes["dimension"].to_numpy()
    left_images = votes["left_image_id"].to_numpy()
    right_images = votes["right_image_id"].to_numpy()
    # Reserve any edge whose assigned test fold would contain an image absent
    # from that fold's training graph. Because assignment is grouped by edge,
    # repeated votes for an image pair can never leak across train and test.
    for dimension in votes["dimension"].unique().to_list():
        for fold in range(folds):
            dimension_indices = np.where(dimensions == dimension)[0]
            train_indices = dimension_indices[edge_fold[dimension_indices] != fold]
            train_images = set(left_images[train_indices]) | set(right_images[train_indices])
            test_indices = dimension_indices[edge_fold[dimension_indices] == fold]
            eligible = np.asarray(
                [
                    left_images[index] in train_images and right_images[index] in train_images
                    for index in test_indices
                ],
                dtype=bool,
            )
            edge_fold[test_indices[~eligible]] = -1
    voter_values = votes["voter_id"].fill_null("__anonymous__").to_numpy()
    voter_fold = _balanced_folds(voter_values, folds, seed + 1)

    timestamps = votes["timestamp"].to_numpy()
    time_test = np.zeros(votes.height, dtype=bool)
    for dimension in votes["dimension"].unique().to_list():
        indices = np.where(votes["dimension"].to_numpy() == dimension)[0]
        valid = indices[np.asarray([timestamps[i] is not None for i in indices])]
        if len(valid):
            ordered = valid[np.argsort(timestamps[valid])]
            start = int(np.floor(len(ordered) * (1 - config["splits"]["time_test_fraction"])))
            time_test[ordered[start:]] = True
    split_frame = pl.DataFrame(
        {
            "vote_id": votes["vote_id"],
            "edge_fold": edge_fold,
            "voter_fold": voter_fold,
            "time_test": time_test,
        }
    )
    output = Path(config["data"]["processed_dir"]) / "splits.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    split_frame.write_parquet(output)
    write_json(
        output.with_suffix(".json"),
        {
            "split_schema_version": 2,
            "edge_grouped": True,
            "outer_folds": folds,
            "seed": seed,
            "rows": votes.height,
            "train_only_edges": int((edge_fold < 0).sum()),
            "edge_fold_counts": {
                str(fold): int((edge_fold == fold).sum()) for fold in range(-1, folds)
            },
        },
    )
    return output


def prepare_data(config: dict[str, Any]) -> tuple[Path, Path]:
    source = Path(config["data"]["interim_dir"]) / "votes.parquet"
    votes = pl.read_parquet(source)
    output = Path(config["data"]["processed_dir"]) / "votes.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    votes.write_parquet(output)
    return output, create_splits(config, source)
