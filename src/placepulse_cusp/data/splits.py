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


def create_splits(config: dict[str, Any], votes_path: str | Path | None = None) -> Path:
    path = Path(votes_path or Path(config["data"]["interim_dir"]) / "votes.parquet")
    votes = pl.read_parquet(path)
    folds = int(config["splits"]["outer_folds"])
    seed = int(config["project"]["seed"])
    rng = np.random.default_rng(seed)
    edge_fold = np.arange(votes.height, dtype=np.int64) % folds
    rng.shuffle(edge_fold)
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
        {"outer_folds": folds, "seed": seed, "rows": votes.height},
    )
    return output


def prepare_data(config: dict[str, Any]) -> tuple[Path, Path]:
    source = Path(config["data"]["interim_dir"]) / "votes.parquet"
    votes = pl.read_parquet(source)
    output = Path(config["data"]["processed_dir"]) / "votes.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    votes.write_parquet(output)
    return output, create_splits(config, source)

