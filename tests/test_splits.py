import json
from pathlib import Path

import polars as pl

from placepulse_cusp.config import load_config
from placepulse_cusp.data.splits import create_splits, grouped_edge_folds


def test_repeated_unordered_edges_are_never_split():
    frame = pl.DataFrame(
        {
            "dimension": ["safety"] * 6,
            "left_image_id": ["a", "b", "a", "c", "d", "d"],
            "right_image_id": ["b", "a", "b", "d", "c", "e"],
        }
    )
    folds = grouped_edge_folds(frame, 3, 1103)
    assert len(set(folds[:3])) == 1
    assert folds[3] == folds[4]


def test_edge_folds_never_hold_out_unseen_images(tmp_path: Path):
    votes = pl.DataFrame(
        {
            "vote_id": [str(index) for index in range(8)],
            "voter_id": [f"v{index % 3}" for index in range(8)],
            "dimension": ["safety"] * 8,
            "left_image_id": ["rare", "a", "a", "b", "b", "c", "c", "a"],
            "right_image_id": ["a", "b", "c", "c", "a", "a", "b", "c"],
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 8),
                interval="1d",
                eager=True,
            ),
        }
    )
    source = tmp_path / "votes.parquet"
    votes.write_parquet(source)
    config = load_config("configs/smoke.yaml")
    config["data"]["processed_dir"] = str(tmp_path / "processed")
    config["splits"]["outer_folds"] = 3
    split_path = create_splits(config, source)
    manifest = json.loads(split_path.with_suffix(".json").read_text("utf-8"))
    assert manifest["split_schema_version"] == 2
    joined = votes.join(pl.read_parquet(split_path), on="vote_id")
    assert joined.filter(pl.col("vote_id") == "0")["edge_fold"].item() == -1
    for fold in range(3):
        train = joined.filter(pl.col("edge_fold") != fold)
        test = joined.filter(pl.col("edge_fold") == fold)
        images = set(train["left_image_id"]) | set(train["right_image_id"])
        assert all(value in images for value in test["left_image_id"])
        assert all(value in images for value in test["right_image_id"])


def test_voter_fold_has_no_voter_leakage(tmp_path: Path):
    votes = pl.DataFrame(
        {
            "vote_id": [str(index) for index in range(12)],
            "voter_id": [f"v{index % 4}" for index in range(12)],
            "dimension": ["safety"] * 12,
            "left_image_id": [f"i{index % 3}" for index in range(12)],
            "right_image_id": [f"i{(index + 1) % 3}" for index in range(12)],
            "timestamp": pl.datetime_range(
                pl.datetime(2020, 1, 1),
                pl.datetime(2020, 1, 12),
                interval="1d",
                eager=True,
            ),
        }
    )
    source = tmp_path / "votes.parquet"
    votes.write_parquet(source)
    config = load_config("configs/smoke.yaml")
    config["data"]["processed_dir"] = str(tmp_path / "processed")
    split_path = create_splits(config, source)
    joined = votes.join(pl.read_parquet(split_path), on="vote_id")
    maximum = (
        joined.group_by("voter_id")
        .agg(pl.col("voter_fold").n_unique().alias("n"))
        .select(pl.col("n").max())
        .item()
    )
    assert maximum == 1
