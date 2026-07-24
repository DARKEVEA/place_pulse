from pathlib import Path

import polars as pl

from placepulse_cusp.config import load_config
from placepulse_cusp.data.schema import standardise_votes


def test_standardisation_preserves_ties_and_real_repeats(tmp_path: Path):
    raw = tmp_path / "votes.csv"
    pl.DataFrame(
        {
            "_id": ["1", "2", "3"],
            "voter_uniqueid": ["a", "a", "a"],
            "study_id": ["safety"] * 3,
            "left": ["x", "x", "x"],
            "right": ["y", "y", "y"],
            "choice": ["equal", "left", "left"],
        }
    ).write_csv(raw)
    config = load_config("configs/smoke.yaml")
    config["data"]["votes_file"] = str(raw)
    config["data"]["interim_dir"] = str(tmp_path / "interim")
    path, audit = standardise_votes(config)
    votes = pl.read_parquet(path)
    assert votes.height == 3
    assert votes.filter(pl.col("choice") == "equal").height == 1
    assert pl.read_parquet(audit).height == 0


def test_standardisation_audits_unknown_question_and_choice(tmp_path: Path):
    raw = tmp_path / "votes.csv"
    pl.DataFrame(
        {
            "_id": ["1", "2", "3"],
            "voter_uniqueid": ["a", "b", "c"],
            "study_id": ["safety", "injected", "safety"],
            "study_question": ["safer", "NA", "safer"],
            "left": ["x", "x", "x"],
            "right": ["y", "y", "y"],
            "choice": ["left", "left", "unknown"],
        }
    ).write_csv(raw)
    config = load_config("configs/smoke.yaml")
    config["data"]["votes_file"] = str(raw)
    config["data"]["interim_dir"] = str(tmp_path / "interim")
    path, audit = standardise_votes(config)
    assert pl.read_parquet(path).height == 1
    reasons = pl.read_parquet(audit)
    assert reasons.height == 2
    assert reasons["reason"].unique().to_list() == ["invalid_required_field"]


def test_kaggle_columns_map_question_timestamp_and_city(tmp_path: Path):
    raw = tmp_path / "votes_clean.csv"
    pl.DataFrame(
        {
            "voter_uniqueid": ["person"],
            "study_id": ["50a68a51fdc9f05596000002"],
            "study_question": ["safer"],
            "left": ["image-left"],
            "right": ["image-right"],
            "choice": ["equal"],
            "place_name_left": ["Amsterdam"],
            "place_name_right": ["Boston"],
            "day": ["2014-04-20"],
            "time": ["21:17:31"],
            "long_left": [4.90],
            "lat_left": [52.37],
            "long_right": [-71.06],
            "lat_right": [42.36],
        }
    ).write_csv(raw)
    config = load_config("configs/confirmatory.yaml")
    config["data"]["votes_file"] = str(raw)
    config["data"]["interim_dir"] = str(tmp_path / "interim")
    path, _ = standardise_votes(config)
    row = pl.read_parquet(path).row(0, named=True)
    assert row["dimension"] == "safety"
    assert row["choice"] == "equal"
    assert row["city_left"] == "Amsterdam"
    assert row["city_right"] == "Boston"
    assert row["timestamp"].isoformat().startswith("2014-04-20T21:17:31")
