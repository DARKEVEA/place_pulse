from pathlib import Path

import polars as pl

from placepulse_cusp.config import load_config
from placepulse_cusp.simulation.generate import generate_vote_table


def test_generator_emits_all_dimensions_and_ties(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config["simulation"].update({"voters": 20, "images": 15, "votes": 600})
    target = generate_vote_table(config, output=tmp_path / "sim.csv")
    frame = pl.read_csv(target)
    assert frame["study_id"].n_unique() == 6
    assert "equal" in frame["choice"].unique().to_list()


def test_null_generator_is_supported(tmp_path: Path):
    config = load_config("configs/smoke.yaml")
    config["simulation"].update({"voters": 10, "images": 8, "votes": 120})
    target = generate_vote_table(
        config, mechanism="null", output=tmp_path / "null.csv"
    )
    assert pl.read_csv(target).height == 120

