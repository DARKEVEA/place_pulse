from __future__ import annotations

import json

from placepulse_cusp import cli
from placepulse_cusp.config import load_config


def test_ensure_prepared_regenerates_splits_when_seed_or_fold_count_changes(
    tmp_path, monkeypatch
):
    config = load_config("configs/safety_provisional_seeds/seed_2207.yaml")
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir()
    processed.mkdir()
    (interim / "votes.parquet").touch()
    (processed / "votes.parquet").touch()
    (processed / "splits.json").write_text(
        json.dumps(
            {
                "split_schema_version": 2,
                "outer_folds": 2,
                "seed": 1103,
            }
        ),
        "utf-8",
    )
    config["data"]["interim_dir"] = str(interim)
    config["data"]["processed_dir"] = str(processed)
    monkeypatch.setattr(cli, "validate_votes", lambda *args: {"status": "ok"})
    calls = []
    monkeypatch.setattr(cli, "prepare_data", lambda value: calls.append(value))

    cli._ensure_prepared(config)

    assert calls == [config]


def test_ensure_prepared_reuses_matching_seed_and_fold_manifest(tmp_path, monkeypatch):
    config = load_config("configs/safety_provisional_seeds/seed_3319.yaml")
    interim = tmp_path / "interim"
    processed = tmp_path / "processed"
    interim.mkdir()
    processed.mkdir()
    (interim / "votes.parquet").touch()
    (processed / "votes.parquet").touch()
    (processed / "splits.json").write_text(
        json.dumps(
            {
                "split_schema_version": 2,
                "outer_folds": 5,
                "seed": 3319,
            }
        ),
        "utf-8",
    )
    config["data"]["interim_dir"] = str(interim)
    config["data"]["processed_dir"] = str(processed)
    monkeypatch.setattr(cli, "validate_votes", lambda *args: {"status": "ok"})
    calls = []
    monkeypatch.setattr(cli, "prepare_data", lambda value: calls.append(value))

    cli._ensure_prepared(config)

    assert calls == []
