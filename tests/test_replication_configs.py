from __future__ import annotations

from placepulse_cusp.config import load_config


def test_replication_dimension_configs_are_frozen_and_isolated():
    dimensions = ["lively", "beautiful", "wealthy", "boring", "depressing"]
    artifact_roots = []

    for dimension in dimensions:
        config = load_config(f"configs/replication_dimensions/{dimension}.yaml")
        assert config["data"]["primary_dimension"] == dimension
        assert config["project"]["seed"] == 1103
        assert config["project"]["device"] == "cuda"
        assert config["simulation"]["calibration_policy"] == "provisional_effective"
        assert config["reporting"]["preflight_only"] is False
        assert config["models"]["epochs"] == 300
        assert config["splits"]["outer_folds"] == 5
        assert config["models"]["random_starts"] == 5
        artifact_roots.append(config["reporting"]["artifacts_dir"])

    assert len(set(artifact_roots)) == len(dimensions)


def test_wealthy_provisional_multiseed_configs_are_frozen_and_isolated():
    seeds = [1103, 2207, 3319, 4421, 5527]
    artifact_roots = []

    for seed in seeds:
        config = load_config(f"configs/wealthy_provisional_seeds/seed_{seed}.yaml")
        assert config["data"]["primary_dimension"] == "wealthy"
        assert config["project"]["seed"] == seed
        assert config["project"]["device"] == "cuda"
        assert config["simulation"]["calibration_policy"] == "provisional_effective"
        assert config["reporting"]["preflight_only"] is False
        assert config["models"]["epochs"] == 300
        assert config["splits"]["outer_folds"] == 5
        assert config["models"]["random_starts"] == 5
        artifact_roots.append(config["reporting"]["artifacts_dir"])

    assert len(set(artifact_roots)) == len(seeds)
