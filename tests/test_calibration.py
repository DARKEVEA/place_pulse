from __future__ import annotations

import json
from pathlib import Path

import pytest

from placepulse_cusp.calibration import assess_calibration, load_calibration_manifest
from placepulse_cusp.provenance import sha256_file


def test_strict_policy_rejects_effective_only_model_recovery():
    result = assess_calibration(
        {"status": "failed", "effective_status": "ok"},
        {"status": "ok"},
    )

    assert result["status"] == "failed"
    assert result["confirmatory"]
    assert result["model_acceptance_field"] == "status"


def test_provisional_policy_accepts_effective_model_and_strict_density():
    result = assess_calibration(
        {"status": "failed", "effective_status": "ok"},
        {"status": "ok"},
        policy="provisional_effective",
    )

    assert result["status"] == "ok"
    assert not result["confirmatory"]
    assert result["model_acceptance_field"] == "effective_status"


def _write_manifest(tmp_path: Path, *, artifact_hash: str) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "policy": "provisional_effective",
                "artifacts": {
                    "model_recovery": {
                        "path": "model.json",
                        "sha256": artifact_hash,
                    },
                    "density_recovery": {
                        "path": "density.json",
                        "sha256": sha256_file(tmp_path / "density.json"),
                    },
                },
            }
        ),
        "utf-8",
    )
    return manifest


def test_manifest_loads_separate_hashed_artifacts(tmp_path):
    model = tmp_path / "model.json"
    density = tmp_path / "density.json"
    model.write_text('{"status":"failed","effective_status":"ok"}', "utf-8")
    density.write_text('{"status":"ok"}', "utf-8")
    manifest = _write_manifest(tmp_path, artifact_hash=sha256_file(model))
    config = {
        "simulation": {
            "calibration_manifest": str(manifest),
            "calibration_policy": "provisional_effective",
        }
    }

    _, _, assessment = load_calibration_manifest(config)

    assert assessment["status"] == "ok"
    assert assessment["artifacts"]["model_recovery"]["path"] == str(model.resolve())


def test_manifest_rejects_changed_artifact(tmp_path):
    model = tmp_path / "model.json"
    density = tmp_path / "density.json"
    model.write_text('{"status":"failed","effective_status":"ok"}', "utf-8")
    density.write_text('{"status":"ok"}', "utf-8")
    manifest = _write_manifest(tmp_path, artifact_hash="0" * 64)
    config = {
        "simulation": {
            "calibration_manifest": str(manifest),
            "calibration_policy": "provisional_effective",
        }
    }

    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_calibration_manifest(config)


def test_formal_provisional_safety_seed_configs_are_isolated():
    from placepulse_cusp.config import load_config

    seeds = [1103, 2207, 3319, 4421, 5527]
    artifact_roots = []
    for seed in seeds:
        config = load_config(f"configs/safety_provisional_seeds/seed_{seed}.yaml")
        assert config["project"]["seed"] == seed
        assert config["project"]["device"] == "cuda"
        assert config["simulation"]["calibration_policy"] == "provisional_effective"
        assert config["reporting"]["preflight_only"] is False
        assert config["models"]["epochs"] == 300
        assert config["splits"]["outer_folds"] == 5
        artifact_roots.append(config["reporting"]["artifacts_dir"])

    assert len(set(artifact_roots)) == len(seeds)
