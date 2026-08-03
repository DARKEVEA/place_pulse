from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from placepulse_cusp.provenance import sha256_file

CALIBRATION_MANIFEST_SCHEMA_VERSION = 1
CALIBRATION_POLICIES = {"strict", "provisional_effective"}


def assess_calibration(
    model_recovery: dict[str, Any],
    density_recovery: dict[str, Any],
    *,
    policy: str = "strict",
) -> dict[str, Any]:
    if policy not in CALIBRATION_POLICIES:
        choices = ", ".join(sorted(CALIBRATION_POLICIES))
        raise ValueError(f"Unknown calibration policy {policy!r}; expected one of: {choices}")

    model_field = "status" if policy == "strict" else "effective_status"
    model_status = model_recovery.get(model_field)
    density_status = density_recovery.get("status")
    reasons = []
    if model_status != "ok":
        reasons.append(f"model_recovery.{model_field}={model_status!r}")
    if density_status != "ok":
        reasons.append(f"density_recovery.status={density_status!r}")

    return {
        "status": "ok" if not reasons else "failed",
        "policy": policy,
        "confirmatory": policy == "strict",
        "model_acceptance_field": model_field,
        "model_status": model_recovery.get("status"),
        "model_effective_status": model_recovery.get("effective_status"),
        "density_status": density_status,
        "reasons": reasons,
    }


def _manifest_artifact(
    manifest_path: Path,
    manifest: dict[str, Any],
    name: str,
) -> tuple[Path, dict[str, Any]]:
    try:
        record = manifest["artifacts"][name]
        relative_path = record["path"]
        expected_hash = record["sha256"].lower()
    except (KeyError, TypeError) as error:
        raise ValueError(f"Calibration manifest has no valid {name!r} artifact") from error

    path = (manifest_path.parent / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen calibration artifact is missing: {path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Frozen calibration artifact hash mismatch for {name}: "
            f"expected {expected_hash}, found {actual_hash}"
        )
    try:
        payload = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Frozen calibration artifact is not valid JSON: {path}") from error
    return path, payload


def load_calibration_manifest(
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    simulation = config.get("simulation", {})
    configured_path = simulation.get("calibration_manifest")
    if not configured_path:
        raise ValueError("simulation.calibration_manifest is required")
    manifest_path = Path(configured_path).resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Calibration manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text("utf-8"))
    if manifest.get("schema_version") != CALIBRATION_MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported calibration manifest schema_version: "
            f"{manifest.get('schema_version')!r}"
        )

    manifest_policy = manifest.get("policy", "strict")
    configured_policy = simulation.get("calibration_policy", "strict")
    if manifest_policy != configured_policy:
        raise ValueError(
            "Calibration policy mismatch: "
            f"config={configured_policy!r}, manifest={manifest_policy!r}"
        )

    model_path, model_recovery = _manifest_artifact(
        manifest_path, manifest, "model_recovery"
    )
    density_path, density_recovery = _manifest_artifact(
        manifest_path, manifest, "density_recovery"
    )
    assessment = assess_calibration(
        model_recovery, density_recovery, policy=configured_policy
    )
    assessment.update(
        {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "label": manifest.get("label"),
            "artifacts": {
                "model_recovery": {
                    "path": str(model_path),
                    "sha256": sha256_file(model_path),
                },
                "density_recovery": {
                    "path": str(density_path),
                    "sha256": sha256_file(density_path),
                },
            },
        }
    )
    return model_recovery, density_recovery, assessment
