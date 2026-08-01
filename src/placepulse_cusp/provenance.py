from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    repository = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={repository.as_posix()}",
                "-C",
                str(repository),
                "rev-parse",
                "HEAD",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def metadata(config: dict[str, Any], inputs: list[str | Path] | None = None) -> dict[str, Any]:
    files = {}
    for item in inputs or []:
        path = Path(item)
        if path.exists() and path.is_file():
            files[str(path)] = sha256_file(path)
    return {
        "created_at": datetime.now(UTC).isoformat(),
        "config_hash": config.get("_meta", {}).get("hash"),
        "git_commit": git_commit(),
        "seed": config["project"]["seed"],
        "inputs": files,
        "run_label": config.get("reporting", {}).get("run_label"),
    }


def _command_output(command: list[str]) -> list[str]:
    try:
        output = subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL, timeout=60
        )
        return [line for line in output.splitlines() if line.strip()]
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []


def write_run_manifest(config: dict[str, Any], inputs: list[str | Path]) -> Path:
    import torch

    target = Path(config["reporting"]["artifacts_dir"]) / "run_manifest.json"
    payload = {
        **metadata(config, inputs),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "pip_freeze": _command_output([sys.executable, "-m", "pip", "freeze"]),
        "conda_list": _command_output(["conda", "list", "--json"]),
    }
    write_json(target, payload)
    return target


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), "utf-8")

