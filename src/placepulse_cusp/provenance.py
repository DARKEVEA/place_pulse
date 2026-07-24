from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config_hash": config.get("_meta", {}).get("hash"),
        "git_commit": git_commit(),
        "seed": config["project"]["seed"],
        "inputs": files,
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), "utf-8")

