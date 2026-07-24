from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    extends = config.pop("extends", None)
    if extends:
        parent = path.parent / extends
        if not parent.suffix:
            parent = parent.with_suffix(".yaml")
        config = _deep_merge(load_config(parent), config)
    config["_meta"] = {
        "path": str(path.resolve()),
        "hash": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest(),
    }
    return config


def artifact_dir(config: dict[str, Any], *parts: str) -> Path:
    root = Path(config["reporting"]["artifacts_dir"])
    path = root.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

