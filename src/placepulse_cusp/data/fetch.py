from __future__ import annotations

import shutil
import urllib.request
from urllib.parse import urlparse
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import kagglehub

from placepulse_cusp.provenance import sha256_file, write_json


def _materialize_downloaded_file(
    downloaded: Path, output_dir: Path, relative_path: str
) -> Path:
    """Extract a requested file when KaggleHub returns a ZIP with its filename."""
    if not zipfile.is_zipfile(downloaded):
        return downloaded

    expected_name = Path(relative_path).name
    target = output_dir / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.extracting")

    with zipfile.ZipFile(downloaded) as archive:
        matches = [
            member
            for member in archive.infolist()
            if not member.is_dir() and Path(member.filename).name == expected_name
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one {expected_name!r} in {downloaded}, found {len(matches)}."
            )
        with archive.open(matches[0]) as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination)

    if downloaded.resolve() == target.resolve():
        backup = downloaded.with_name(f"{downloaded.name}.zip")
        index = 1
        while backup.exists():
            backup = downloaded.with_name(f"{downloaded.name}.zip.{index}")
            index += 1
        downloaded.replace(backup)
    temporary.replace(target)
    return target


def _copy_or_extract(source: Path, raw_dir: Path) -> list[Path]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        copied = []
        for item in source.iterdir():
            if item.is_file():
                target = raw_dir / item.name
                if item.resolve() != target.resolve():
                    shutil.copy2(item, target)
                copied.append(target)
        return copied
    target = raw_dir / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    if zipfile.is_zipfile(target):
        extract_dir = raw_dir / target.stem
        extract_dir.mkdir(exist_ok=True)
        with zipfile.ZipFile(target) as archive:
            archive.extractall(extract_dir)
        return [p for p in extract_dir.rglob("*") if p.is_file()]
    return [target]


def fetch_data(config: dict[str, Any], source: str | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    selected = source or data_cfg.get("local_source")
    if selected:
        files = _copy_or_extract(Path(selected), raw_dir)
        mode = "local"
    elif data_cfg.get("kaggle"):
        kaggle_cfg = data_cfg["kaggle"]
        output_dir = Path(kaggle_cfg.get("output_dir", raw_dir / "kaggle"))
        output_dir.mkdir(parents=True, exist_ok=True)
        files = []
        for relative_path in kaggle_cfg["files"]:
            downloaded = Path(
                kagglehub.dataset_download(
                    kaggle_cfg["handle"],
                    path=relative_path,
                    output_dir=str(output_dir),
                )
            )
            if downloaded.is_dir():
                candidate = downloaded / relative_path
                if not candidate.exists():
                    matches = list(downloaded.rglob(Path(relative_path).name))
                    if not matches:
                        raise FileNotFoundError(
                            f"KaggleHub reported {downloaded}, but {relative_path} was not found."
                        )
                    candidate = matches[0]
                downloaded = candidate
            files.append(_materialize_downloaded_file(downloaded, output_dir, relative_path))
        mode = "kaggle"
    else:
        url = data_cfg.get("download_url")
        if not url:
            result = {
                "status": "manual_source_required",
                "message": (
                    "The Figshare landing page is recorded for provenance, but no stable raw-file "
                    "URL is configured. Set data.local_source or data.download_url."
                ),
                "source_page": data_cfg.get("source_url"),
                "files": [],
            }
            write_json(raw_dir / "fetch_manifest.json", result)
            return result
        raw_dir.mkdir(parents=True, exist_ok=True)
        target = raw_dir / Path(urlparse(url).path).name
        urllib.request.urlretrieve(url, target)
        files = _copy_or_extract(target, raw_dir)
        mode = "download"
    result = {
        "status": "ok",
        "mode": mode,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": selected
        or (data_cfg.get("kaggle") or {}).get("handle")
        or data_cfg.get("download_url"),
        "source_page": data_cfg.get("source_url"),
        "upstream_source_page": data_cfg.get("upstream_source_url"),
        "dataset_version": (data_cfg.get("kaggle") or {}).get("version"),
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(files)
        ],
    }
    write_json(raw_dir / "fetch_manifest.json", result)
    return result
