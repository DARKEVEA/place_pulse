from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from placepulse_cusp.data import fetch


def test_kaggle_fetch_downloads_only_configured_files(tmp_path: Path, monkeypatch):
    source = tmp_path / "remote" / "votes_clean.csv"
    source.parent.mkdir()
    source.write_text("choice,left,right\nleft,a,b\n", "utf-8")
    calls = []

    def fake_download(handle, path=None, output_dir=None):
        calls.append((handle, path, output_dir))
        target = Path(output_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())
        return str(target)

    monkeypatch.setattr(fetch.kagglehub, "dataset_download", fake_download)
    config = {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "source_url": "https://example.test",
            "local_source": None,
            "kaggle": {
                "handle": "owner/dataset",
                "version": 2,
                "files": ["votes_clean.csv"],
                "output_dir": str(tmp_path / "raw" / "kaggle"),
            },
        }
    }
    result = fetch.fetch_data(config)
    assert result["status"] == "ok"
    assert result["mode"] == "kaggle"
    assert calls == [
        ("owner/dataset", "votes_clean.csv", str(tmp_path / "raw" / "kaggle"))
    ]
    assert Path(result["files"][0]["path"]).name == "votes_clean.csv"


def test_kaggle_fetch_extracts_zip_returned_with_csv_name(tmp_path: Path, monkeypatch):
    def fake_download(handle, path=None, output_dir=None):
        target = Path(output_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(target, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr(path, "choice,left,right\nleft,a,b\n")
        return str(target)

    monkeypatch.setattr(fetch.kagglehub, "dataset_download", fake_download)
    output_dir = tmp_path / "raw" / "kaggle"
    config = {
        "data": {
            "raw_dir": str(tmp_path / "raw"),
            "source_url": "https://example.test",
            "local_source": None,
            "kaggle": {
                "handle": "owner/dataset",
                "version": 2,
                "files": ["votes_clean.csv"],
                "output_dir": str(output_dir),
            },
        }
    }

    result = fetch.fetch_data(config)

    target = Path(result["files"][0]["path"])
    assert target.read_text("utf-8") == "choice,left,right\nleft,a,b\n"
    assert (output_dir / "votes_clean.csv.zip").is_file()
