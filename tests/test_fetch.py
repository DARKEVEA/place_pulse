from pathlib import Path

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
