from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import polars as pl

from placepulse_cusp.constants import CHOICES, DIMENSIONS
from placepulse_cusp.provenance import write_json

ALIASES = {
    "vote_id": ("vote_id", "_id", "id", "response_id"),
    "voter_id": ("voter_id", "voter_uniqueid", "user_id", "participant_id"),
    "study_id": ("study_id", "study", "question_id", "dimension"),
    "study_question": ("study_question", "question", "prompt"),
    "left_image_id": ("left_image_id", "left", "left_id", "image_left"),
    "right_image_id": ("right_image_id", "right", "right_id", "image_right"),
    "choice": ("choice", "answer", "winner", "response"),
    "timestamp": ("timestamp", "created_at", "date"),
    "day": ("day", "vote_day"),
    "time": ("time", "vote_time"),
    "city_left": ("city_left", "place_name_left", "left_city"),
    "city_right": ("city_right", "place_name_right", "right_city"),
    "longitude_left": ("longitude_left", "long_left", "lng_left"),
    "latitude_left": ("latitude_left", "lat_left"),
    "longitude_right": ("longitude_right", "long_right", "lng_right"),
    "latitude_right": ("latitude_right", "lat_right"),
}

CHOICE_ALIASES = {
    "left": "left",
    "l": "left",
    "1": "left",
    "right": "right",
    "r": "right",
    "2": "right",
    "equal": "equal",
    "tie": "equal",
    "same": "equal",
    "0": "equal",
}

QUESTION_ALIASES = {
    "safer": "safety",
    "safe": "safety",
    "livelier": "lively",
    "lively": "lively",
    "more beautiful": "beautiful",
    "beautiful": "beautiful",
    "wealthier": "wealthy",
    "wealthy": "wealthy",
    "more boring": "boring",
    "boring": "boring",
    "more depressing": "depressing",
    "depressing": "depressing",
}


def _read_table(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tab", ".tsv"}:
        separator = "\t" if suffix in {".tab", ".tsv"} else ","
        return pl.read_csv(path, separator=separator, infer_schema_length=10000)
    if suffix in {".parquet", ".pq"}:
        return pl.read_parquet(path)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return pl.read_ndjson(path) if suffix != ".json" else pl.read_json(path)
    raise ValueError(f"Unsupported table format: {path}")


def discover_votes_file(config: dict[str, Any]) -> Path:
    configured = config["data"].get("votes_file")
    if configured and Path(configured).exists():
        return Path(configured)
    raw_dir = Path(config["data"]["raw_dir"])
    candidates = [
        path
        for path in raw_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".csv", ".tsv", ".tab", ".json", ".jsonl", ".parquet"}
        and "vote" in path.name.lower()
    ]
    if not candidates:
        raise FileNotFoundError("No raw vote-level table found. Configure data.votes_file.")
    return max(candidates, key=lambda path: path.stat().st_size)


def _column_mapping(columns: list[str]) -> dict[str, str]:
    lowered = {column.lower(): column for column in columns}
    mapping = {}
    for canonical, aliases in ALIASES.items():
        for alias in aliases:
            if alias.lower() in lowered:
                mapping[lowered[alias.lower()]] = canonical
                break
    return mapping


def _normalise_id(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"null", "none", "nan", "na"} else None


def _hash_voter(value: Any) -> str | None:
    text = _normalise_id(value)
    if text is None:
        return None
    return hashlib.sha256(f"placepulse-cusp:v1:{text}".encode()).hexdigest()


def _normalise_string(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.String).str.strip_chars()
    return (
        pl.when(value.str.to_lowercase().is_in(["", "null", "none", "nan", "na"]))
        .then(None)
        .otherwise(value)
    )


def _optional_column(
    frame: pl.DataFrame, name: str, dtype: pl.DataType = pl.String
) -> pl.DataFrame:
    if name not in frame.columns:
        return frame.with_columns(pl.lit(None, dtype=dtype).alias(name))
    return frame


def standardise_votes(config: dict[str, Any]) -> tuple[Path, Path]:
    """Convert a raw export into the canonical vote table without row-wise Python work."""
    source = discover_votes_file(config)
    frame = _read_table(source).rename(_column_mapping(_read_table_columns(source)))
    required = {"study_id", "left_image_id", "right_image_id", "choice"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Raw vote table misses required fields: {sorted(missing)}")
    if "vote_id" not in frame.columns:
        frame = frame.with_row_index("vote_id")
    for name in (
        "voter_id",
        "timestamp",
        "study_question",
        "day",
        "time",
        "city_left",
        "city_right",
        "longitude_left",
        "latitude_left",
        "longitude_right",
        "latitude_right",
    ):
        frame = _optional_column(frame, name)

    dimension_map = {
        str(key).lower(): value
        for key, value in config["data"].get("dimension_map", {}).items()
    }
    generic_dimension_aliases = {
        "safe": "safety",
        "safer": "safety",
        "safety": "safety",
        "lively": "lively",
        "beautiful": "beautiful",
        "beauty": "beautiful",
        "wealth": "wealthy",
        "wealthy": "wealthy",
        "boring": "boring",
        "depressing": "depressing",
        "depress": "depressing",
        **dimension_map,
    }
    study_expr = _normalise_string("study_id").str.to_lowercase().replace_strict(
        generic_dimension_aliases, default=None, return_dtype=pl.String
    )
    question_expr = _normalise_string("study_question").str.to_lowercase().replace_strict(
        QUESTION_ALIASES, default=None, return_dtype=pl.String
    )
    timestamp_text = pl.coalesce(
        [
            _normalise_string("timestamp"),
            pl.concat_str(
                [_normalise_string("day"), _normalise_string("time")],
                separator=" ",
                ignore_nulls=False,
            ),
        ]
    )
    working = frame.select(
        _normalise_string("vote_id").alias("vote_id"),
        _normalise_string("voter_id").alias("voter_id_raw"),
        _normalise_string("study_id").alias("study_id"),
        pl.coalesce([question_expr, study_expr]).alias("dimension"),
        _normalise_string("left_image_id").alias("left_image_id"),
        _normalise_string("right_image_id").alias("right_image_id"),
        _normalise_string("choice")
        .str.to_lowercase()
        .replace_strict(CHOICE_ALIASES, default=None, return_dtype=pl.String)
        .alias("choice"),
        timestamp_text.alias("timestamp_raw"),
        _normalise_string("city_left").alias("city_left"),
        _normalise_string("city_right").alias("city_right"),
        pl.col("longitude_left").cast(pl.Float64, strict=False),
        pl.col("latitude_left").cast(pl.Float64, strict=False),
        pl.col("longitude_right").cast(pl.Float64, strict=False),
        pl.col("latitude_right").cast(pl.Float64, strict=False),
    ).with_columns(
        pl.when(
            pl.any_horizontal(
                pl.col("vote_id").is_null(),
                pl.col("left_image_id").is_null(),
                pl.col("right_image_id").is_null(),
                pl.col("choice").is_null() | ~pl.col("choice").is_in(CHOICES),
                pl.col("dimension").is_null() | ~pl.col("dimension").is_in(DIMENSIONS),
            )
        )
        .then(pl.lit("invalid_required_field"))
        .when(pl.col("left_image_id") == pl.col("right_image_id"))
        .then(pl.lit("self_comparison"))
        .otherwise(None)
        .alias("reason")
    )
    invalid = working.filter(pl.col("reason").is_not_null()).select("vote_id", "reason")
    valid = working.filter(pl.col("reason").is_null()).drop("reason")
    duplicate_audit = (
        valid.filter(pl.col("vote_id").is_duplicated())
        .select("vote_id")
        .with_columns(pl.lit("duplicate_vote_id").alias("reason"))
    )
    valid = valid.unique(subset=["vote_id"], keep="first", maintain_order=True)

    unique_voters = valid.select("voter_id_raw").drop_nulls().unique().with_columns(
        pl.col("voter_id_raw")
        .map_elements(_hash_voter, return_dtype=pl.String)
        .alias("voter_id")
    )
    votes = (
        valid.join(unique_voters, on="voter_id_raw", how="left")
        .drop("voter_id_raw")
        .with_columns(
            pl.col("timestamp_raw")
            .cast(pl.String)
            .str.to_datetime(strict=False, time_zone="UTC")
            .alias("timestamp")
        )
        .drop("timestamp_raw")
    )
    if votes.height:
        votes = _add_suspicious_flags(votes)
    audit_frame = pl.concat([invalid, duplicate_audit], how="vertical")

    interim = Path(config["data"]["interim_dir"])
    interim.mkdir(parents=True, exist_ok=True)
    votes_path = interim / "votes.parquet"
    audit_path = interim / "cleaning_audit.parquet"
    votes.write_parquet(votes_path)
    audit_frame.write_parquet(audit_path)
    write_json(
        interim / "standardisation.json",
        {
            "source": str(source),
            "rows_in": frame.height,
            "rows_out": votes.height,
            "dropped": audit_frame.height,
            "drop_reasons": dict(audit_frame.group_by("reason").len().iter_rows()),
            "city_metadata_available": bool(
                votes.height
                and votes["city_left"].is_not_null().any()
                and votes["city_right"].is_not_null().any()
            ),
        },
    )
    return votes_path, audit_path


def _read_table_columns(path: Path) -> list[str]:
    """Read only enough of a source to determine its original column names."""
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt", ".tab", ".tsv"}:
        separator = "\t" if suffix in {".tab", ".tsv"} else ","
        return pl.read_csv(path, separator=separator, n_rows=0).columns
    return _read_table(path).columns


def _add_suspicious_flags(votes: pl.DataFrame) -> pl.DataFrame:
    voter_stats = (
        votes.filter(pl.col("voter_id").is_not_null())
        .group_by("voter_id")
        .agg(
            pl.len().alias("n"),
            (pl.col("choice") == "left").mean().alias("left_rate"),
            (pl.col("choice") == "equal").mean().alias("equal_rate"),
            pl.col("timestamp")
            .sort()
            .diff()
            .dt.total_milliseconds()
            .median()
            .alias("median_ms"),
        )
        .with_columns(
            (
                (
                    (pl.col("n") >= 50)
                    & ((pl.col("left_rate") >= 0.98) | (pl.col("left_rate") <= 0.02))
                )
                | ((pl.col("n") >= 50) & (pl.col("median_ms") < 750))
            )
            .fill_null(False)
            .alias("suspicious")
        )
        .select("voter_id", "suspicious")
    )
    return votes.join(voter_stats, on="voter_id", how="left").with_columns(
        pl.col("suspicious").fill_null(False)
    )
