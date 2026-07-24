from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from placepulse_cusp.provenance import metadata, write_json


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def find(self, item: str) -> str:
        if item not in self.parent:
            self.parent[item] = item
            self.size[item] = 1
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def validate_votes(config: dict[str, Any], votes_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(votes_path or Path(config["data"]["interim_dir"]) / "votes.parquet")
    if not path.exists():
        result = {
            "status": "DATA_INSUFFICIENT",
            "reasons": ["standardised_vote_table_missing"],
            "provenance": metadata(config),
        }
        _write(config, result)
        return result
    votes = pl.read_parquet(path)
    primary = config["data"]["primary_dimension"]
    safety = votes.filter(pl.col("dimension") == primary)
    identified = safety.filter(pl.col("voter_id").is_not_null())
    threshold = config["data"]["minimum"]["votes_per_repeat_voter"]
    repeat_voters = (
        identified.group_by("voter_id").len().filter(pl.col("len") >= threshold).height
        if identified.height
        else 0
    )
    union = UnionFind()
    for left, right in safety.select("left_image_id", "right_image_id").iter_rows():
        union.union(left, right)
    component_sizes = Counter(union.find(item) for item in union.parent)
    largest_fraction = (
        max(component_sizes.values(), default=0) / max(len(union.parent), 1)
    )
    minimum = config["data"]["minimum"]
    reasons = []
    if safety.height < minimum["safety_votes"]:
        reasons.append(f"safety_votes<{minimum['safety_votes']}")
    if identified.height < minimum["identified_safety_votes"]:
        reasons.append(f"identified_safety_votes<{minimum['identified_safety_votes']}")
    if repeat_voters < minimum["repeat_voters"]:
        reasons.append(f"repeat_voters<{minimum['repeat_voters']}")
    if largest_fraction < minimum["largest_component_fraction"]:
        reasons.append(f"largest_component_fraction<{minimum['largest_component_fraction']}")
    choice_counts = dict(votes.group_by("choice").len().iter_rows())
    dimension_counts = dict(votes.group_by("dimension").len().iter_rows())
    unexpected_choices = sorted(
        str(value) for value in set(choice_counts) - {"left", "right", "equal"}
    )
    unexpected_dimensions = sorted(
        str(value) for value in set(dimension_counts) - set(config["data"]["dimensions"])
    )
    if unexpected_choices:
        reasons.append(f"unexpected_choices:{','.join(unexpected_choices)}")
    if unexpected_dimensions:
        reasons.append(f"unexpected_dimensions:{','.join(unexpected_dimensions)}")
    result = {
        "status": "ok" if not reasons else "DATA_INSUFFICIENT",
        "reasons": reasons,
        "rows": votes.height,
        "choice_counts": choice_counts,
        "dimension_counts": dimension_counts,
        "identified_primary_votes": identified.height,
        "repeat_voters": repeat_voters,
        "images": len(union.parent),
        "largest_component_fraction": largest_fraction,
        "timestamp_missing_fraction": float(votes["timestamp"].is_null().mean()) if votes.height else 1.0,
        "suspicious_votes": int(votes["suspicious"].sum()) if votes.height else 0,
        "provenance": metadata(config, [path]),
    }
    _write(config, result)
    return result


def _write(config: dict[str, Any], result: dict[str, Any]) -> None:
    processed = Path(config["data"]["processed_dir"])
    processed.mkdir(parents=True, exist_ok=True)
    write_json(processed / "data_validation.json", result)
