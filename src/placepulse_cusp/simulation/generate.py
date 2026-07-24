from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from placepulse_cusp.constants import DIMENSIONS


def _draw_choice(delta: np.ndarray, log_tie: float, rng: np.random.Generator) -> np.ndarray:
    logits = np.column_stack(
        [delta / 2, -delta / 2, np.full(len(delta), np.log(2.0) + log_tie)]
    )
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    draws = rng.random(len(delta))
    cumulative = probabilities.cumsum(1)
    return (draws[:, None] > cumulative).sum(1)


def generate_vote_table(
    config: dict[str, Any],
    *,
    mechanism: str = "mixture",
    output: str | Path | None = None,
) -> Path:
    sim = config["simulation"]
    seed = config["project"]["seed"]
    rng = np.random.default_rng(seed)
    n_voters, n_images, total_votes = sim["voters"], sim["images"], sim["votes"]
    voters = np.asarray([f"voter_{i:05d}" for i in range(n_voters)])
    images = np.asarray([f"image_{i:05d}" for i in range(n_images)])
    classes = rng.choice(3, n_voters, p=[0.48, 0.34, 0.18])
    base = {dimension: rng.normal(0, 1, n_images) for dimension in DIMENSIONS}
    class_shift = rng.normal(0, 0.7, (3, n_images))
    rows = []
    start = datetime(2015, 1, 1, tzinfo=timezone.utc)
    per_dimension = max(total_votes // len(DIMENSIONS), 1)
    labels = np.array(["left", "right", "equal"])
    for dimension_index, dimension in enumerate(DIMENSIONS):
        voter_index = rng.integers(0, n_voters, per_dimension)
        left = rng.integers(0, n_images, per_dimension)
        right = rng.integers(0, n_images, per_dimension)
        same = left == right
        right[same] = (right[same] + 1) % n_images
        utility = base[dimension].copy()
        delta = utility[left] - utility[right]
        if mechanism in {"mixture", "cusp"} and dimension == "safety":
            delta += class_shift[classes[voter_index], left] - class_shift[
                classes[voter_index], right
            ]
        elif mechanism == "continuous" and dimension == "safety":
            z = rng.normal(size=n_voters)
            q = rng.normal(scale=0.6, size=n_images)
            delta += z[voter_index] * (q[left] - q[right])
        choices = labels[_draw_choice(delta, -1.1, rng)]
        for j in range(per_dimension):
            rows.append(
                {
                    "vote_id": f"{dimension_index}_{j}",
                    "voter_uniqueid": voters[voter_index[j]],
                    "study_id": dimension,
                    "left": images[left[j]],
                    "right": images[right[j]],
                    "choice": choices[j],
                    "timestamp": (start + timedelta(seconds=len(rows) * 17)).isoformat(),
                }
            )
    target = Path(output or config["data"].get("votes_file") or "data/raw/smoke_votes.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(target)
    return target

