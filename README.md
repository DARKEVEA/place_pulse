# Place Pulse CUSP

Reproducible analysis of whether a single shared Place Pulse ranking is
predictively sufficient. The pipeline preserves ties and voter histories,
compares scalar, continuous-preference, and latent-class Davidson models, and
only evaluates stochastic CUSP geometry after preregistered heterogeneity and
bimodality gates pass.

## Quick start

```bash
uv sync --extra dev
uv run ppc simulate generate --config configs/smoke.yaml
uv run ppc run all --config configs/smoke.yaml
uv run pytest
```

For real data, place an official Place Pulse vote export in `data/raw/` and set
`data.local_source` in `configs/confirmatory.yaml`, then run:

```bash
uv run ppc data fetch --config configs/confirmatory.yaml
uv run ppc run all --config configs/confirmatory.yaml
```

## Kaggle real-data pipeline

The confirmatory configuration is pinned to Kaggle dataset
`shubham6147/mit-place-pulse`, version 2. It selectively downloads only
`votes_clean.csv` (about 481 MB), not the 110,688 street-view JPEGs.

Store a Kaggle API access token at `~/.kaggle/access_token` with mode `600`,
then run:

```bash
uv sync --extra dev
uv run ppc data fetch --config configs/confirmatory.yaml
uv run ppc data validate --config configs/confirmatory.yaml
uv run ppc data prepare --resume --config configs/confirmatory.yaml
```

`data validate` regenerates the canonical table by default, preventing a stale
smoke-data artifact from being mistaken for real data. Use `--resume` only
after the raw-file hash and standardisation manifest have already been checked.
The adapter maps `study_question` to the six dimensions, combines `day` and
`time` into UTC timestamps, hashes `voter_uniqueid`, preserves `equal`, and
retains city and coordinate fields. Invalid questions and choices are written
to `data/interim/cleaning_audit.parquet`.

The complete confirmatory model run is intentionally separate:

```bash
uv run ppc run all --config configs/confirmatory.yaml
```

This command is compute-intensive on the 1.56-million-vote table and should be
started only after reviewing `data/processed/data_validation.json`.

The pipeline never silently substitutes aggregate image scores for raw votes.
If voter-linked comparison records are unavailable, it emits
`DATA_INSUFFICIENT` with machine-readable reasons.

See `artifacts/report/experiment_report.html` and
`artifacts/report/verdict.json` after a run.
