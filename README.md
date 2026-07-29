# Place Pulse CUSP

Reproducible analysis of whether a single shared Place Pulse ranking is
predictively sufficient. The pipeline preserves ties and voter histories,
compares scalar, continuous-preference, and latent-class Davidson models, and
only evaluates stochastic CUSP geometry after preregistered heterogeneity and
bimodality gates pass.

## Quick start

```powershell
conda activate arch
python -m pip install -r requirements-dev.txt -e .
ppc simulate generate --config configs/smoke.yaml
ppc run all --config configs/smoke.yaml
python -m pytest
```

For real data, place an official Place Pulse vote export in `data/raw/` and set
`data.local_source` in `configs/confirmatory.yaml`, then run:

```bash
ppc data fetch --config configs/confirmatory.yaml
ppc run all --config configs/confirmatory.yaml
```

## Kaggle real-data pipeline

The confirmatory configuration is pinned to Kaggle dataset
`shubham6147/mit-place-pulse`, version 2. It selectively downloads only
`votes_clean.csv` (about 481 MB), not the 110,688 street-view JPEGs.

Store a Kaggle API access token at `~/.kaggle/access_token` with mode `600`,
then run:

```bash
python -m pip install -r requirements-dev.txt -e .
ppc data fetch --config configs/confirmatory.yaml
ppc data validate --config configs/confirmatory.yaml
ppc data prepare --resume --config configs/confirmatory.yaml
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
ppc run all --config configs/confirmatory.yaml
```

This command is compute-intensive on the 1.56-million-vote table and should be
started only after reviewing `data/processed/data_validation.json`.

The pipeline never silently substitutes aggregate image scores for raw votes.
If voter-linked comparison records are unavailable, it emits
`DATA_INSUFFICIENT` with machine-readable reasons.

See `artifacts/report/experiment_report.html` and
`artifacts/report/verdict.json` after a run.

## Revised calibration and validation workflow

The first completed CUDA run is retained under `artifacts/cuda` as
`RUN_001_DIAGNOSTIC`. It must not be interpreted as confirmatory evidence.
The revised protocol first calibrates M0--M3 on synthetic data and then runs
Safety only:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ppc simulate validate-models --config configs/calibration_cuda.yaml
ppc simulate validate-density --config configs/calibration_cuda.yaml
ppc run heterogeneity --config configs/calibration_cuda.yaml
```

Review `artifacts/run_002_calibration` before freezing the code and config.
Only after every calibration gate passes should the six-dimension internal
validation be run:

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
ppc run all --config configs/revised_validation_cuda.yaml
```

`RUN_003_REVISED_VALIDATION` is an internal revised validation because the
first run informed the method changes. A new or independent dataset is
required for a future confirmatory claim.

## GPU execution

Explicit MPS and CUDA profiles are provided. They fail if the requested
accelerator is unavailable instead of silently falling back to CPU:

```bash
ppc gpu check --device mps --config configs/real_preflight_mps.yaml
ppc gpu check --device cuda --config configs/real_preflight_cuda.yaml
ppc run heterogeneity --resume --config configs/real_preflight_mps.yaml
ppc run heterogeneity --resume --config configs/real_preflight_cuda.yaml
```

Use Apple MPS for local preflight and an NVIDIA CUDA host for the primary
confirmatory run. See [the GPU execution runbook](docs/GPU_RUNBOOK.md) for
installation, verification, monitoring, resume, and failure-recovery commands.
