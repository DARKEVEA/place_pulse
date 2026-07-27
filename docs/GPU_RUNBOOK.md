# GPU execution runbook

This project uses PyTorch for M0–M3 fitting. The two supported accelerator
profiles are:

- Apple silicon with the PyTorch `mps` backend, for local development and
  real-data preflight;
- an NVIDIA GPU with the PyTorch `cuda` backend, preferably Linux or WSL2,
  for the confirmatory run.

Polars/PyArrow data preparation, bootstrap aggregation, SciPy quadrature, and
report generation remain on CPU. They are not neural-model training workloads
and do not silently move an M0–M3 fit off the selected GPU. Both GPU
configurations request an explicit backend and fail immediately if it is
unavailable.

## 1. Common rules

Run commands from the repository root. Do not use
`configs/confirmatory.yaml` for a device-specific run; use one of:

```text
configs/real_preflight_mps.yaml
configs/real_preflight_cuda.yaml
configs/confirmatory_mps.yaml
configs/confirmatory_cuda.yaml
```

The MPS and CUDA profiles write to separate artifact directories. Checkpoint
names also contain the complete configuration hash, so a checkpoint produced
for one backend is never resumed by the other.

Before a long run:

```bash
python -m pytest
ppc data validate --config configs/confirmatory.yaml
ppc data prepare --resume --config configs/confirmatory.yaml
```

Do not run the MPS and CUDA profiles concurrently against the same artifact
directory. The supplied profiles already prevent that.

## 2. Apple silicon / MPS

Requirements:

- an Apple-silicon Mac;
- a current macOS release supported by the installed PyTorch build;
- a native arm64 Python, not an Intel Python under Rosetta.

Install and verify:

```bash
cd /Users/darkevea/code/place_pulse
conda create -n placepulse python=3.12
conda activate placepulse
python -m pip install -r requirements-dev.txt -e .
uname -m
python -c "import platform, torch; print(platform.machine()); print(torch.__version__); print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"
ppc gpu check --device mps --config configs/real_preflight_mps.yaml
ppc gpu benchmark --device mps --size 2048 --iterations 5 --config configs/real_preflight_mps.yaml
```

Both MPS booleans must be `True`, and the selected device in the JSON output
must be `mps`. An explicit MPS profile does not fall back to CPU.

Run the short, full-table preflight:

```bash
ppc run heterogeneity --resume --config configs/real_preflight_mps.yaml
```

This exercises nested selection, edge holdout, final refitting, voter holdout,
and time holdout with deliberately tiny epoch counts. Its estimates are
engineering diagnostics, not scientific results.

If MPS is the only available accelerator, start the confirmatory run with:

```bash
caffeinate -dimsu ppc run heterogeneity --resume --config configs/confirmatory_mps.yaml
```

After inspecting the Safety result, continue all preregistered replications:

```bash
caffeinate -dimsu ppc run all --resume --config configs/confirmatory_mps.yaml
```

MPS currently warns that some indexed reduction operations may be
nondeterministic. The pipeline still fixes every seed and records the selected
backend. For the primary confirmatory claim, CUDA is preferable because its
deterministic controls are more mature.

Do not set `PYTORCH_ENABLE_MPS_FALLBACK=1` for confirmatory work. That option
can move unsupported operations to CPU and obscures the requested
accelerator-only training policy.

## 3. RTX 3060 / CUDA (recommended confirmatory path)

Linux or Ubuntu under WSL2 is the supported deployment path. Install a
CUDA-enabled PyTorch build compatible with the host driver. A separate local
CUDA Toolkit is not required when the selected PyTorch wheel carries its CUDA
runtime dependencies.

First confirm that the operating system can see the GPU:

```bash
nvidia-smi
```

Then install the project:

```bash
cd /path/to/place_pulse
conda create -n placepulse python=3.12
conda activate placepulse
python -m pip install -r requirements-dev.txt -e .
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Record `python -m pip freeze` with each archived experiment so the exact
package versions and CUDA-enabled PyTorch build can be reproduced.

Put a Kaggle access token at `~/.kaggle/access_token`, restrict its
permissions, and either re-fetch the single vote table or copy the existing
data directories:

```bash
chmod 600 ~/.kaggle/access_token
ppc data fetch --config configs/confirmatory_cuda.yaml
ppc data validate --config configs/confirmatory_cuda.yaml
ppc data prepare --resume --config configs/confirmatory_cuda.yaml
```

The fetch command downloads `votes_clean.csv`, not the 3 GB image archive.

Verify CUDA and benchmark it:

```bash
CUDA_VISIBLE_DEVICES=0 ppc gpu check --device cuda --config configs/real_preflight_cuda.yaml
CUDA_VISIBLE_DEVICES=0 ppc gpu benchmark --device cuda --size 4096 --iterations 10 --config configs/real_preflight_cuda.yaml
```

The output must report `selected: cuda`, an RTX 3060 device name, its compute
capability, and available GPU memory.

Run the full-table engineering preflight:

```bash
CUDA_VISIBLE_DEVICES=0 ppc run heterogeneity --resume --config configs/real_preflight_cuda.yaml
```

In another terminal, monitor utilization and memory:

```bash
nvidia-smi --loop=2
```

For the confirmatory Safety analysis:

```bash
CUDA_VISIBLE_DEVICES=0 ppc run heterogeneity --resume --config configs/confirmatory_cuda.yaml
```

Then complete the remaining dimensions, gates, CUSP comparison, and report:

```bash
CUDA_VISIBLE_DEVICES=0 ppc run all --resume --config configs/confirmatory_cuda.yaml
```

Every completed outer fold is checkpointed. If power or the process is
interrupted, repeat the same command with `--resume`. Do not change the config
mid-run: any change creates a new configuration hash and therefore a new
checkpoint lineage.

On a 12 GB RTX 3060, the current full-batch tensors fit comfortably in the
observed preflight footprint. If CUDA reports allocator fragmentation rather
than true exhaustion, retry from the same checkpoint with:

```bash
PYTORCH_ALLOC_CONF=backend:cudaMallocAsync CUDA_VISIBLE_DEVICES=0 ppc run heterogeneity --resume --config configs/confirmatory_cuda.yaml
```

Do not lower epochs, folds, random starts, bootstrap counts, or candidate
models in the confirmatory profile to work around runtime. Those values are
part of the preregistered analysis.

## 4. Reading the outputs

Device-specific artifacts are isolated at:

```text
artifacts/real_preflight_mps/
artifacts/real_preflight_cuda/
artifacts/mps/
artifacts/cuda/
```

The model-comparison JSON records the requested/selected backend, PyTorch
version, platform, and GPU details under `provenance.compute`. Confirmatory
CUDA outputs of primary interest are:

```text
artifacts/cuda/metrics/safety_model_comparison.json
artifacts/cuda/report/verdict.json
artifacts/cuda/report/experiment_report.html
```

Preflight outputs must never be copied into the confirmatory artifact
directory or cited as scientific evidence.

## 5. Failure diagnosis

- `CUDA was requested but ... false`: inspect `nvidia-smi`; verify that this
  environment installed a CUDA-enabled PyTorch Linux wheel.
- `MPS was requested but is unavailable`: verify native arm64 Python and run
  outside a restricted/sandboxed process.
- CUDA out of memory: stop other GPU processes shown by `nvidia-smi`; resume
  from the last fold; use `cudaMallocAsync` only for fragmentation.
- A resumed command refits all folds: the configuration, code version, or
  result schema changed. Preserve the old artifacts rather than renaming their
  checkpoints into the new lineage.
- Low GPU utilization between fits is expected during Parquet loading,
  encoding, bootstrapping, SciPy density fitting, and report generation.
