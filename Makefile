PYTHON ?= uv run
CONFIG ?= configs/confirmatory.yaml

.PHONY: sync test smoke all gpu-check gpu-benchmark preflight-mps preflight-cuda \
	confirmatory-mps confirmatory-cuda clean-results

sync:
	uv sync --extra dev

test:
	$(PYTHON) pytest

smoke:
	$(PYTHON) ppc simulate generate --config configs/smoke.yaml
	$(PYTHON) ppc run all --config configs/smoke.yaml

all:
	$(PYTHON) ppc run all --config $(CONFIG)

gpu-check:
	$(PYTHON) ppc gpu check --device $(DEVICE)

gpu-benchmark:
	$(PYTHON) ppc gpu benchmark --device $(DEVICE)

preflight-mps:
	$(PYTHON) ppc run heterogeneity --resume --config configs/real_preflight_mps.yaml

preflight-cuda:
	$(PYTHON) ppc run heterogeneity --resume --config configs/real_preflight_cuda.yaml

confirmatory-mps:
	$(PYTHON) ppc run all --resume --config configs/confirmatory_mps.yaml

confirmatory-cuda:
	$(PYTHON) ppc run all --resume --config configs/confirmatory_cuda.yaml

clean-results:
	$(PYTHON) ppc clean artifacts --config $(CONFIG)
