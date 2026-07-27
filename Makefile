PYTHON ?= python
CONFIG ?= configs/confirmatory.yaml

.PHONY: sync test smoke all gpu-check gpu-benchmark preflight-mps preflight-cuda \
	confirmatory-mps confirmatory-cuda clean-results

sync:
	$(PYTHON) -m pip install -r requirements-dev.txt -e .

test:
	$(PYTHON) -m pytest

smoke:
	ppc simulate generate --config configs/smoke.yaml
	ppc run all --config configs/smoke.yaml

all:
	ppc run all --config $(CONFIG)

gpu-check:
	ppc gpu check --device $(DEVICE)

gpu-benchmark:
	ppc gpu benchmark --device $(DEVICE)

preflight-mps:
	ppc run heterogeneity --resume --config configs/real_preflight_mps.yaml

preflight-cuda:
	ppc run heterogeneity --resume --config configs/real_preflight_cuda.yaml

confirmatory-mps:
	ppc run all --resume --config configs/confirmatory_mps.yaml

confirmatory-cuda:
	ppc run all --resume --config configs/confirmatory_cuda.yaml

clean-results:
	ppc clean artifacts --config $(CONFIG)
