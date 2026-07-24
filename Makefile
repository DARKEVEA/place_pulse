PYTHON ?= uv run
CONFIG ?= configs/confirmatory.yaml

.PHONY: sync test smoke all clean-results

sync:
	uv sync --extra dev

test:
	$(PYTHON) pytest

smoke:
	$(PYTHON) ppc simulate generate --config configs/smoke.yaml
	$(PYTHON) ppc run all --config configs/smoke.yaml

all:
	$(PYTHON) ppc run all --config $(CONFIG)

clean-results:
	$(PYTHON) ppc clean artifacts --config $(CONFIG)

