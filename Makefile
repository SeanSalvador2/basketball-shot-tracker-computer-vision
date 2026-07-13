# Basketball Shot Tracker — Stage A build/run entrypoints.
# Every command is reproducible: fixed seeds live in configs, figures regenerate
# from committed configs (see docs/REPRODUCING.md).
#
# Install uses a project-local virtualenv (.venv) to stay isolated from system packages.
# torch/torchvision come from the CPU wheel index (no multi-GB CUDA download). NOTE: the
# Stage-A build container firewalls that index; see reports/phase1_pipeline.md for the
# documented fallback used there. On a normal machine `make setup` is the clean path.

VENV ?= .venv
PY ?= $(VENV)/bin/python
TORCH_CPU_INDEX = https://download.pytorch.org/whl/cpu

.PHONY: help setup test demo eda ablations reports synth clean

help:
	@echo "Targets:"
	@echo "  setup      Create .venv, install CPU torch + the bball package (editable, dev extras)"
	@echo "  test       Run the pytest suite (geometry, FSM, bridging, leakage guards)"
	@echo "  synth      Generate the committed synthetic session bundle (EDA/ablations/demo input)"
	@echo "  eda        Run every EDA analysis -> reports/figures/eda/ + reports/phase1_eda.md"
	@echo "  ablations  Run the non-droppable ablation matrix (A1 A5 A6 A7 A8 A9) under MLflow"
	@echo "  demo       Execute notebooks/demo.ipynb top-to-bottom (nbconvert --execute)"
	@echo "  reports    Regenerate report figures from tracked runs"
	@echo "  clean      Remove caches and generated artifacts (keeps committed figures)"

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install --index-url $(TORCH_CPU_INDEX) torch==2.8.0 torchvision==0.23.0
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

synth:
	$(PY) -m bball.synth.build_bundle --config configs/synth_bundle.yaml

eda:
	$(PY) -m bball.eval.run_eda --config configs/eda.yaml

ablations:
	$(PY) scripts/run_ablations.py --which A1 A5 A6 A7 A8 A9

demo:
	$(PY) -m nbconvert --to notebook --execute --inplace \
		--ExecutePreprocessor.timeout=1800 notebooks/demo.ipynb

# Regenerate every committed report figure (EDA + ablations) from configs + seeds.
reports: eda ablations

clean:
	rm -rf .pytest_cache **/__pycache__ src/**/__pycache__ outputs/ .cache/
	find . -name '*.pyc' -delete

app:  ## run the local web workbench (calibrate / label / zones / results)
	.venv/bin/uvicorn bball.app.server:app --host 0.0.0.0 --port 8000
