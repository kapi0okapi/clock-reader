.PHONY: install dev lint format test preview jobs models

install:  ## package + training extras
	pip install -e ".[train]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff format .
	ruff check --fix .

test:
	pytest -q

preview:  ## render 32 synthetic clocks with their masks into data/synth_preview
	clock-synth --out data/synth_preview --n 32

models:   ## pretrained encoders + YOLO detector
	bash scripts/download_models.sh

jobs:     ## render all Slurm scripts into slurm/generated/
	python3 slurm/generate.py --all
