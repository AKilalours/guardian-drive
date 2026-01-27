PY := python

.PHONY: install dev lint format test train eval bench clean

install:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e .

dev:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

train:
	$(PY) -m guardian_drive.train.train_physio --config configs/wesad_physio.yaml

eval:
	$(PY) -m guardian_drive.eval.eval_physio --config configs/wesad_physio.yaml

bench:
	$(PY) -m guardian_drive.bench.bench_physio --config configs/wesad_physio.yaml

clean:
	rm -rf .pytest_cache .ruff_cache __pycache__ **/__pycache__ dist build *.egg-info reports/*

sqi_sweep:
	$(PY) -m guardian_drive.eval.eval_sqi_sweep --config configs/wesad_physio.yaml

multi_eval:
	$(PY) -m guardian_drive.eval.eval_multi_subject --config configs/wesad_physio.yaml

