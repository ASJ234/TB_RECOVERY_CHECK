.PHONY: install clean lint test train eda api deploy help

VENV_DIR ?= venv
PYTHON := python3

help:
	@echo "TB Recovery Check - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install     Create venv and install dependencies"
	@echo "  lint        Run ruff linter"
	@echo "  test        Run pytest"
	@echo "  train       Run full training pipeline (Aim 1 + Aim 2)"
	@echo "  train-aim1  Train Aim 1 model only"
	@echo "  train-aim2  Train Aim 2 model only"
	@echo "  eda         Launch Jupyter for EDA notebooks"
	@echo "  api         Start FastAPI on localhost:8000"
	@echo "  deploy      Start FastAPI + ngrok tunnel"
	@echo "  clean       Remove venv, caches, build artifacts"
	@echo "  data        Re-process raw data into clean CSVs"

install:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

lint:
	ruff check src/ tests/

test:
	python -m pytest tests/ -v --tb=short

train:
	python -m src.pipeline.training_pipeline --aim all --force

train-aim1:
	python -m src.pipeline.training_pipeline --aim 1 --force

train-aim2:
	python -m src.pipeline.training_pipeline --aim 2 --force

data:
	python -m src.data.clean_data

eda:
	. $(VENV_DIR)/bin/activate && jupyter notebook notebooks/

api:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

deploy:
	bash src/deploy/start_ngrok.sh

clean:
	rm -rf $(VENV_DIR) __pycache__ .pytest_cache models/metadata/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true
