.PHONY: install clean lint test train eda api deploy help train-monitor synthetic drift-check mlflow-ui mlflow-ui-bg simulate-stream simulate-list simulate-plot

VENV_DIR ?= venv
PYTHON := python3

help:
	@echo "TB Recovery Check - Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  install         Create venv and install dependencies"
	@echo "  lint            Run ruff linter"
	@echo "  test            Run pytest"
	@echo "  train           Run full training pipeline (Aim 1 + Aim 2)"
	@echo "  train-aim1      Train Aim 1 model only"
	@echo "  train-aim2      Train Aim 2 model only"
	@echo "  train-monitor   Run training + drift monitoring"
	@echo "  synthetic       Generate synthetic drift data (static Gaussian/swap)"
	@echo "  drift-check     Run drift check on synthetic vs reference"
	@echo "  simulate-stream Run 72-hour LLM-driven stream simulation (SCENARIO=name, PACE=1.0)"
	@echo "  simulate-all    Run all drift scenarios sequentially"
	@echo "  simulate-list   List available drift scenarios"
	@echo "  mlflow-ui       Launch MLflow UI for drift metrics"
	@echo "  eda             Launch Jupyter for EDA notebooks"
	@echo "  api             Start FastAPI on localhost:8000"
	@echo "  deploy          Start FastAPI + ngrok tunnel"
	@echo "  clean           Remove venv, caches, build artifacts"
	@echo "  data            Re-process raw data into clean CSVs"
	@echo "  xai             Generate XAI PDF reports for champion models"
	@echo "  xai-clean       Clean cached SHAP explainers"

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

train-monitor:
	python -m src.pipeline.training_pipeline --aim all --monitoring

synthetic:
	python -m src.monitoring.synthetic_drift

drift-check:
	PYTHONPATH=. python scripts/drift_check.py

simulate-stream:
	PYTHONPATH=. python -m src.simulation.stream_simulator --scenario $(or $(SCENARIO),gradual_age_shift) --aim $(or $(AIM),aim1) --pace $(or $(PACE),1.0)

simulate-all:
	PYTHONPATH=. python -m src.simulation.stream_simulator --scenario all --aim $(or $(AIM),aim1) --pace $(or $(PACE),1.0)

simulate-list:
	PYTHONPATH=. python -m src.simulation.stream_simulator --list-scenarios

mlflow-ui:
	@echo "Open http://127.0.0.1:5000 in your browser."
	@echo "Press Ctrl+C to stop."
	MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --cors-allowed-origins http://127.0.0.1:5000

mlflow-ui-bg:
	@echo "Starting MLflow UI in background at http://127.0.0.1:5000"
	MLFLOW_TRACKING_URI=sqlite:///mlruns/mlflow.db nohup mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --host 127.0.0.1 --port 5000 --cors-allowed-origins http://127.0.0.1:5000 > /tmp/mlflow-ui.log 2>&1 &
	@echo "PID: $$!"
	@echo "Logs: tail -f /tmp/mlflow-ui.log"
	@echo "Stop: kill $$!"

eda:
	. $(VENV_DIR)/bin/activate && jupyter notebook notebooks/

api:
	fuser -k 8000/tcp 2>/dev/null || true
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

deploy:
	bash src/deploy/start_ngrok.sh

clean:
	rm -rf $(VENV_DIR) __pycache__ .pytest_cache models/metadata/*.json
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true

xai:
	PYTHONPATH=. python scripts/generate_xai_reports.py

xai-clean:
	PYTHONPATH=. python scripts/_cleanup_explainers.py