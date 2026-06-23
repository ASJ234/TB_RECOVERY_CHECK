# TB Recovery Check

Machine learning models to predict tuberculosis treatment outcomes and identify at-risk household contacts, using cohort data from Kampala, Uganda.

## Aims

- **Aim 1**: Predict TB patients at baseline who will not sputum-culture convert at months 2 and 6 (non-converters at risk of treatment failure).
- **Aim 2**: Identify contacts of index-TB cases at risk of developing household TB disease, and predict contacts resistant to TB infection despite persistent exposure.

## Project Structure

```
data/
  raw/                          Original Excel files
  processed/                    Cleaned CSV datasets
notebooks/
  01_eda.ipynb                  Exploratory data analysis
  02_aim1_non_conversion.ipynb  Aim 1 modeling notebook
  03_aim2_contact_risk.ipynb    Aim 2 modeling notebook
src/
  data/
    load_data.py                Read and standardize Excel sheets
    clean_data.py               Clean, standardize, build target variables
    feature_engineering.py      Preprocessor pipelines, train/test splits
  models/
    train.py                    Train LR, Random Forest, XGBoost with CV + SMOTE
    predict.py                  Single and batch inference
    evaluate.py                 ROC/PR curves, confusion matrices, metrics
  pipeline/
    config.py                   Configuration management
    training_pipeline.py        Automated retraining orchestrator
  api/
    main.py                     FastAPI application (6 endpoints)
    schemas.py                  Pydantic request/response models
    dependencies.py             Model loader with caching
  deploy/
    start_ngrok.sh              FastAPI + ngrok tunnel launcher
models/
  registry/
    model_registry.csv          Version tracking across all training runs
    aim1_non_conversion/        Versioned model artifacts (.pkl)
    aim2_contact_risk/          Versioned model artifacts (.pkl)
  scalers/                      Saved preprocessors per version
  metadata/                     JSON with params, metrics, data/code hashes
tests/
  test_data.py                  11 tests for data loading and cleaning
  test_models.py                8 tests for training, prediction, evaluation
  test_pipeline.py              3 tests for config, API imports, hashing
.github/workflows/
  ci.yml                        Lint + test + train on pull request
  cd.yml                        Scheduled retraining + model validation
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full training pipeline
python -m src.pipeline.training_pipeline --aim all --force

# Start the API server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Deploy with ngrok tunnel
bash src/deploy/start_ngrok.sh
```

Or use the Makefile:

```bash
make install     # Create venv and install dependencies
make train       # Run full training pipeline
make api         # Start FastAPI at localhost:8000
make deploy      # Start API + ngrok tunnel
make test        # Run all tests
make eda         # Launch Jupyter notebooks
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with deployed model versions |
| `POST` | `/predict/aim1` | Predict non-conversion risk for a TB patient |
| `POST` | `/predict/aim2` | Predict contact risk |
| `POST` | `/predict/batch` | Batch predictions from a list of instances |
| `POST` | `/predict/csv` | Upload CSV for batch predictions |
| `GET` | `/model/info` | Champion model metadata (features, params, metrics) |
| `POST` | `/cache/clear` | Clear cached model instances |

Example prediction request:

```bash
curl -X POST http://localhost:8000/predict/aim1 \
  -H "Content-Type: application/json" \
  -d '{"SEX": "M", "AGE_YEARS": 42, "COUGH": "YES", "HIV_STATUS": "NEGATIVE"}'
```

## Model Version Control

Every training run produces versioned artifacts:

```
models/registry/
  model_registry.csv             # Full lineage: model, version, params_hash,
                                 #   data_hash, CV metrics, timestamp
  aim1_non_conversion/
    v1_logistic_regression.pkl
    v2_random_forest.pkl
    v3_xgboost.pkl              # Each retrain increments the version
```

- **Data hash** (SHA256): If input data changes, models are invalidated
- **Params hash**: Tracks exact hyperparameter configuration
- **Champion selection**: Best model by CV AUC is promoted for API serving
- **Registry CSV**: Single source of truth auditable across runs

## CI/CD

- **CI** (`.github/workflows/ci.yml`): Runs on PR — lint with `ruff`, run `pytest`, execute training pipeline
- **CD** (`.github/workflows/cd.yml`): Runs weekly or on merge to main — retrains models, archives artifacts, validates performance

## Deployment

The API can be served locally behind an ngrok tunnel for external access:

```bash
bash src/deploy/start_ngrok.sh
# Output: API is LIVE at https://abc123.ngrok.io
```

Docker:

```bash
docker build -t tb-recovery-check .
docker run -p 8000:8000 tb-recovery-check
```

## Data Sources

Two Excel files from the MTI-Plus cohort in Kampala, Uganda:

- `MTIPLUS dataset.xlsx` — Merged patient data with demographics, baseline symptoms, follow-up at months 2 and 5
- `MTIPLUS DEMOGRAPHIC DATA (1) (1).xlsx` — Five sheets: PATIENTS, Merged Dataset, FOLLOWUP, HEALTHY CONTACTS

## License

Research use only.
