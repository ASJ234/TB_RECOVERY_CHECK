# TB Recovery Check

Machine learning models to predict tuberculosis treatment outcomes and identify at-risk household contacts, using cohort data from Kampala, Uganda.

## Aims

- **Aim 1**: Predict TB patients at baseline who will not sputum-culture convert at months 2 and 5 (non-converters at risk of treatment failure). Two modeling strategies:
  - *Strict*: Culture-based labels only (n=11), LR + LOOCV, 4 features — exploratory only
  - *Imputed*: Missing outcomes assumed converted (n=218), full pipeline — sensitivity analysis
- **Aim 2**: Identify contacts of index-TB cases at risk of developing household TB disease, and predict contacts resistant to TB infection despite persistent exposure.

## Project Structure

```
data/
  raw/                              Original Excel files
  cleaned/                          Cleaned CSV datasets
    aim1_patients.csv                    Raw-cleaned (target still has NaN where unknown)
    aim1_patients_strict.csv             Strict M2 culture labels only (n=11, no imputation)
    aim1_patients_imputed.csv            Missing values filled with median/mode
    aim2_contacts.csv                   Raw-cleaned
    aim2_contacts_imputed.csv            Missing values filled with median/mode
    followup.csv                        Cleaned follow-up visits
notebooks/
  01_eda.ipynb                      Exploratory data analysis
  02_aim1_non_conversion.ipynb      Aim 1 modeling notebook
  03_aim2_contact_risk.ipynb        Aim 2 modeling notebook
src/
  data/
    load_data.py                    Read and standardize Excel sheets
    clean_data.py                   Clean, standardize, build target variables
    feature_engineering.py          Preprocessor pipelines, train/test splits
  models/
    train.py                        Train LR, RF, XGBoost + strict LR with LOOCV
    predict.py                      Single and batch inference
    evaluate.py                     ROC/PR curves, LOOCV per-fold tables, metrics
  pipeline/
    config.py                       Configuration management
    training_pipeline.py            Automated retraining orchestrator
  api/
    main.py                         FastAPI application (7+ endpoints)
    schemas.py                      Pydantic request/response models
    dependencies.py                 Model loader with caching
  deploy/
    start_ngrok.sh                  FastAPI + ngrok tunnel launcher
models/
  registry/
    model_registry.csv              Version tracking across all training runs
    aim1_non_conversion/            Versioned imputed model artifacts (.pkl)
    aim1_non_conversion_strict/     Versioned strict model artifacts (.pkl)
    aim2_contact_risk/              Versioned model artifacts (.pkl)
  scalers/                          Saved preprocessors per version
  metadata/                         JSON with params, metrics, data/code hashes, LOOCV reports
tests/
  test_data.py                      15 tests for data loading and cleaning
  test_models.py                    12 tests for training, prediction, evaluation
  test_pipeline.py                  4 tests for config, API imports, hashing
.github/workflows/
  ci.yml                            Lint + test + train on pull request
  cd.yml                            Scheduled retraining + model validation
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

## Aim 1 Modeling Strategy

Aim 1 uses a **two-phase approach** to handle extreme label sparsity (only 11 of 218 patients have M2 culture results).

### Phase 1 — Strict Model (Exploratory)

| Component | Detail |
|---|---|
| **Samples** | 11 patients with confirmed M2 culture results |
| **Outcome** | Culture-based: `No Growth` / `Negative` → 0 (converted), `Positive` → 1 (non-conversion) |
| **Features** | `AGE (YEARS)`, `BMI`, `SEX`, `baseline_symptom_count` (sum of 7 baseline symptoms) |
| **Model** | Logistic regression only, L2 regularization (`C=1.0`, `class_weight="balanced"`) |
| **Cross-validation** | Leave-one-out (LOOCV) — each fold trains on 10 samples, tests on 1 |
| **No SMOTE** | Rejected at n=11 — synthetic samples amplify noise |
| **Purpose** | Exploratory only, not clinically deployable. Generates hypotheses about feature directions. |

Per-fold predictions, aggregate metrics (accuracy, precision, recall, F1, AUC-ROC), and a confusion matrix are saved as a JSON report in `models/metadata/aim1_non_conversion/loocv_*.json`.

### Phase 2 — Imputed Model (Sensitivity Analysis)

| Component | Detail |
|---|---|
| **Samples** | All 218 patients (207 unlabeled imputed as converted) |
| **Outcome** | `TARGET_NON_CONVERSION_ANY` — M2/M5 culture max, missing filled as 0 |
| **Features** | 18 baseline features (demographics, symptoms, comorbidities) |
| **Models** | Logistic Regression, Random Forest, XGBoost |
| **Cross-validation** | 5-fold stratified (or LOOCV if n < 20) |
| **SMOTE** | Optional, enabled via config |
| **Purpose** | Sensitivity analysis only. 97%+ accuracy is driven by imputed majority class — do not use clinically. |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check with deployed model versions |
| `POST` | `/predict/aim1` | Predict non-conversion risk (imputed model, 18 features) |
| `POST` | `/predict/aim1/strict` | Predict non-conversion risk (strict model, 4 features) |
| `POST` | `/predict/aim2` | Predict contact risk |
| `POST` | `/predict/batch` | Batch predictions from a list of instances |
| `POST` | `/predict/csv` | Upload CSV for batch predictions |
| `GET` | `/model/info` | Champion model metadata (features, params, metrics) |
| `POST` | `/cache/clear` | Clear cached model instances |

Example prediction requests:

```bash
# Aim 1 — imputed model (18 features)
curl -X POST http://localhost:8000/predict/aim1 \
  -H "Content-Type: application/json" \
  -d '{"SEX": "M", "AGE_YEARS": 42, "COUGH": "YES", "HIV_STATUS": "NEGATIVE"}'

# Aim 1 — strict model (4 features)
curl -X POST http://localhost:8000/predict/aim1/strict \
  -H "Content-Type: application/json" \
  -d '{"SEX": "M", "AGE_YEARS": 42, "BMI": 22.5, "baseline_symptom_count": 3}'
```

## Model Version Control

Every training run produces versioned artifacts:

```
models/registry/
  model_registry.csv                  # Full lineage: model, version, params_hash,
                                      #   data_hash, CV metrics, timestamp
  aim1_non_conversion/                # Imputed model artifacts (sensitivity)
    v1_logistic_regression.pkl
    v2_random_forest.pkl
    v3_xgboost.pkl
  aim1_non_conversion_strict/         # Strict model artifacts (exploratory)
    v1_strict_logistic.pkl
```

- **Data hash** (SHA256): If input data changes, models are invalidated
- **Params hash**: Tracks exact hyperparameter configuration
- **Champion selection**: Best model by CV AUC is promoted for API serving (imputed); strict model is served from its own registry
- **Registry CSV**: Single source of truth auditable across runs
- **LOOCV reports**: Per-fold predictions and aggregate metrics saved as JSON in `models/metadata/aim1_non_conversion/loocv_*.json`

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
