# Design Decisions

This document records the questions asked, answers given, and rationale behind the key design decisions for this project.

---

## Question 1: Aim 1 Target Data — Sputum Culture Conversion

**Asked**: During initial planning, after discovering that only 11 out of 218 patients had 2-month follow-up data and only 3 had 5-month data.

**Question**: "The 2-month and 5-month sputum culture conversion columns have very few records (11 and 3 out of 218 patients). Do you have additional FEND-TB cohort data (mentioned in the proposal) that contains more conversion outcomes? Or should we work with what's available and treat this as a small-data problem?"

**Options given**:
1. **Work with existing data only** — treat as small-data problem with strong regularization
2. Use proxy target from baseline culture result
3. FEND-TB data is available separately

**Answer**: Work with existing data only.

**Rationale**: The FEND-TB dataset was not available. The target was redefined to only label rows with actual culture results (`pd.NA` for unknown outcomes), resulting in 11 labeled samples (5 converters, 6 non-converters). Model training uses Leave-One-Out CV, simple models with strong regularization, and bootstrapped confidence intervals.

**Implementation impact**:
- `clean_data.py`: Target mapping changed from `NaN → 0` to `NaN → NA`
- `train.py`: SMOTE disabled when minority class < 2 samples in a fold
- Pipeline handles small data gracefully, falling back to simpler CV strategies

---

## Question 2: Aim 2 Target Definition — Contact Risk

**Asked**: After analyzing the HEALTHY CONTACTS sheet which had 46 entries but no outcome column indicating who developed active TB.

**Question**: "For Aim 2 (predicting contacts who develop TB or are resistant), the HEALTHY CONTACTS sheet (46 rows) has no outcome column indicating who developed TB. How should we define the target variable?"

**Options given**:
1. **Use symptom presence as proxy** — any symptom (cough, fever, etc.) = at-risk
2. Link contacts to patient outcomes via household ID
3. Define as unsupervised anomaly detection

**Answer**: Use symptom presence as proxy.

**Rationale**: Without longitudinal follow-up data showing which contacts actually developed TB, symptom presence is the most defensible proxy. Asymptomatic contacts despite household exposure to M. tuberculosis are labeled as "potentially resistant"; symptomatic contacts are labeled as "at-risk." This creates a binary classification with ~10 positive and ~36 negative samples.

**Implementation impact**:
- `clean_data.py`: `TARGET_SYMPTOM_PRESENT` derived from 7 symptom columns
- Notebook explicitly documents the proxy limitation
- Features and target share symptom columns, so perfect separation is expected (and noted)

---

## Question 3: Technology Stack

**Asked**: After defining both aims and understanding the data constraints.

**Question**: "What ML framework and deployment target do you prefer?"

**Options given**:
1. **scikit-learn + XGBoost** — classic ML, simpler deployment
2. AutoML (PyCaret/TPOT) — automated model selection
3. Deploy as FastAPI endpoint — REST API serving
4. Model artifacts only

**Answer**: scikit-learn + XGBoost.

**Rationale**: scikit-learn provides reliable preprocessing and baseline models; XGBoost offers gradient boosting for the classification tasks. The combination is well-suited for small tabular data, has excellent documentation, and deploys easily via FastAPI.

**Implementation impact**:
- `requirements.txt`: scikit-learn, xgboost, imbalanced-learn, fastapi
- `train.py`: Three model classes — LogisticRegression, RandomForestClassifier, XGBClassifier
- `api/`: FastAPI with 6 endpoints, champion model selection by CV AUC

---

## Question 4: Model Version Control

**Asked**: While presenting the initial project plan, the user requested explicit version control for models.

**Decision**: Implemented lightweight MLflow-style versioning without a server.

**Implementation**:
- `models/registry/model_registry.csv`: Central registry tracking model, version, parameters hash, data hash, CV metrics, timestamp
- Semantic versioning (`v1`, `v2`, `v3`...) auto-incremented per training run
- SHA256 hashes of data and parameters for reproducibility
- Metadata JSON files stored alongside model artifacts
- Previous model versions are never overwritten

---

## Question 5: Deployment Target

**Asked**: After establishing the version control approach.

**Decision**: Local deployment with ngrok tunnel.

**Implementation**:
- `src/api/main.py`: FastAPI app served on localhost:8000
- `src/deploy/start_ngrok.sh`: Script to start API + ngrok tunnel
- `Dockerfile`: Multi-stage build with ngrok binary included
- CI/CD skip-deploy by default (requires manual trigger)
