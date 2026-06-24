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

**Rationale**: The FEND-TB dataset was not available. Two complementary strategies were adopted:

- **Strict labels** (exploratory): 11 patients with M2 culture results mapped via culture-based logic (`No Growth`/`Negative` → 0, `Positive` → 1). No imputation. Used for LOOCV with LR only.
- **Imputed labels** (sensitivity): All 218 patients, with NaN targets filled as 0 (conservative clinical assumption). Allows full pipeline with RF/XGBoost.

**Implementation impact**:
- `clean_data.py`: `build_aim1_strict_dataset()` builds strict M2 culture labels with `baseline_symptom_count` and `M2_DISCORDANT` flags; `build_aim1_dataset()` fills NaN targets as 0 for the imputed path
- `feature_engineering.py`: Two pipelines — `prepare_aim1_strict_data()` (4 features, no test split) and `prepare_aim1_data()` (18 features, standard split)
- `train.py`: `train_aim1_strict()` runs LR + LOOCV with per-fold tracking; `train_models()` handles RF/XGBoost for the imputed set
- SMOTE disabled when minority class < 2 samples in a fold

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

---

## Question 6: Imputing Missing Aim 1 Targets as Negative — Two-Phase Strategy

**Asked**: After training the pipeline on only 11 labeled Aim 1 samples, all three models produced `CV AUC=nan` due to SMOTE crashing in CV folds (minority class of 2 could not provide `k_neighbors=3` for SMOTE). Test set evaluation on 3 samples was meaningless.

**Question**: "What other options are available to achieve the set objectives?"

**Options given**:
1. **Use all data for training** — set `test_size=0`, disable SMOTE, rely on CV
2. Disable SMOTE only, keep test split
3. **Impute missing targets as negative** — assume patients lost to follow-up converted (conservative clinical assumption), yielding 218 labeled rows
4. Keep existing approach with stronger manual regularization

**Answer**: Two-phase strategy combining options 1 and 3:

- **Phase 1 (strict)**: Only the 11 culture-labeled patients, LR + LOOCV, 4 features — exploratory only
- **Phase 2 (imputed)**: All 218 patients, NaN→0, LR/RF/XGBoost, 5-fold CV — sensitivity analysis

**Rationale**: The strict model is the most honest estimate of model performance given the data — every prediction is tested on held-out real labels. However, with n=11, the variance is extreme (AUC-ROC near 0.5, metrics flip with single-sample changes). The imputed model provides a larger-sample estimate but with severe label noise (assuming all 207 unlabeled patients converted). Running both and comparing coefficient directions determines whether the signal from the 11 labeled patients is genuine.

**Key constraints for the strict model**:
- Only 4 features (AGE, BMI, SEX, `baseline_symptom_count`) to respect the one-in-ten rule (~0.6 events per predictor with 6 failures)
- Logistic regression only (RF/XGBoost need n>100 per class)
- No SMOTE (synthetic samples amplify noise at n=11)
- L2 regularization (C=1.0) with `class_weight="balanced"`
- LOOCV forced (each fold leaves out 1 of 11, trains on 10)

**Risks**:
- **Strict model**: AUC-ROC is inherently near-random at n=11. The model generates hypotheses, not predictions.
- **Imputed model**: 97%+ accuracy is driven by the imputed majority class. Precision on the failure class is ~44%. Do not deploy clinically.
- **Evaluation**: The strict model's per-fold predictions are more informative than aggregate metrics — examining which individual patients were misclassified reveals where more data is needed.

**Implementation impact**:
- `clean_data.py`: `build_aim1_dataset()` fills NaN targets with 0, adds `IS_IMPUTED` flag; `build_aim1_strict_dataset()` builds culture-based labels only, adds `baseline_symptom_count` and `M2_DISCORDANT`
- `config/pipeline_config.json`: `aim1.test_size: 0.0`, `aim1.use_smote: false`, plus strict feature/target config
- `training_pipeline.py`: `run_aim1()` runs Phase 1 (strict) then Phase 2 (imputed) sequentially
- `evaluate.py`: `evaluate_loocv()` prints per-fold prediction table and saves JSON report
- Strict model registry under `aim1_non_conversion_strict/`; imputed under `aim1_non_conversion/`

---

## Question 7: Merging Demographic Data for Symptom Features

**Asked**: After the pipeline ran on imputed data, the API returned `500 Internal Server Error` on prediction requests. The error was `ValueError: Cannot use median strategy with non-numeric data: could not convert string to float: 'NO'`.

**Root cause**: The Aim 1 dataset had two issues:
1. **Missing symptom columns**: `build_aim1_dataset()` only loaded the "Merged" sheet from `MTIPLUS dataset.xlsx`, which lacks COUGH, FEVER, WEIGHT LOSS, NIGHT SWEATS, CHEST PAIN, and HEMOPTYSIS columns. These symptoms exist in the demographic PATIENTS sheet (`MTIPLUS DEMOGRAPHIC DATA (1) (1).xlsx`) and all 207 `Sample ID`s overlap.
2. **Column name mismatch**: Temperature column is `TEMPERATURE_CELCIUS` (underscore) in the data but was hardcoded as `TEMPERATURE CELCIUS` (space) in feature definitions.
3. **Wrong API type for TB_CONTACT**: `TB_CONTACT` stores a numeric count of contacts (1–12), not a yes/no string, but the API schema defined it as `Optional[str]`.

**Fix**: 
1. Merged demographic PATIENTS data on `Sample ID` to bring in symptom columns, with `_standardize_yes_no` applied
2. Fixed `clean_patients` to convert `TEMPERATURE_CELCIUS` to numeric
3. Updated `FEATURE_MAP_AIM1` and feature engineering `feature_cols` to match actual column names
4. Changed `TB_CONTACT` type from `str` to `float` in `Aim1PredictionRequest`

**Implementation impact**:
- `clean_data.py`: `build_aim1_dataset()` now merges `load_demographic_patients()` on `Sample ID`, adds `TEMPERATURE_CELCIUS` to numeric conversion list
- `feature_engineering.py`: Fixed `TEMPERATURE_CELCIUS` (underscore) and removed `DYSPENA` (not in data)
- `api/main.py`, `api/schemas.py`: Updated FEATURE_MAP, removed DYSPENA, fixed TB_CONTACT type
- Model now trains on 18 features with full symptom coverage; champion RF achieves CV AUC=0.651

---

## Question 8: Aim 2 Target Leakage — Symptom Presence Proxy

**Asked**: After testing Aim 2, the model gave `CV AUC=1.000` and `Test AUC=1.000` with perfect but meaningless predictions. All-NO input gave probability 0.358 instead of near-zero.

**Identified problem**: Aim 2's target (`TARGET_SYMPTOM_PRESENT`) is derived from the same symptom columns used as features (COUGH, FEVER, WEIGHT LOSS, etc.). The model trivially learns "if any symptom = YES, predict 1." This is circular — no predictive value for new data.

**Options given**:
1. **Accept as symptom cluster detector (proof-of-concept)** — document the limitation, keep the API working
2. Redefine as unsupervised anomaly detection (Isolation Forest)
3. Drop Aim 2, focus on Aim 1
4. Obtain real outcome data (longitudinal follow-up, TB registry linkage)

**Answer**: Option 1 — accept Aim 2 as a proof-of-concept demonstrating the API infrastructure works. The model detects which contacts self-report symptoms, which is not predictive of future TB risk.

**Rationale**: No longitudinal outcome data is available for the 46 contacts. The "HEALTHY CONTACTS" sheet is a cross-sectional survey with no follow-up. Without a real outcome (who developed TB), supervised learning cannot produce a clinically meaningful model. The endpoint remains available to demonstrate the API works end-to-end.

**Impact**:
- The Aim 2 endpoint is kept but its output should not be interpreted as TB risk
- Documented here and in the API response schema notes
- Future improvement requires linking contacts to TB registry outcomes or conducting follow-up visits

---

## Question 9: Strict Model Feature Selection and Label Fix

**Asked**: During the Aim 1 redesign to a two-phase strategy, the culture label mapping was found to misclassify `"MTBc Negative"` as non-conversion (failure) because the original `map_conversion` only checked for `"No Growth"`.

**Question**: "Which features should the strict model use, and how should the culture labels be mapped?"

**Answer**:

### Label mapping fix
The original `map_conversion` in `build_aim1_dataset()` only matched `"No Growth"`:
```python
return 0 if "No Growth" in str(val) else 1  # Bug: "MTBc Negative" → 1
```
This was fixed to also match `"Negative"` / `"MTBc Negative"`:
```python
s = str(val).strip().lower()
if s in ("no growth", "negative", "mtbc negative"): return 0
if "positive" in s: return 1
```
After fix: 6 non-conversions (was 7). Strict and imputed counts now agree.

### Feature selection (4 features only)
| Feature | Rationale |
|---|---|
| `AGE (YEARS)` | Well-established TB risk factor; ~96% complete |
| `BMI` | Strong predictor of delayed conversion; ~77% complete |
| `SEX` | Male sex associated with higher failure rates |
| `baseline_symptom_count` | Robust count proxy (0–7), preserves d.f. vs 7 individual symptom dummies |

### Excluded features and why
| Excluded | Reason |
|---|---|
| `HIV_STATUS` | Only ~8% HIV-positive — insufficient variance |
| `HAS_DIABETES` | Very low prevalence (~3%), high missingness |
| Smoking, alcohol | No consensus on independent effect on culture conversion |
| Socioeconomic proxies | Distal causes mediated by nutritional status (BMI) |
| Education, occupation | Multi-level categorical — would need 6+ dummies, impossible at n=11 |
| Temperature | Already captured in `baseline_symptom_count` (fever) |

**Rationale**: The one-in-ten rule for logistic regression suggests ~0.6 features (6 events ÷ 10). Using 4 features with L2 regularization violates the rule but is justified because the model is explicitly exploratory — it identifies candidate feature directions, not deployable predictions.

**Implementation impact**:
- `clean_data.py`: `map_conversion` and `map_strict` both handle all culture value variants
- `feature_engineering.py`: `get_aim1_strict_features_target()` returns only the 4 selected features; `build_preprocessor` treats SEX as binary, AGE/BMI/symptom_count as numeric
- The imputed model continues using all 18 features (different data regime, different constraints)
