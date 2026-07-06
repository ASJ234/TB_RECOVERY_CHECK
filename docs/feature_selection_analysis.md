# Feature Selection Analysis

## 1. Factors Considered When Choosing Features

### 1.1 Strict Model (Aim 1, n=11, 4 features)

#### Selected Features

| Feature | Rationale |
|---|---|
| `AGE (YEARS)` | Well-established TB risk factor; ~96% complete |
| `BMI` | Strong predictor of delayed conversion; ~77% complete |
| `SEX` | Male sex associated with higher failure rates |
| `baseline_symptom_count` | Robust count proxy (0–7), preserves d.f. vs 7 individual symptom dummies |

#### Selection Criteria

1. **One-in-ten rule** — With only 6 non-conversion events, logistic regression's rule of thumb (1 predictor per 10 events) allows ~0.6 features. Using 4 is a deliberate violation, explicitly flagged as exploratory only.

2. **Clinical evidence** — All 4 features are well-established TB risk factors in the literature.

3. **Data completeness** — AGE (96%) and BMI (77%) had the highest completeness among candidate features.

4. **Degrees of freedom preservation** — `baseline_symptom_count` was chosen over 7 individual symptom dummies to avoid consuming 7 d.f. when only 6 positive events exist.

#### Excluded Features and Reasons

| Excluded Feature | Reason for Exclusion |
|---|---|
| `HIV_STATUS` | Only ~8% HIV-positive — insufficient variance for meaningful signal |
| `HAS_DIABETES` | Very low prevalence (~3%), high missingness |
| `SMOKES` / `CONSUMES_ALCOHOL` | No consensus in literature on independent effect on culture conversion |
| Socioeconomic proxies | Distal causes mediated by nutritional status (BMI) |
| `TEMPERATURE_CELCIUS` | Already captured in `baseline_symptom_count` (fever) |
| Education / Occupation | Multi-level categorical — would need 6+ dummies, impossible at n=11 |
| `PAST_TB_DIAGNOSIS` / `TB_CONTACT` | High missingness, unclear direct pathway to non-conversion |

### 1.2 Imputed Model (Aim 1, n=218, 18 features)

#### Selected Features

```
SEX, AGE (YEARS), TEMPERATURE_CELCIUS, COUGH, FEVER, WEIGHT LOSS,
NIGHT SWEATS, CHEST PAIN, HEMOPTYSIS, HIV_STATUS, HAS_DIABETES,
SMOKES, CONSUMES_ALCOHOL, PAST_TB_DIAGNOSIS, TB_CONTACT,
NUMBER_OF_OCCUPANTS, BMI, BASELINE_POSITIVE
```

#### Selection Criteria

1. **Data availability** — All columns with sufficient completeness were included (no aggressive pre-selection), since the imputed model is a sensitivity analysis.

2. **Sample size support** — With 218 samples and ~11 positive events, the one-in-ten rule is satisfied (110 samples needed for 18 features).

3. **Coverage** — Features span demographics, symptoms, comorbidities, and socioeconomic proxies to capture all potential signal.

4. **Column name alignment** — Column names were adjusted to match actual Excel data (e.g., `TEMPERATURE_CELCIUS` not `TEMPERATURE CELCIUS`) discovered during debugging.

### 1.3 Aim 2 — Contact Risk (n=46, 12 features)

#### Selected Features

```
AGE, SEX, WEIGHT, TEMPERATURE, COUGH, FEVER, WEIGHT LOSS,
NIGHT SWEATS, DYSPNEA, CHEST PAIN, HEMOPTYSIS, HIV STATUS
```

#### Selection Criteria

1. **All available symptom columns** — used as features, but this is a known target leakage issue (target is derived from the same symptom columns).

2. **Accepted as a proof-of-concept** — no real outcome data exists for contacts (no longitudinal follow-up).

3. **Cross-sectional constraint** — The "HEALTHY CONTACTS" sheet is a one-time survey with no follow-up, limiting what can be predicted.

---

## 2. Impact of Feature Selection on Model Performance

### 2.1 Strict Model (4 features, n=11, LR + LOOCV)

```
LOOCV AUC:   0.133  (near-random)
Accuracy:    0.455  (5/11 correct)
Precision:   0.500
Recall:      0.667
F1:          0.571
```

**Impact of using only 4 features**: The extreme dimensionality reduction (4 features for 6 events) was necessary to make training possible at all. The model still has near-random AUC because n=11 is inherently insufficient — metrics flip with single-sample changes. The 4-feature choice lets the model at least converge and produce coefficient directions for hypothesis generation, but performance is too poor for any clinical use.

**Impact of excluded features**: Adding features like `HIV_STATUS` (8% prevalence), `HAS_DIABETES` (3%), or education/occupation (multi-level dummies) would make the model impossible to fit — each dummy variable consumes a degree of freedom when there are only 6 positive events. Excluding them was the only viable option.

### 2.2 Imputed Model (18 features, n=218)

| Model | CV AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | **0.158** | 0.748 | 0.000 | 0.000 |
| Random Forest (champion) | **0.554** | 0.881 | 0.045 | 0.167 |
| XGBoost | **0.294** | 0.940 | 0.000 | 0.000 |

**Impact of using 18 features**: The wide feature set lets flexible models (RF, XGBoost) achieve high train AUC (0.99–1.0) but they severely overfit — CV AUCs drop to 0.55 and 0.29. The gap between train and CV performance reveals that the 18 features plus imputed labels (97% majority class) primarily learn to predict the dominant class. Logistic Regression's precision=0.0 shows it never predicts the positive class — it defaults to always predicting the majority (imputed converted).

**Impact of included symptom features**: 7 symptom columns + `baseline_symptom_count` are highly redundant. Including both individual symptom dummies AND the count creates multicollinearity that hurts LR but gives RF/XGBoost more splitting opportunities — which is why RF has the best CV AUC (0.554).

**Impact of excluded strict features**: `HIV_STATUS`, `HAS_DIABETES`, smoking, alcohol were excluded from the strict model but ARE included here (218 samples can support them). They add marginal signal — LR's AUC barely moves (0.158) regardless of feature count.

### 2.3 Aim 2 (12 features, n=36)

| Model | CV AUC | Accuracy | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 1.000 | 0.944 | 0.800 | 1.000 |
| Random Forest | 1.000 | 1.000 | 1.000 | 1.000 |
| XGBoost | 1.000 | 1.000 | 1.000 | 1.000 |

**Impact of included features**: Perfect metrics are **not** a sign of good feature selection — they're evidence of **target leakage**. The 7 symptom columns used as features (`COUGH`, `FEVER`, etc.) are also used to construct the target (`TARGET_SYMPTOM_PRESENT`). The model trivially learns "if ANY symptom = YES, predict 1." This makes all 12 features equally useless for actual TB risk prediction.

**Impact of excluded features**: No real outcome data (longitudinal follow-up) exists for the 46 contacts. Even if more features existed, the fundamental issue is the circular target definition, not feature selection.

---

## 3. Summary

The feature selection impacts are dominated by the **label sparsity problem**, not the feature set itself:

| Model | Key Issue | Root Cause |
|---|---|---|
| **Strict (4 features)** | Near-random AUC (0.133) | Only 11 labeled samples for 6 events |
| **Imputed (18 features)** | High train / low CV AUC gap | 97% imputed majority class, label noise |
| **Aim 2 (12 features)** | Perfect but meaningless (AUC=1.0) | Target leakage — features equal target |
