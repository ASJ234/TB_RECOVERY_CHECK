from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class DriftScenario:
    name: str
    description: str
    target_aim: str
    drift_curves: Dict[str, str]
    prompt_hints: str
    tags: List[str] = field(default_factory=list)


SCENARIOS: Dict[str, DriftScenario] = {
    "gradual_age_shift": DriftScenario(
        name="gradual_age_shift",
        description="Mean patient age drifts from 35 to 55 over 72 hours, with downstream effects on comorbidities",
        target_aim="aim1",
        drift_curves={
            "AGE (YEARS)": "linear_up",
            "BMI": "linear_down",
            "TEMPERATURE_CELCIUS": "none",
            "TB_CONTACT": "none",
            "NUMBER_OF_OCCUPANTS": "none",
            "BASELINE_POSITIVE": "none",
            "TARGET_NON_CONVERSION_ANY": "linear_up",
        },
        prompt_hints="Gradually increase mean age. Older patients have slightly lower BMI and higher non-conversion risk.",
        tags=["gradual", "demographic"],
    ),
    "sudden_outbreak": DriftScenario(
        name="sudden_outbreak",
        description="At hour 48, a sudden spike in hemoptysis and fever cases simulates a localized outbreak",
        target_aim="aim1",
        drift_curves={
            "HEMOPTYSIS": "step_up_mid",
            "FEVER": "step_up_mid",
            "COUGH": "step_up_mid",
            "TEMPERATURE_CELCIUS": "step_up_mid",
            "TARGET_NON_CONVERSION_ANY": "step_up_mid",
        },
        prompt_hints="Before hour 48, keep distributions near baseline. Starting at hour 48, sharply increase hemoptysis, fever, and cough rates. Temperature should rise slightly.",
        tags=["sudden", "clinical"],
    ),
    "seasonal_bmi": DriftScenario(
        name="seasonal_bmi",
        description="BMI oscillates sinusoidally simulating seasonal nutritional variation",
        target_aim="aim1",
        drift_curves={
            "BMI": "sine",
            "AGE (YEARS)": "none",
            "TEMPERATURE_CELCIUS": "none",
            "TARGET_NON_CONVERSION_ANY": "sine",
        },
        prompt_hints="BMI follows a sine wave with 72-hour period. Non-conversion rate inversely correlates with BMI.",
        tags=["cyclical", "nutritional"],
    ),
    "cohort_composition": DriftScenario(
        name="cohort_composition",
        description="Gradual influx of HIV-positive patients from 5% to 25% over 72 hours",
        target_aim="aim1",
        drift_curves={
            "HIV_STATUS": "linear_up",
            "BMI": "linear_down",
            "WEIGHT LOSS": "linear_up",
            "TARGET_NON_CONVERSION_ANY": "linear_up",
        },
        prompt_hints="Gradually increase HIV-positive rate from 5% to 25%. HIV+ patients tend to have lower BMI, more weight loss, and higher non-conversion.",
        tags=["gradual", "comorbidity"],
    ),
    "measurement_bias": DriftScenario(
        name="measurement_bias",
        description="Temperature readings systematically shift +0.5C after hour 36 due to instrument recalibration",
        target_aim="aim1",
        drift_curves={
            "TEMPERATURE_CELCIUS": "step_up_mid",
            "AGE (YEARS)": "none",
            "BMI": "none",
            "COUGH": "none",
            "FEVER": "none",
            "TARGET_NON_CONVERSION_ANY": "none",
        },
        prompt_hints="Only temperature drifts. Starting at hour 36, increase mean temperature by 0.5C. Other features stay at baseline.",
        tags=["instrument", "subtle"],
    ),
    "adversarial": DriftScenario(
        name="adversarial",
        description="Non-conversion rate flips sharply at hour 60, causing model AUC to collapse — tests model drift detection",
        target_aim="aim1",
        drift_curves={
            "TARGET_NON_CONVERSION_ANY": "step_up_mid",
            "AGE (YEARS)": "none",
            "BMI": "none",
            "FEVER": "none",
            "COUGH": "none",
            "HEMOPTYSIS": "none",
        },
        prompt_hints="Keep all feature distributions at baseline for the first 60 hours. At hour 60, sharply increase non-conversion rate from 3% to 30%. The prediction surface should shift dramatically.",
        tags=["adversarial", "model_drift"],
    ),
    "mixed": DriftScenario(
        name="mixed",
        description="Combines gradual age shift + cohort composition + seasonal BMI for a realistic multi-factor drift scenario",
        target_aim="aim1",
        drift_curves={
            "AGE (YEARS)": "linear_up",
            "BMI": "sine",
            "HIV_STATUS": "linear_up",
            "TEMPERATURE_CELCIUS": "none",
            "TARGET_NON_CONVERSION_ANY": "linear_up",
        },
        prompt_hints="Combine multiple drift signals: age gradually rises, HIV rate increases, BMI oscillates seasonally. Non-conversion rate drifts up slowly.",
        tags=["mixed", "realistic"],
    ),
}


def list_scenarios() -> List[str]:
    return list(SCENARIOS.keys())


def get_scenario(name: str) -> Optional[DriftScenario]:
    return SCENARIOS.get(name)
