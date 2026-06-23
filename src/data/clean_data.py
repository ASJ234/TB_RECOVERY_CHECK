import pandas as pd
import numpy as np
from src.data.load_data import (
    load_merged_patients,
    load_demographic_patients,
    load_followup,
    load_healthy_contacts,
    get_processsed_path,
)


def _standardize_yes_no(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("YES", "Y", "TRUE", "POSITIVE"):
        return "YES"
    if s in ("NO", "N", "FALSE", "NEGATIVE"):
        return "NO"
    return val


def _standardize_sex(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("M", "MALE"):
        return "M"
    if s in ("F", "FEMALE"):
        return "F"
    return val


def _clean_followup_month(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower().replace(" ", "")
    if "month" in s or "momth" in s or "montn" in s or "motnh" in s:
        num = "".join(c for c in s if c.isdigit())
        return f"MONTH_{num}" if num else val
    return val


def clean_patients(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    bool_cols = [
        "COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
        "DYSPENA", "CHEST PAIN", "HEMOPTYSIS",
    ]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(_standardize_yes_no)

    yes_no_cols = [
        "HAS_DIABETES", "HAS_CANCER", "SMOKES", "CONSUMES_ALCOHOL",
        "PAST_TB_DIAGNOSIS", "TB_CONTACT", "HAS DIABETES (YES/NO)",
        "HAS CANCER (YES/NO)", "SMOKES CIGARETTES (YES/NO)",
        "CONSUMES ALCOHOL (YES/NO)",
        "EVER BEEN DIAGNOSED WITH TB IN THE PAST? (YES/NO)",
        "CONSUMED ANTIBIOTIC(S) IN THE PAST 6 MONTHS  (YES/NO)",
    ]
    for c in yes_no_cols:
        if c in df.columns:
            df[c] = df[c].apply(_standardize_yes_no)

    if "SEX" in df.columns:
        df["SEX"] = df["SEX"].apply(_standardize_sex)

    if "HIV STATUS" in df.columns:
        df["HIV STATUS"] = df["HIV STATUS"].apply(_standardize_yes_no)

    if "HIV_STATUS" in df.columns:
        df["HIV_STATUS"] = df["HIV_STATUS"].apply(_standardize_yes_no)

    for age_col in ["AGE (YEARS)", "AGE"]:
        if age_col in df.columns:
            df[age_col] = pd.to_numeric(df[age_col], errors="coerce")

    for weight_col in ["WEIGHT (KG)", "WEIGHT", "WEIGHT IN KGS"]:
        if weight_col in df.columns:
            df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce")

    for height_col in ["HEIGHT (M)", "HEIGHT (CM)", "HEIGHT"]:
        if height_col in df.columns:
            df[height_col] = pd.to_numeric(df[height_col], errors="coerce")

    for temp_col in ["TEMPERATURE CELICIUS", "TEMP IN CENTIGRADE", "TEMPERATURE"]:
        if temp_col in df.columns:
            df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")

    if "BMI" in df.columns:
        df["BMI"] = pd.to_numeric(df["BMI"], errors="coerce")

    return df


def clean_followup(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "FOLLOW UP MONTH" in df.columns:
        df["FOLLOW UP MONTH"] = df["FOLLOW UP MONTH"].apply(_clean_followup_month)

    bool_cols = ["COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
                  "DYSPNEA", "CHEST PAIN", "HEMOPTYSIS"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(_standardize_yes_no)

    numeric_cols = ["AGE", "WEIGHT IN KGS", "TEMP IN CENTIGRADE"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "HIV STATUS" in df.columns:
        df["HIV STATUS"] = df["HIV STATUS"].apply(_standardize_yes_no)

    return df


def clean_healthy_contacts(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["SEX"] = df["SEX"].apply(_standardize_sex)

    bool_cols = ["COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
                  "DYSPNEA", "CHEST PAIN", "HEMOPTYSIS"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].apply(_standardize_yes_no)

    for c in ["AGE", "WEIGHT", "TEMPERATURE"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "HIV STATUS" in df.columns:
        df["HIV STATUS"] = df["HIV STATUS"].apply(_standardize_yes_no)

    return df


def build_aim1_dataset() -> pd.DataFrame:
    df = load_merged_patients()
    df = clean_patients(df)

    def map_conversion(val):
        if pd.isna(val):
            return pd.NA
        return 0 if "No Growth" in str(val) else 1

    df["TARGET_NON_CONVERSION_2M"] = df["2 MONTHS"].map(map_conversion)
    df["TARGET_NON_CONVERSION_5M"] = df["5 MONTHS"].map(map_conversion)
    df["TARGET_NON_CONVERSION_ANY"] = df[
        ["TARGET_NON_CONVERSION_2M", "TARGET_NON_CONVERSION_5M"]
    ].max(axis=1).where(df[["TARGET_NON_CONVERSION_2M", "TARGET_NON_CONVERSION_5M"]].notna().any(axis=1), pd.NA)

    baseline_col = "BASELINE_RESULT" if "BASELINE_RESULT" in df.columns else "Unnamed: 4"
    df["BASELINE_POSITIVE"] = df[baseline_col].map(
        lambda x: 1 if pd.notna(x) and "Positive" in str(x) else (0 if pd.notna(x) else pd.NA)
    )

    return df


def build_aim2_dataset() -> pd.DataFrame:
    df = load_healthy_contacts()
    df = clean_healthy_contacts(df)

    symptom_cols = ["COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
                     "DYSPNEA", "CHEST PAIN", "HEMOPTYSIS"]
    for c in symptom_cols:
        if c in df.columns:
            df[c + "_BINARY"] = df[c].map(lambda x: 1 if x == "YES" else 0)

    binary_cols = [c + "_BINARY" for c in symptom_cols if c + "_BINARY" in df.columns]
    df["TARGET_SYMPTOM_PRESENT"] = df[binary_cols].sum(axis=1).clip(0, 1) if binary_cols else 0

    return df


def save_clean_datasets():
    aim1 = build_aim1_dataset()
    aim2 = build_aim2_dataset()

    aim1.to_csv(get_processsed_path("aim1_patients.csv"), index=False)
    aim2.to_csv(get_processsed_path("aim2_contacts.csv"), index=False)

    fu = clean_followup(load_followup())
    fu.to_csv(get_processsed_path("followup.csv"), index=False)

    print(f"Aim 1 dataset: {len(aim1)} rows, {aim1['TARGET_NON_CONVERSION_ANY'].notna().sum()} labeled")
    print(f"Aim 2 dataset: {len(aim2)} rows, {aim2['TARGET_SYMPTOM_PRESENT'].sum()} symptomatic")
    print(f"Follow-up data: {len(fu)} rows")


if __name__ == "__main__":
    save_clean_datasets()
