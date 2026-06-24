import pandas as pd
import numpy as np
from src.data.load_data import (
    load_merged_patients,
    load_demographic_patients,
    load_followup,
    load_healthy_contacts,
    get_cleaned_path,
)


def _standardize_yes_no(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("YES", "Y", "TRUE", "POSITIVE"):
        return "YES"
    if s in ("NO", "N", "FALSE", "NEGATIVE"):
        return "NO"
    if s in ("#", "", "?", "UNKNOWN", "N/A", "NA"):
        return np.nan
    return val


def _standardize_sex(val):
    if pd.isna(val):
        return np.nan
    s = str(val).strip().upper()
    if s in ("M", "MALE"):
        return "M"
    if s in ("F", "FEMALE"):
        return "F"
    if s in ("#", "", "?", "UNKNOWN", "N/A", "NA"):
        return np.nan
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

    for temp_col in ["TEMPERATURE_CELCIUS", "TEMPERATURE CELICIUS", "TEMP IN CENTIGRADE", "TEMPERATURE"]:
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

    demo = load_demographic_patients()
    symptom_cols = ["COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
                     "CHEST PAIN", "HEMOPTYSIS"]
    for c in symptom_cols:
        if c in demo.columns:
            demo[c] = demo[c].apply(_standardize_yes_no)
    merge_cols = ["Sample ID"] + [c for c in symptom_cols if c in demo.columns]
    df = df.merge(demo[merge_cols], on="Sample ID", how="left")

    def map_conversion(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip().lower()
        if s in ("no growth", "negative", "mtbc negative"):
            return 0
        if "positive" in s:
            return 1
        return pd.NA

    df["TARGET_NON_CONVERSION_2M"] = df["2 MONTHS"].map(map_conversion)
    df["TARGET_NON_CONVERSION_5M"] = df["5 MONTHS"].map(map_conversion)

    had_followup = df[["TARGET_NON_CONVERSION_2M", "TARGET_NON_CONVERSION_5M"]].notna().any(axis=1)
    df["TARGET_NON_CONVERSION_ANY"] = df[
        ["TARGET_NON_CONVERSION_2M", "TARGET_NON_CONVERSION_5M"]
    ].max(axis=1).fillna(0).astype(int)
    df["IS_IMPUTED"] = (~had_followup).astype(int)

    baseline_col = "BASELINE_RESULT" if "BASELINE_RESULT" in df.columns else "Unnamed: 4"
    df["BASELINE_POSITIVE"] = df[baseline_col].map(
        lambda x: 1 if pd.notna(x) and "Positive" in str(x) else (0 if pd.notna(x) else pd.NA)
    )

    return df


def build_aim1_strict_dataset() -> pd.DataFrame:
    df = load_merged_patients()
    df = clean_patients(df)

    demo = load_demographic_patients()
    symptom_cols = ["COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
                     "CHEST PAIN", "HEMOPTYSIS"]
    for c in symptom_cols:
        if c in demo.columns:
            demo[c] = demo[c].apply(_standardize_yes_no)
    merge_cols = ["Sample ID"] + [c for c in symptom_cols if c in demo.columns]
    df = df.merge(demo[merge_cols], on="Sample ID", how="left")

    culture_col = "2 MONTHS"
    conversion_col = "2_MONTHS_RESULT"

    def map_strict(val):
        if pd.isna(val):
            return pd.NA
        s = str(val).strip().lower()
        if s in ("no growth", "negative", "mtbc negative"):
            return 0
        if "positive" in s:
            return 1
        return pd.NA

    df["TARGET_STRICT_M2"] = df[culture_col].map(map_strict)

    df["baseline_symptom_count"] = 0
    for c in symptom_cols:
        if c in df.columns:
            df["baseline_symptom_count"] += df[c].map(
                lambda x: 1 if str(x).strip().upper() == "YES" else 0
            ).fillna(0).astype(int)

    has_culture = df[culture_col].notna()
    has_conversion = df[conversion_col].notna()
    df["M2_DISCORDANT"] = 0
    both = has_culture & has_conversion
    for idx in df[both].index:
        culture_val = str(df.loc[idx, culture_col]).strip().lower()
        conv_val = str(df.loc[idx, conversion_col]).strip().lower()
        culture_failure = "positive" in culture_val or culture_val == "mtbc positive"
        conv_failure = conv_val == "no conversion"
        if culture_failure != conv_failure:
            df.loc[idx, "M2_DISCORDANT"] = 1

    df["IS_IMPUTED"] = (~has_culture).astype(int)

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


TARGET_COLS = {
    "TARGET_NON_CONVERSION_ANY", "TARGET_NON_CONVERSION_2M", "TARGET_NON_CONVERSION_5M",
    "TARGET_SYMPTOM_PRESENT", "IS_IMPUTED",
    "TARGET_STRICT_M2", "M2_DISCORDANT",
}

DATE_COLS = {"BASELINE", "DATE"}


def _impute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in TARGET_COLS or col in DATE_COLS:
            continue
        if df[col].dtype in (float, int) or df[col].dtype.kind in ("i", "f"):
            median_val = df[col].median(skipna=True)
            if pd.notna(median_val):
                df[col] = df[col].fillna(median_val)
            else:
                df[col] = df[col].fillna(0)
        else:
            most_freq = df[col].mode(dropna=True)
            if len(most_freq) > 0:
                df[col] = df[col].fillna(most_freq.iloc[0])
            else:
                df[col] = df[col].fillna("NO")
    return df


def save_clean_datasets():
    aim1 = build_aim1_dataset()
    aim2 = build_aim2_dataset()

    aim1.to_csv(get_cleaned_path("aim1_patients.csv"), index=False)
    aim2.to_csv(get_cleaned_path("aim2_contacts.csv"), index=False)

    aim1_strict = build_aim1_strict_dataset()
    n_strict = aim1_strict["TARGET_STRICT_M2"].notna().sum()
    n_discordant = aim1_strict["M2_DISCORDANT"].sum() if "M2_DISCORDANT" in aim1_strict.columns else 0
    aim1_strict.to_csv(get_cleaned_path("aim1_patients_strict.csv"), index=False)

    fu = clean_followup(load_followup())
    fu.to_csv(get_cleaned_path("followup.csv"), index=False)

    aim1_imp = _impute_features(aim1)
    aim2_imp = _impute_features(aim2)
    aim1_imp.to_csv(get_cleaned_path("aim1_patients_imputed.csv"), index=False)
    aim2_imp.to_csv(get_cleaned_path("aim2_contacts_imputed.csv"), index=False)

    n_imputed = aim1["IS_IMPUTED"].sum() if "IS_IMPUTED" in aim1.columns else 0
    print(f"Aim 1 (imputed) dataset: {len(aim1)} rows, {aim1['TARGET_NON_CONVERSION_ANY'].notna().sum()} labeled ({n_imputed} imputed as negative)")
    print(f"Aim 1 (strict) dataset: {n_strict} strictly labeled ({n_discordant} discordant)")
    print(f"Aim 2 dataset: {len(aim2)} rows, {aim2['TARGET_SYMPTOM_PRESENT'].sum()} symptomatic")
    print(f"Follow-up data: {len(fu)} rows")


if __name__ == "__main__":
    save_clean_datasets()
