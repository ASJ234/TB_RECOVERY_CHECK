import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
CLEANED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cleaned"

PATIENT_FILE = RAW_DATA_DIR / "MTIPLUS dataset.xlsx"
DEMOGRAPHIC_FILE = RAW_DATA_DIR / "MTIPLUS DEMOGRAPHIC DATA (1) (1).xlsx"


def load_merged_patients() -> pd.DataFrame:
    df = pd.read_excel(PATIENT_FILE, sheet_name="Merged")
    df = df.iloc[1:].reset_index(drop=True)
    rename_map = {
        "Unnamed: 4": "BASELINE_RESULT",
        "Unnamed: 6": "TEMPERATURE_CELCIUS",
        "Unnamed: 24": "2_MONTHS_RESULT",
        "Unnamed: 26": "5_MONTHS_RESULT",
        "Unnamed: 35": "HIV_STATUS",
        "Unnamed: 36": "HAS_DIABETES",
        "Unnamed: 37": "HAS_CANCER",
        "Unnamed: 38": "SMOKES",
        "Unnamed: 39": "CONSUMES_ALCOHOL",
        "Unnamed: 40": "PAST_TB_DIAGNOSIS",
        "Unnamed: 41": "IF_YES_WHEN",
        "Unnamed: 42": "TREATED_FOR_TB",
        "Unnamed: 43": "TB_CONTACT",
        "Unnamed: 44": "NUMBER_OF_OCCUPANTS",
    }
    df = df.rename(columns=rename_map)
    df["SOURCE"] = "merged"
    return df


def load_demographic_patients() -> pd.DataFrame:
    df = pd.read_excel(DEMOGRAPHIC_FILE, sheet_name="PATIENTS")
    rename_map = {
        "Unnamed: 0": "INDEX",
        "Unnamed: 12": "HEIGHT_M",
        "DURATION.1": "WEIGHT_LOSS_DURATION",
        "DURATION.2": "NIGHT_SWEATS_DURATION",
        "DURATION.3": "DYSPNEA_DURATION",
        "DURATION.4": "CHEST_PAIN_DURATION",
        "DURATION.5": "HEMOPTYSIS_DURATION",
        "DURATION.6": "OTHERS_DURATION",
    }
    df = df.rename(columns=rename_map)
    df["SOURCE"] = "demographic"
    return df


def load_followup() -> pd.DataFrame:
    df = pd.read_excel(DEMOGRAPHIC_FILE, sheet_name="FOLLOWUP")
    rename_map = {
        "Sample ID ": "Sample ID",
        "DURATION.1": "WEIGHT_LOSS_DURATION",
        "DURATION.2": "NIGHT_SWEATS_DURATION",
        "DURATION.3": "DYSPNEA_DURATION",
        "DURATION.4": "CHEST_PAIN_DURATION",
        "DURATION.5": "HEMOPTYSIS_DURATION",
        "DURATION.6": "OTHERS_DURATION",
        "DURATION.7": "ARV_DURATION",
    }
    df = df.rename(columns=rename_map)
    return df


def load_healthy_contacts() -> pd.DataFrame:
    df = pd.read_excel(DEMOGRAPHIC_FILE, sheet_name="HEALTHY CONTACTS")
    rename_map = {
        "STUDY INDENTIFICATION NUMBER": "STUDY_ID",
        "Unnamed: 1": "CONTACT_NUMBER",
        "WEEKS": "COUGH_WEEKS",
        "WEEKS.1": "FEVER_WEEKS",
        "WEEKS.2": "WEIGHT_LOSS_WEEKS",
        "WEEKS.3": "NIGHT_SWEATS_WEEKS",
        "WEEKS.4": "DYSPNEA_WEEKS",
        "WEEKS.5": "CHEST_PAIN_WEEKS",
        "WEEKS.6": "HEMOPTYSIS_WEEKS",
        "OTHER": "OTHER_SYMPTOMS",
    }
    df = df.rename(columns=rename_map)
    return df


def load_merged_dataset_sheet() -> pd.DataFrame:
    df = pd.read_excel(DEMOGRAPHIC_FILE, sheet_name="Merged Dataset")
    df = df.iloc[1:].reset_index(drop=True)
    rename_map = {
        "Unnamed: 4": "BASELINE_RESULT",
        "Unnamed: 6": "TEMPERATURE_CELCIUS",
        "Unnamed: 24": "2_MONTHS_RESULT",
        "Unnamed: 26": "5_MONTHS_RESULT",
        "Unnamed: 35": "HIV_STATUS",
        "Unnamed: 36": "HAS_DIABETES",
        "Unnamed: 37": "HAS_CANCER",
        "Unnamed: 38": "SMOKES",
        "Unnamed: 39": "CONSUMES_ALCOHOL",
        "Unnamed: 40": "PAST_TB_DIAGNOSIS",
        "Unnamed: 41": "IF_YES_WHEN",
        "Unnamed: 42": "TREATED_FOR_TB",
        "Unnamed: 43": "TB_CONTACT",
        "Unnamed: 44": "NUMBER_OF_OCCUPANTS",
    }
    df = df.rename(columns=rename_map)
    df["SOURCE"] = "merged_demographic"
    return df


def get_cleaned_path(filename: str) -> Path:
    CLEANED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    return CLEANED_DATA_DIR / filename
