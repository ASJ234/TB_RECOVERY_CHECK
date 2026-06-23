import pytest
import pandas as pd
from pathlib import Path
from src.data.load_data import (
    load_merged_patients,
    load_demographic_patients,
    load_followup,
    load_healthy_contacts,
)
from src.data.clean_data import (
    clean_patients,
    clean_followup,
    clean_healthy_contacts,
    build_aim1_dataset,
    build_aim2_dataset,
)


class TestLoadData:
    def test_load_merged_patients(self):
        df = load_merged_patients()
        assert len(df) > 0
        assert "Sample ID" in df.columns
        assert "SEX" in df.columns

    def test_load_demographic_patients(self):
        df = load_demographic_patients()
        assert len(df) > 0
        assert "Sample ID" in df.columns

    def test_load_followup(self):
        df = load_followup()
        assert len(df) > 0
        assert "FOLLOW UP MONTH" in df.columns

    def test_load_healthy_contacts(self):
        df = load_healthy_contacts()
        assert len(df) > 0
        assert "STUDY_ID" in df.columns


class TestCleanData:
    def test_clean_patients_standardizes_sex(self):
        import numpy as np
        df = pd.DataFrame({"SEX": ["M", "F", "Male", "Female", np.nan]})
        cleaned = clean_patients(df)
        assert cleaned["SEX"].iloc[0] == "M"
        assert cleaned["SEX"].iloc[1] == "F"
        assert cleaned["SEX"].iloc[2] == "M"
        assert cleaned["SEX"].iloc[3] == "F"
        assert pd.isna(cleaned["SEX"].iloc[4])

    def test_clean_patients_standardizes_yes_no(self):
        import numpy as np
        df = pd.DataFrame({"COUGH": ["YES", "NO", "Yes", "no", np.nan]})
        cleaned = clean_patients(df)
        assert cleaned["COUGH"].iloc[0] == "YES"
        assert cleaned["COUGH"].iloc[1] == "NO"
        assert cleaned["COUGH"].iloc[2] == "YES"
        assert cleaned["COUGH"].iloc[3] == "NO"

    def test_clean_followup_standardizes_month(self):
        df = pd.DataFrame({"FOLLOW UP MONTH": ["MONTH 2", "MOMTH 2", "MONTH5", None]})
        cleaned = clean_followup(df)
        assert cleaned["FOLLOW UP MONTH"].iloc[0] == "MONTH_2"
        assert cleaned["FOLLOW UP MONTH"].iloc[1] == "MONTH_2"
        assert cleaned["FOLLOW UP MONTH"].iloc[2] == "MONTH_5"

    def test_build_aim1_dataset(self):
        df = build_aim1_dataset()
        assert "TARGET_NON_CONVERSION_ANY" in df.columns
        assert "TARGET_NON_CONVERSION_2M" in df.columns
        assert "TARGET_NON_CONVERSION_5M" in df.columns

    def test_build_aim2_dataset(self):
        df = build_aim2_dataset()
        assert "TARGET_SYMPTOM_PRESENT" in df.columns


class TestFeatureEngineering:
    def test_prepare_aim1_data(self):
        from src.data.clean_data import save_clean_datasets
        save_clean_datasets()
        from src.data.feature_engineering import prepare_aim1_data
        result = prepare_aim1_data(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test, preprocessor, feature_cols = result
        if X_train is not None and len(X_train) > 0:
            assert len(feature_cols) > 0
            assert preprocessor is not None

    def test_prepare_aim2_data(self):
        from src.data.clean_data import save_clean_datasets
        save_clean_datasets()
        from src.data.feature_engineering import prepare_aim2_data
        result = prepare_aim2_data(test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test, preprocessor, feature_cols = result
        if X_train is not None and len(X_train) > 0:
            assert len(feature_cols) > 0
            assert preprocessor is not None
