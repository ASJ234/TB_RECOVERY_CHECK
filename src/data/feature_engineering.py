import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SCALERS_DIR = MODELS_DIR / "scalers"


def _derive_bmi(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    weight_col = None
    height_col = None
    for c in ["WEIGHT (KG)", "WEIGHT", "WEIGHT IN KGS"]:
        if c in df.columns:
            weight_col = c
            break
    for c in ["HEIGHT (M)", "HEIGHT (CM)", "HEIGHT"]:
        if c in df.columns:
            height_col = c
            break

    if weight_col and height_col:
        height = df[height_col]
        if height_col == "HEIGHT (CM)":
            height = height / 100.0
        bmi = pd.to_numeric(df[weight_col], errors="coerce") / (
            pd.to_numeric(height, errors="coerce") ** 2
        )
        df["BMI_CALCULATED"] = np.where(np.isfinite(bmi), bmi, np.nan)

    return df


def get_aim1_features_target(df: pd.DataFrame):
    df = _derive_bmi(df)

    feature_cols = [
        "SEX", "AGE (YEARS)", "TEMPERATURE CELCIUS",
        "COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
        "DYSPENA", "CHEST PAIN", "HEMOPTYSIS",
        "HIV_STATUS", "HAS_DIABETES", "SMOKES", "CONSUMES_ALCOHOL",
        "PAST_TB_DIAGNOSIS", "TB_CONTACT",
        "NUMBER_OF_OCCUPANTS", "BMI", "BASELINE_POSITIVE",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_model = df[feature_cols + ["TARGET_NON_CONVERSION_ANY"]].copy()
    df_model = df_model[df_model["TARGET_NON_CONVERSION_ANY"].notna()].copy()

    return df_model, feature_cols


def get_aim2_features_target(df: pd.DataFrame):
    df = _derive_bmi(df)

    feature_cols = [
        "AGE", "SEX", "WEIGHT", "TEMPERATURE",
        "COUGH", "FEVER", "WEIGHT LOSS", "NIGHT SWEATS",
        "DYSPNEA", "CHEST PAIN", "HEMOPTYSIS",
        "HIV STATUS",
    ]
    feature_cols = [c for c in feature_cols if c in df.columns]

    df_model = df[feature_cols + ["TARGET_SYMPTOM_PRESENT"]].copy()

    return df_model, feature_cols


def _categorize_features(df: pd.DataFrame, feature_cols: list):
    numeric_cols = []
    categorical_cols = []
    binary_cols = []

    for c in feature_cols:
        if c not in df.columns:
            continue
        if df[c].dtype in (np.float64, np.int64, float, int):
            numeric_cols.append(c)
        elif df[c].dropna().nunique() <= 2:
            binary_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols, binary_cols


def build_preprocessor(df: pd.DataFrame, feature_cols: list):
    numeric_cols, categorical_cols, binary_cols = _categorize_features(df, feature_cols)

    transformers = []

    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))

    if categorical_cols:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            categorical_cols,
        ))

    if binary_cols:
        transformers.append((
            "bin",
            Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="NO")),
                ("encoder", OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False)),
            ]),
            binary_cols,
        ))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )
    return preprocessor


def prepare_aim1_data(test_size=0.2, random_state=42):
    df = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data" / "processed" / "aim1_patients.csv"
    )
    df_model, feature_cols = get_aim1_features_target(df)

    if len(df_model) < 2:
        return None, None, None, None, None, None, feature_cols

    X = df_model[feature_cols]
    y = df_model["TARGET_NON_CONVERSION_ANY"].astype(int)

    if test_size * len(X) >= 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X.iloc[:0], y, y.iloc[:0]

    preprocessor = build_preprocessor(X_train, feature_cols)
    return X_train, X_test, y_train, y_test, preprocessor, feature_cols


def prepare_aim2_data(test_size=0.2, random_state=42):
    df = pd.read_csv(
        Path(__file__).resolve().parents[2] / "data" / "processed" / "aim2_contacts.csv"
    )
    df_model, feature_cols = get_aim2_features_target(df)

    X = df_model[feature_cols]
    y = df_model["TARGET_SYMPTOM_PRESENT"].astype(int)

    if test_size * len(X) >= 1:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = X, X.iloc[:0], y, y.iloc[:0]

    preprocessor = build_preprocessor(X_train, feature_cols)
    return X_train, X_test, y_train, y_test, preprocessor, feature_cols


def save_scaler(preprocessor, aim: str, version: str):
    SCALERS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCALERS_DIR / f"{aim}_preprocessor_{version}.pkl"
    joblib.dump(preprocessor, path)
    return path


def load_scaler(aim: str, version: str):
    path = SCALERS_DIR / f"{aim}_preprocessor_{version}.pkl"
    return joblib.load(path)
