import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
METADATA_DIR = MODELS_DIR / "metadata"


def predict(aim: str, model_name: str = "xgboost", version: str = None):
    if version is None:
        champion = _get_champion_model(aim)
        if champion is None:
            raise ValueError(f"No trained model found for aim={aim}")
        version = champion["version"]
        model_name = champion["model"]

    path = REGISTRY_DIR / aim / f"{version}_{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    pipeline = joblib.load(path)
    return pipeline


def predict_single(pipeline, features: dict, feature_cols: list) -> dict:
    df = pd.DataFrame([features])
    df = df[feature_cols] if all(c in df.columns for c in feature_cols) else df.reindex(columns=feature_cols)

    proba = pipeline.predict_proba(df)[0]
    pred = pipeline.predict(df)[0]

    return {
        "prediction": int(pred),
        "probability": float(proba[1]),
        "confidence": float(max(proba)),
    }


def predict_batch(pipeline, df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    df = df.reindex(columns=feature_cols)
    proba = pipeline.predict_proba(df)
    preds = pipeline.predict(df)

    result = df.copy()
    result["prediction"] = preds
    result["probability_positive"] = proba[:, 1]
    result["confidence"] = np.max(proba, axis=1)
    return result


def _get_champion_model(aim: str):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    best_score = -1.0
    for meta_path in METADATA_DIR.glob(f"{aim}_*.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        score = meta.get("cv_auc_mean")
        if score is not None and not np.isnan(score) and score > best_score:
            best_score = score
            best = meta
    return best


def get_model_info(aim: str, model_name: str = "xgboost", version: str = None):
    if version is None:
        champion = _get_champion_model(aim)
        if champion is None:
            return None
        version = champion["version"]
        model_name = champion["model"]

    meta_path = METADATA_DIR / f"{aim}_{model_name}_{version}.json"
    if not meta_path.exists():
        return None
    with open(meta_path) as f:
        return json.load(f)
