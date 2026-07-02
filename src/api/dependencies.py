import joblib
import json
from pathlib import Path
from typing import Optional

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
METADATA_DIR = MODELS_DIR / "metadata"

_model_cache = {}


def _get_champion_meta(aim: str) -> Optional[dict]:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    best_score = -1.0
    for meta_path in METADATA_DIR.glob(f"{aim}_*.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        # Skip files that belong to a different aim (e.g. aim1_non_conversion_strict
        # files matching the aim1_non_conversion glob).
        if meta.get("aim") != aim:
            continue
        score = meta.get("cv_auc_mean")
        # Fall back to train_auc when cv_auc_mean is NaN or missing.
        if score is None or (isinstance(score, float) and score != score):
            score = meta.get("train_auc")
        if score is not None and not (
            isinstance(score, float) and score != score
        ) and score > best_score:
            best_score = score
            best = meta
    return best


def get_model(aim: str, model_name: str = "xgboost", version: Optional[str] = None):
    cache_key = f"{aim}_{model_name}_{version or 'champion'}"
    if cache_key in _model_cache:
        return _model_cache[cache_key]

    if version is None:
        champion = _get_champion_meta(aim)
        if champion is None:
            raise RuntimeError(f"No trained model found for aim={aim}")
        version = champion["version"]
        model_name = champion["model"]

    path = REGISTRY_DIR / aim / f"{version}_{model_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    pipeline = joblib.load(path)
    _model_cache[cache_key] = (pipeline, version, model_name)
    return pipeline, version, model_name


def get_feature_cols(aim: str) -> list:
    meta = _get_champion_meta(aim)
    if meta is None:
        raise RuntimeError(f"No trained model metadata found for aim={aim}")
    return meta.get("feature_cols", [])


_explainer_cache = {}


def get_explainer(aim: str, model_name: str, version: str):
    from src.explain.shap_explainer import load_explainer
    key = f"{aim}_{model_name}_{version}"
    if key not in _explainer_cache:
        _explainer_cache[key] = load_explainer(aim, model_name, version)
    return _explainer_cache[key]


def clear_explainer_cache():
    _explainer_cache.clear()


def clear_cache():
    _model_cache.clear()
    _explainer_cache.clear()
