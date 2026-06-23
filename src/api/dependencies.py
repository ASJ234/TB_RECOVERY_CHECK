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
        score = meta.get("cv_auc_mean")
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


def clear_cache():
    _model_cache.clear()
