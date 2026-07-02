import numpy as np
import pandas as pd
import shap
from datetime import datetime
from typing import Any


def global_explanation_to_json(
    aim: str,
    model: str,
    version: str,
    feature_names: list[str],
    mean_abs_shap: np.ndarray,
    std_shap: np.ndarray,
    base_value: float,
    n_background_samples: int,
) -> dict[str, Any]:
    """Convert global SHAP explanation to JSON-serializable dict."""
    return {
        "aim": aim,
        "model": model,
        "version": version,
        "feature_names": feature_names,
        "mean_abs_shap": mean_abs_shap.tolist(),
        "std_shap": std_shap.tolist(),
        "base_value": float(base_value),
        "n_background_samples": int(n_background_samples),
        "generated_at": datetime.now().isoformat(),
    }


def instance_explanation_to_json(
    aim: str,
    model: str,
    version: str,
    prediction: int,
    probability: float,
    base_value: float,
    features: dict[str, float],
    shap_values: dict[str, float],
    feature_names: list[str],
    plot_data: dict | None = None,
) -> dict[str, Any]:
    """Convert instance SHAP explanation to JSON-serializable dict."""
    return {
        "aim": aim,
        "model": model,
        "version": version,
        "prediction": int(prediction),
        "probability": float(probability),
        "base_value": float(base_value),
        "features": features,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "plot_data": plot_data,
        "generated_at": datetime.now().isoformat(),
    }


def create_instance_plot_data(
    shap_values: np.ndarray,
    base_value: float,
    features: np.ndarray,
    feature_names: list[str],
    summary_plot_b64: str | None = None,
    waterfall_plot_b64: str | None = None,
    force_plot_b64: str | None = None,
) -> dict:
    """Create plot data dict for instance explanation."""
    return {
        "summary_plot_base64": summary_plot_b64,
        "waterfall_plot_base64": waterfall_plot_b64,
        "force_plot_base64": force_plot_b64,
    }


def map_shap_to_original_features(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    original_feature_names: list[str],
    preprocessor,
) -> tuple[dict[str, float], list[str]]:
    """
    Map SHAP values from transformed features back to original features.
    
    Aggregates SHAP values for one-hot encoded columns back to original categorical features.
    """
    feature_to_indices = {feat: [] for feat in original_feature_names}

    for idx, tname in enumerate(transformed_feature_names):
        for ofeat in original_feature_names:
            if tname == ofeat or tname.startswith(ofeat + "_") or tname.startswith(ofeat + " "):
                feature_to_indices[ofeat].append(idx)
                break

    original_shap = {}
    for feat in original_feature_names:
        indices = feature_to_indices.get(feat, [])
        if indices:
            original_shap[feat] = float(np.sum(shap_values[indices]))
        else:
            original_shap[feat] = 0.0

    original_shap = {k: v for k, v in original_shap.items() if v != 0.0}
    return original_shap, list(original_shap.keys())


def create_background_sample(
    X: pd.DataFrame,
    n_samples: int = 500,
    random_state: int = 42,
    method: str = "kmeans",
) -> Any:
    """Create background sample for SHAP explainer."""
    if len(X) <= n_samples:
        return X
    
    if method == "kmeans":
        return shap.kmeans(X, n_samples, random_state=random_state)
    elif method == "random":
        return X.sample(n=n_samples, random_state=random_state)
    else:
        return X.sample(n=n_samples, random_state=random_state)