import warnings
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import shap
import joblib
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
EXPLAINERS_DIR = MODELS_DIR / "explainers"
EXPLANATIONS_DIR = MODELS_DIR / "explanations"

_SKLEARN_WARN = "Found unknown categories in columns"


def create_explainer(pipeline, X_background: pd.DataFrame, model_type: str):
    """Create appropriate SHAP explainer based on model type."""
    classifier = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    X_transformed = preprocessor.transform(X_background)

    if hasattr(classifier, "estimators_") or model_type in ("random_forest", "xgboost"):
        if model_type == "xgboost":
            # XGBoost with categorical features requires tree_path_dependent
            explainer = shap.TreeExplainer(
                classifier,
                feature_perturbation="tree_path_dependent",
            )
        else:
            try:
                explainer = shap.TreeExplainer(
                    classifier,
                    data=X_transformed,
                    feature_perturbation="interventional",
                    approximate=True,
                )
            except Exception:
                explainer = shap.TreeExplainer(
                    classifier,
                    feature_perturbation="tree_path_dependent",
                )
    else:
        explainer = shap.LinearExplainer(
            classifier,
            X_transformed,
            feature_perturbation="interventional",
        )

    return explainer, preprocessor


def _get_background_sample(X: pd.DataFrame, n_samples: int = 500, random_state: int = 42) -> pd.DataFrame:
    """Get stratified background sample for SHAP computation."""
    if len(X) <= n_samples:
        return X
    return X.sample(n=n_samples, random_state=random_state)


def compute_global_shap(
    explainer,
    X: pd.DataFrame,
    feature_names: list[str],
    preprocessor,
    max_samples: int = 500,
) -> dict:
    """Compute global SHAP values (mean |SHAP|) for a dataset."""
    X_sample = _get_background_sample(X, max_samples)
    X_transformed = preprocessor.transform(X_sample)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SKLEARN_WARN)
        # check_additivity is TreeExplainer-only; LinearExplainer rejects it.
        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X_transformed, check_additivity=False)
        else:
            shap_values = explainer.shap_values(X_transformed)

    # shap_values can be a list (multi-class) or a 3-D array (XGBoost binary)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    std_shap = np.std(shap_values, axis=0)
    transformed_feature_names = _get_transformed_feature_names(preprocessor, feature_names)

    # If SHAP width matches original features, use original names (handles some
    # tree_path_dependent cases where OHE isn't applied); otherwise use transformed.
    if shap_values.shape[1] == len(feature_names):
        final_feature_names = list(feature_names)
        features_for_plot = X_sample.values
    else:
        final_feature_names = transformed_feature_names
        features_for_plot = X_transformed

    base_val = explainer.expected_value
    if not np.isscalar(base_val):
        base_val = base_val[1]

    return {
        "shap_values": shap_values,
        "features": features_for_plot,
        "feature_names": final_feature_names,
        "mean_abs_shap": mean_abs_shap,
        "std_shap": std_shap,
        "base_value": float(base_val),
        "n_samples": len(X_sample),
    }


def compute_instance_shap(
    explainer,
    X_instance: pd.DataFrame,
    feature_names: list[str],
    preprocessor,
) -> dict:
    """Compute SHAP values for a single instance."""
    X_transformed = preprocessor.transform(X_instance)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_SKLEARN_WARN)
        # check_additivity is TreeExplainer-only; LinearExplainer rejects it.
        if isinstance(explainer, shap.TreeExplainer):
            shap_values = explainer.shap_values(X_transformed, check_additivity=False)
        else:
            shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]

    base_value = explainer.expected_value
    if not np.isscalar(base_value):
        base_value = base_value[1]

    transformed_feature_names = _get_transformed_feature_names(preprocessor, feature_names)
    # Match feature axis to the right names/values
    if shap_values.shape[1] == len(feature_names):
        final_feature_names = list(feature_names)
        features_arr = X_instance.values[0]
    else:
        final_feature_names = transformed_feature_names
        features_arr = X_transformed[0]

    return {
        "shap_values": shap_values[0],
        "features": features_arr,
        "feature_names": final_feature_names,
        "base_value": float(base_value),
    }


def _get_transformed_feature_names(preprocessor, original_feature_names: list[str]) -> list[str]:
    """Get feature names after ColumnTransformer transformation."""
    feature_names = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps") and "encoder" in trans.named_steps:
            encoder = trans.named_steps["encoder"]
            if hasattr(encoder, "get_feature_names_out"):
                encoded_names = encoder.get_feature_names_out(cols)
                feature_names.extend(encoded_names)
            else:
                feature_names.extend(cols)
        else:
            feature_names.extend(cols)
    return feature_names


def map_shap_to_original_features(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    original_feature_names: list[str],
    preprocessor,
) -> tuple[np.ndarray, list[str]]:
    """Aggregate SHAP values from one-hot encoded features back to original features."""
    feature_to_indices = {feat: [] for feat in original_feature_names}

    for idx, tname in enumerate(transformed_feature_names):
        for ofeat in original_feature_names:
            if tname == ofeat or tname.startswith(ofeat + "_") or tname.startswith(ofeat + " "):
                feature_to_indices[ofeat].append(idx)
                break

    aggregated_shap = np.zeros(len(original_feature_names))
    for i, feat in enumerate(original_feature_names):
        indices = feature_to_indices.get(feat, [])
        if indices:
            aggregated_shap[i] = np.sum(shap_values[indices])
        else:
            aggregated_shap[i] = 0.0

    return aggregated_shap, original_feature_names


def save_explainer(explainer, aim: str, model_name: str, version: str):
    """Save SHAP explainer to disk."""
    EXPLAINERS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPLAINERS_DIR / aim / f"{model_name}_{version}_explainer.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(explainer, path)
    return path


def load_explainer(aim: str, model_name: str, version: str):
    """Load SHAP explainer from disk."""
    path = EXPLAINERS_DIR / aim / f"{model_name}_{version}_explainer.joblib"
    return joblib.load(path)


def save_global_explanation(global_result: dict, aim: str, model_name: str, version: str):
    """Save global SHAP explanation as JSON."""
    EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = EXPLANATIONS_DIR / aim / f"{model_name}_{version}_global.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    import json
    from datetime import datetime

    serializable = {
        "aim": aim,
        "model": model_name,
        "version": version,
        "feature_names": global_result["feature_names"],
        "mean_abs_shap": global_result["mean_abs_shap"].tolist(),
        "std_shap": global_result["std_shap"].tolist(),
        "base_value": global_result["base_value"],
        "n_background_samples": global_result["n_samples"],
        "generated_at": datetime.now().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    return path


def load_global_explanation(aim: str, model_name: str, version: str) -> dict:
    """Load global SHAP explanation from JSON."""
    import json
    path = EXPLANATIONS_DIR / aim / f"{model_name}_{version}_global.json"
    with open(path) as f:
        return json.load(f)