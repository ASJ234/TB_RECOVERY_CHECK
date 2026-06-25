import numpy as np
import pandas as pd
import warnings
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from typing import Optional

from src.monitoring.statistical_tests import psi as compute_psi


def _get_reference_predictions(
    model,
    reference_data: pd.DataFrame,
    reference_labels: pd.Series,
    feature_cols: list,
) -> dict:
    df = reference_data[feature_cols].copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        proba = model.predict_proba(df)
        preds = model.predict(df)
    return {
        "predictions": preds,
        "probabilities": proba[:, 1],
        "labels": reference_labels.values,
        "n": len(df),
    }


def _get_current_metrics(
    model,
    current_data: pd.DataFrame,
    current_labels: pd.Series,
    feature_cols: list,
) -> dict:
    df = current_data[feature_cols].copy()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        proba = model.predict_proba(df)
        preds = model.predict(df)

    y_true = current_labels.values
    y_prob = proba[:, 1]
    metrics = {
        "n": len(df),
        "accuracy": float(accuracy_score(y_true, preds)),
    }
    if len(np.unique(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    else:
        metrics["auc"] = float("nan")
        metrics["avg_precision"] = float("nan")
    return metrics, y_prob, preds


def compute_model_drift(
    model,
    reference_data: pd.DataFrame,
    reference_labels: pd.Series,
    current_data: pd.DataFrame,
    current_labels: pd.Series,
    feature_cols: list,
    reference_metrics: Optional[dict] = None,
    auc_drop_threshold: float = 0.05,
    prediction_psi_threshold: float = 0.1,
) -> dict:
    ref_preds_info = _get_reference_predictions(
        model, reference_data, reference_labels, feature_cols
    )
    cur_metrics_dict, cur_probs, cur_preds = _get_current_metrics(
        model, current_data, current_labels, feature_cols
    )

    if reference_metrics is None:
        ref_auc = float(
            roc_auc_score(ref_preds_info["labels"], ref_preds_info["probabilities"])
            if len(np.unique(ref_preds_info["labels"])) > 1
            else float("nan")
        )
        reference_metrics = {
            "auc": ref_auc,
            "accuracy": float(accuracy_score(ref_preds_info["labels"], ref_preds_info["predictions"])),
        }
        if not np.isnan(ref_auc):
            reference_metrics["avg_precision"] = float(
                average_precision_score(ref_preds_info["labels"], ref_preds_info["probabilities"])
            )
        else:
            reference_metrics["avg_precision"] = float("nan")

    cur_auc = cur_metrics_dict.get("auc", float("nan"))
    ref_auc = reference_metrics.get("auc", float("nan"))
    auc_drop = (ref_auc - cur_auc) if not (np.isnan(ref_auc) or np.isnan(cur_auc)) else float("nan")
    performance_drift = bool(
        not np.isnan(auc_drop) and auc_drop > auc_drop_threshold
    )

    prediction_psi_result = compute_psi(
        ref_preds_info["probabilities"],
        cur_probs,
        num_bins=10,
        feature_name="prediction_score",
    )
    prediction_drift = prediction_psi_result["drift_detected"]
    prediction_psi_val = prediction_psi_result["statistic"]

    return {
        "model_drift_detected": performance_drift or prediction_drift,
        "performance_drift": performance_drift,
        "prediction_drift": prediction_drift,
        "auc_drop": auc_drop if not np.isnan(auc_drop) else None,
        "auc_drop_threshold": auc_drop_threshold,
        "prediction_psi": prediction_psi_val if not np.isnan(prediction_psi_val) else None,
        "prediction_psi_threshold": prediction_psi_threshold,
        "reference_metrics": {
            "n": int(ref_preds_info["n"]),
            "auc": reference_metrics.get("auc"),
            "accuracy": reference_metrics.get("accuracy"),
            "avg_precision": reference_metrics.get("avg_precision"),
        },
        "current_metrics": {
            "n": int(cur_metrics_dict["n"]),
            "auc": cur_metrics_dict.get("auc"),
            "accuracy": cur_metrics_dict["accuracy"],
            "avg_precision": cur_metrics_dict.get("avg_precision"),
        },
    }
