import numpy as np
import pandas as pd
from typing import Optional

from src.monitoring.statistical_tests import (
    ks_test,
    chi_square_test,
    psi,
    categorical_psi,
)


def _infer_feature_types(df: pd.DataFrame) -> dict:
    types = {}
    for c in df.columns:
        nunique = df[c].dropna().nunique()
        if nunique == 1:
            types[c] = "constant"
        elif nunique <= 2:
            types[c] = "binary"
        elif df[c].dtype in (np.float64, np.int64, float, int):
            types[c] = "numeric"
        else:
            types[c] = "categorical"
    return types


def compute_data_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: Optional[list] = None,
    feature_types: Optional[dict] = None,
    psi_bins: int = 10,
) -> dict:
    if feature_cols is None:
        feature_cols = [c for c in reference_df.columns if c in current_df.columns]
    if feature_types is None:
        feature_types = _infer_feature_types(reference_df[feature_cols])

    per_feature = {}
    drift_count = 0
    total_tests = 0

    for col in feature_cols:
        ftype = feature_types.get(col, "numeric")
        ref_vals = reference_df[col].dropna().values
        cur_vals = current_df[col].dropna().values

        if len(ref_vals) == 0 or len(cur_vals) == 0:
            per_feature[col] = {"error": "empty column", "drift_detected": False}
            continue

        if ftype == "constant":
            per_feature[col] = {
                "type": "constant",
                "drift_detected": False,
                "reference_n": int(len(ref_vals)),
                "current_n": int(len(cur_vals)),
                "info": "constant feature — skipped",
            }
            continue

        results = []
        if ftype == "numeric":
            ks_res = ks_test(ref_vals, cur_vals, feature_name=col)
            psi_res = psi(ref_vals, cur_vals, num_bins=psi_bins, feature_name=col)
            results = [ks_res, psi_res]
        elif ftype == "binary":
            chi_res = chi_square_test(ref_vals, cur_vals, feature_name=col)
            cpsi_res = categorical_psi(ref_vals, cur_vals, feature_name=col)
            results = [chi_res, cpsi_res]
        else:
            chi_res = chi_square_test(ref_vals, cur_vals, feature_name=col)
            cpsi_res = categorical_psi(ref_vals, cur_vals, feature_name=col)
            results = [chi_res, cpsi_res]

        feature_drifted = any(r["drift_detected"] for r in results)
        if feature_drifted:
            drift_count += 1
        total_tests += len(results)

        per_feature[col] = {
            "type": ftype,
            "tests": results,
            "drift_detected": feature_drifted,
            "reference_n": int(len(ref_vals)),
            "current_n": int(len(cur_vals)),
        }

    n_features = len(feature_cols)
    drift_ratio = drift_count / n_features if n_features > 0 else 0.0

    return {
        "data_drift_detected": drift_ratio > 0.3,
        "drift_ratio": drift_ratio,
        "n_features": n_features,
        "drift_count": drift_count,
        "total_tests": total_tests,
        "per_feature": per_feature,
    }
