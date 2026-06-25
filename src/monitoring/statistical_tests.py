import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon


def ks_test(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "",
) -> dict:
    stat, p_value = ks_2samp(reference, current)
    return {
        "feature": feature_name,
        "test": "ks",
        "statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05),
    }


def chi_square_test(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "",
) -> dict:
    ref_counts = pd.Series(reference).value_counts()
    cur_counts = pd.Series(current).value_counts()
    all_categories = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_dist = np.array([ref_counts.get(c, 0) for c in all_categories], dtype=float)
    cur_dist = np.array([cur_counts.get(c, 0) for c in all_categories], dtype=float)
    if ref_dist.sum() == 0 or cur_dist.sum() == 0:
        return {
            "feature": feature_name,
            "test": "chi_square",
            "statistic": float("nan"),
            "p_value": float("nan"),
            "drift_detected": False,
        }
    ref_prop = ref_dist / ref_dist.sum() * cur_dist.sum()
    valid = ref_prop > 0
    if not valid.all():
        table = np.array([ref_dist, cur_dist])
        table = table[:, valid]
        if table.shape[1] < 2:
            return {
                "feature": feature_name,
                "test": "chi_square",
                "statistic": float("nan"),
                "p_value": float("nan"),
                "drift_detected": False,
            }
    else:
        table = np.array([ref_dist, cur_dist])
    with np.errstate(all="ignore"):
        stat, p_value, _, _ = chi2_contingency(table, correction=False)
    return {
        "feature": feature_name,
        "test": "chi_square",
        "statistic": float(stat) if not np.isnan(stat) else float("nan"),
        "p_value": float(p_value) if not np.isnan(p_value) else float("nan"),
        "drift_detected": bool(p_value < 0.05) if not np.isnan(p_value) else False,
    }


def psi(
    reference: np.ndarray,
    current: np.ndarray,
    num_bins: int = 10,
    feature_name: str = "",
) -> dict:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return {
            "feature": feature_name,
            "test": "psi",
            "statistic": float("nan"),
            "drift_detected": False,
        }
    all_vals = np.concatenate([reference, current])
    if np.all(all_vals == all_vals[0]):
        return {
            "feature": feature_name,
            "test": "psi",
            "statistic": 0.0,
            "drift_detected": False,
        }
    bins = np.percentile(reference, np.linspace(0, 100, num_bins + 1))
    bins[0] = -np.inf
    bins[-1] = np.inf
    ref_counts = np.histogram(reference, bins=bins)[0].astype(float)
    cur_counts = np.histogram(current, bins=bins)[0].astype(float)
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    psi_val = np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct))
    return {
        "feature": feature_name,
        "test": "psi",
        "statistic": float(psi_val),
        "drift_detected": bool(psi_val > 0.1),
    }


def categorical_psi(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "",
) -> dict:
    ref_counts = pd.Series(reference).value_counts()
    cur_counts = pd.Series(current).value_counts()
    all_cats = sorted(set(ref_counts.index) | set(cur_counts.index))
    ref_pct = np.array([ref_counts.get(c, 0) for c in all_cats], dtype=float)
    cur_pct = np.array([cur_counts.get(c, 0) for c in all_cats], dtype=float)
    ref_pct = ref_pct / ref_pct.sum() if ref_pct.sum() > 0 else ref_pct
    cur_pct = cur_pct / cur_pct.sum() if cur_pct.sum() > 0 else cur_pct
    ref_pct = np.clip(ref_pct, 1e-6, None)
    cur_pct = np.clip(cur_pct, 1e-6, None)
    psi_val = np.sum((ref_pct - cur_pct) * np.log(ref_pct / cur_pct))
    return {
        "feature": feature_name,
        "test": "categorical_psi",
        "statistic": float(psi_val),
        "drift_detected": bool(psi_val > 0.1),
    }


def js_divergence(
    reference: np.ndarray,
    current: np.ndarray,
    feature_name: str = "",
) -> dict:
    if np.issubdtype(reference.dtype, np.number):
        reference = reference[~np.isnan(reference)]
    if np.issubdtype(current.dtype, np.number):
        current = current[~np.isnan(current)]
    if len(reference) == 0 or len(current) == 0:
        return {
            "feature": feature_name,
            "test": "js_divergence",
            "statistic": float("nan"),
            "drift_detected": False,
        }
    num_bins = min(50, len(np.unique(reference)))
    if num_bins < 2:
        return {
            "feature": feature_name,
            "test": "js_divergence",
            "statistic": 0.0,
            "drift_detected": False,
        }
    all_vals = np.concatenate([reference, current])
    if np.issubdtype(all_vals.dtype, np.number):
        bins = np.histogram_bin_edges(all_vals, bins="auto")
    else:
        bins = num_bins
    ref_hist, _ = np.histogram(reference, bins=bins)
    cur_hist, _ = np.histogram(current, bins=bins)
    ref_p = ref_hist.astype(float) / ref_hist.sum()
    cur_p = cur_hist.astype(float) / cur_hist.sum()
    js_val = jensenshannon(ref_p, cur_p)
    return {
        "feature": feature_name,
        "test": "js_divergence",
        "statistic": float(js_val),
        "drift_detected": bool(js_val > 0.1),
    }
