import re
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


MONITORING_DIR = Path(__file__).resolve().parents[2] / "monitoring_reports"


def _sanitize_metric_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.\/ :]", "_", name)


def generate_drift_report(
    data_drift_result: Optional[dict] = None,
    model_drift_result: Optional[dict] = None,
) -> dict:
    report = {
        "timestamp": datetime.now().isoformat(),
        "data_drift": data_drift_result,
        "model_drift": model_drift_result,
    }

    drift_detected = False
    if data_drift_result and data_drift_result.get("data_drift_detected"):
        drift_detected = True
    if model_drift_result and model_drift_result.get("model_drift_detected"):
        drift_detected = True

    report["drift_detected"] = drift_detected
    return report


def should_retrain(report: dict, log_only: bool = True) -> bool:
    if log_only:
        return False
    return report.get("drift_detected", False)


def save_drift_report(report: dict, path: Optional[Path] = None) -> Path:
    if path is None:
        MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = MONITORING_DIR / f"drift_report_{ts}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return path


def log_to_mlflow(
    report: dict,
    experiment_name: str = "drift_monitoring",
    run_name: Optional[str] = None,
):
    import mlflow

    mlflow.set_tracking_uri("sqlite:///" + str(Path(__file__).resolve().parents[2] / "mlruns" / "mlflow.db"))
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("drift_detected", report.get("drift_detected", False))

        data_drift = report.get("data_drift", {})
        if data_drift:
            mlflow.log_metric("data_drift_ratio", data_drift.get("drift_ratio", 0.0))
            mlflow.log_metric("data_drift_count", data_drift.get("drift_count", 0))
            mlflow.log_param("data_drift_detected", data_drift.get("data_drift_detected", False))
            for feat, feat_info in data_drift.get("per_feature", {}).items():
                for test in feat_info.get("tests", []):
                    tag = _sanitize_metric_name(f"{feat}_{test['test']}")
                    mlflow.log_metric(f"{tag}_statistic", test.get("statistic", 0.0))
                    if "p_value" in test:
                        mlflow.log_metric(f"{tag}_p_value", test.get("p_value", 0.0))

        model_drift = report.get("model_drift", {})
        if model_drift:
            mlflow.log_metric("auc_drop", model_drift.get("auc_drop", 0.0) or 0.0)
            mlflow.log_metric("prediction_psi", model_drift.get("prediction_psi", 0.0) or 0.0)
            mlflow.log_param("performance_drift", model_drift.get("performance_drift", False))
            mlflow.log_param("prediction_drift", model_drift.get("prediction_drift", False))
            ref = model_drift.get("reference_metrics", {})
            cur = model_drift.get("current_metrics", {})
            for k in ("auc", "accuracy", "avg_precision"):
                rv = ref.get(k)
                cv = cur.get(k)
                if rv is not None and not (isinstance(rv, float) and np.isnan(rv)):
                    mlflow.log_metric(f"reference_{k}", rv)
                if cv is not None and not (isinstance(cv, float) and np.isnan(cv)):
                    mlflow.log_metric(f"current_{k}", cv)

        mlflow.log_dict(report, "drift_report.json")
