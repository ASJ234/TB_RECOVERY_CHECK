import numpy as np
import pandas as pd
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, recall_score,
    precision_score, roc_auc_score, accuracy_score,
)
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "models" / "metadata"


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str, aim: str, version: str,
                   X_train: pd.DataFrame = None, feature_cols: list = None):
    if len(X_test) == 0:
        print(f"  {model_name} ({version}): no test set to evaluate")
        return {}

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model": model_name,
        "version": version,
        "aim": aim,
        "n_test": len(X_test),
        "accuracy": float(np.mean(y_pred == y_test)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)) if len(np.unique(y_test)) > 1 else float("nan"),
        "avg_precision": float(average_precision_score(y_test, y_proba)),
    }

    print(f"  {model_name} ({version}): Test AUC={metrics['roc_auc']:.3f}, "
          f"F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f}")

    _plot_roc_curve(y_test, y_proba, model_name, aim, version)
    _plot_pr_curve(y_test, y_proba, model_name, aim, version)
    _plot_confusion_matrix(y_test, y_pred, model_name, aim, version)

    # Generate SHAP explanations
    X_explain = X_train if X_train is not None else X_test
    if X_explain is not None and feature_cols is not None:
        generate_shap_explanations(pipeline, X_explain, feature_cols, aim, model_name, version, is_loocv=False)

    return metrics


def _plot_roc_curve(y_true, y_proba, model_name, aim, version):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {model_name} ({version})")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    out_dir = REPORTS_DIR / aim
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"roc_{model_name}_{version}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_pr_curve(y_true, y_proba, model_name, aim, version):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve - {model_name} ({version})")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)

    out_dir = REPORTS_DIR / aim
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"pr_{model_name}_{version}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def evaluate_loocv(per_fold: list, model_name: str = "strict_logistic",
                    aim: str = "aim1_non_conversion", version: str = "v1",
                    X: pd.DataFrame = None, y: pd.Series = None,
                    pipeline = None, feature_cols: list = None):
    df = pd.DataFrame(per_fold)

    y_true = df["true_label"].values
    y_pred = df["predicted_label"].values
    y_proba = df["probability"].values

    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    roc_auc_val = float("nan")
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc_val = float(roc_auc_score(y_true, y_proba))
        except Exception:
            pass

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "model": model_name,
        "version": version,
        "aim": aim,
        "cv_method": "LOOCV",
        "n_samples": len(per_fold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_val,
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
        },
    }

    print(f"\n  LOOCV Results ({model_name} {version}):")
    print(f"    Accuracy:  {accuracy:.3f} ({int(tn + tp)}/{len(per_fold)})")
    print(f"    Precision: {precision:.3f}")
    print(f"    Recall:    {recall:.3f}")
    print(f"    F1:        {f1:.3f}")
    print(f"    AUC-ROC:   {roc_auc_val:.3f}" if not np.isnan(roc_auc_val) else "    AUC-ROC:   N/A")
    print(f"    Confusion: TN={int(tn)} FP={int(fp)} FN={int(fn)} TP={int(tp)}")

    print("\n  Per-fold predictions:")
    print(f"  {'Sample ID':25s} {'True':5s} {'Pred':5s} {'Prob':6s} {'Disc':5s}")
    print(f"  {'-'*25} {'-'*5} {'-'*5} {'-'*6} {'-'*5}")
    for row in per_fold:
        marker = " *" if row["discordant"] else "  "
        print(f"  {row['sample_id']:25s} {row['true_label']:5d} {row['predicted_label']:5d} "
              f"{row['probability']:.4f} {'Y' if row['discordant'] else 'N':5s}{marker}")

    print("\n  Summary:")
    correct = df["correct"].sum()
    failures_correct = df[(df["true_label"] == 1) & (df["correct"])].shape[0]
    failures_total = df[df["true_label"] == 1].shape[0]
    converted_correct = df[(df["true_label"] == 0) & (df["correct"])].shape[0]
    converted_total = df[df["true_label"] == 0].shape[0]
    false_pos = df[(df["true_label"] == 0) & (df["predicted_label"] == 1)].shape[0]
    false_neg = df[(df["true_label"] == 1) & (df["predicted_label"] == 0)].shape[0]
    print(f"    Correct: {int(correct)}/{len(per_fold)} ({accuracy*100:.1f}%)")
    print(f"    Failures detected: {failures_correct}/{failures_total}")
    print(f"    Converted detected: {converted_correct}/{converted_total}")
    print(f"    False alarms (FP): {false_pos}")
    print(f"    Missed failures (FN): {false_neg}")

    _plot_loocv_cm(y_true, y_pred, model_name, aim, version)

    report = {
        "metrics": metrics,
        "per_fold": per_fold,
    }
    out_dir = REPORTS_DIR / aim
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"loocv_{model_name}_{version}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"    Report saved: {report_path}")

    # Generate SHAP explanations
    if X is not None and pipeline is not None and feature_cols is not None:
        generate_shap_explanations(pipeline, X, feature_cols, aim, model_name, version, is_loocv=True)

    return metrics


def _plot_loocv_cm(y_true, y_pred, model_name, aim, version):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Converted", "Failure"],
           yticklabels=["Converted", "Failure"],
           xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"LOOCV Confusion Matrix - {model_name} ({version})")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    out_dir = REPORTS_DIR / aim
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"loocv_cm_{model_name}_{version}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion_matrix(y_true, y_pred, model_name, aim, version):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["Negative", "Positive"],
           yticklabels=["Negative", "Positive"],
           xlabel="Predicted", ylabel="Actual")
    ax.set_title(f"Confusion Matrix - {model_name} ({version})")

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    out_dir = REPORTS_DIR / aim
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"cm_{model_name}_{version}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def generate_shap_explanations(pipeline, X, feature_cols, aim, model_name, version, is_loocv=False):
    """Generate SHAP explainer and save explanations and summary plots."""
    try:
        from src.explain import (
            create_explainer, compute_global_shap,
            plot_shap_summary, global_explanation_to_json,
            save_explainer, save_global_explanation
        )
        from src.explain.shap_explainer import _get_background_sample
        from src.pipeline.config import get_config

        config = get_config()
        if not config.xai.enabled:
            return

        bg_limit = config.xai.background_samples
        X_bg = _get_background_sample(X[feature_cols], bg_limit)

        explainer, preprocessor = create_explainer(pipeline, X_bg, model_name)
        global_shap = compute_global_shap(explainer, X_bg, feature_cols, preprocessor, max_samples=bg_limit)

        # Save explainer & global JSON
        save_explainer(explainer, aim, model_name, version)
        save_global_explanation(global_shap, aim, model_name, version)

        # Generate summary plot path
        from src.explain.shap_explainer import _get_transformed_feature_names
        trans_feature_names = _get_transformed_feature_names(preprocessor, feature_cols)
        X_transformed = preprocessor.transform(X_bg)

        prefix = "loocv_" if is_loocv else ""
        plot_dir = REPORTS_DIR / aim
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_dir / f"{prefix}shap_summary_{model_name}_{version}.png"

        plot_shap_summary(
            global_shap["shap_values"],
            X_transformed,
            trans_feature_names,
            plot_path,
            plot_type="bar",
            max_display=config.xai.max_display_features,
            figsize=config.xai.summary_figsize
        )
        print(f"    SHAP artifacts saved: {plot_path}")
    except Exception as e:
        print(f"    Warning: Failed to generate SHAP explanations: {e}")
