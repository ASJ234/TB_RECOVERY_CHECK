import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, f1_score, recall_score,
    precision_score, roc_auc_score
)
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parents[2] / "models" / "metadata"


def evaluate_model(pipeline, X_test: pd.DataFrame, y_test: pd.Series,
                   model_name: str, aim: str, version: str):
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
