"""
XAI Report Generator
Generates SHAP-based explainability reports for all trained TB Recovery Check models.
Produces summary plots, waterfall plots for sample patients, and a consolidated report.
"""
import sys
import json
import warnings
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib

from src.explain.shap_explainer import (
    create_explainer,
    compute_global_shap,
    compute_instance_shap,
    load_explainer,
    load_global_explanation,
    save_explainer,
    save_global_explanation,
    EXPLAINERS_DIR,
    EXPLANATIONS_DIR,
)
from src.explain.visualizations import (
    plot_shap_summary,
    plot_shap_waterfall,
    plot_shap_force,
)

MODELS_DIR = ROOT / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
REPORTS_DIR = ROOT / "reports" / "xai"
DATA_DIR = ROOT / "data" / "cleaned"


def get_champion_models():
    """Scan the registry directories for actual model .pkl files on disk."""
    if not REGISTRY_DIR.exists():
        print(f"ERROR: Registry not found at {REGISTRY_DIR}")
        return []

    # Only process known aim directories
    KNOWN_AIMS = {"aim1_non_conversion", "aim1_non_conversion_strict", "aim2_contact_risk"}

    champions = []
    for aim_dir in sorted(REGISTRY_DIR.iterdir()):
        if not aim_dir.is_dir():
            continue
        aim = aim_dir.name
        if aim not in KNOWN_AIMS:
            continue
        # Find all .pkl files, pick the latest by version number
        pkl_files = sorted(aim_dir.glob("*.pkl"))
        if not pkl_files:
            continue
        # Parse version and model name from filename like v3_xgboost.pkl
        best = None
        best_version_num = -1
        for pkl in pkl_files:
            stem = pkl.stem  # e.g. "v3_xgboost"
            parts = stem.split("_", 1)
            if len(parts) == 2:
                ver_str, model_name = parts[0], parts[1]
                try:
                    ver_num = int(ver_str.lstrip("v"))
                except ValueError:
                    continue
                if ver_num > best_version_num:
                    best_version_num = ver_num
                    best = {"aim": aim, "model_name": model_name, "version": ver_str}
        if best:
            champions.append(best)
    return champions


def load_pipeline(aim: str, model_name: str, version: str):
    """Load a saved sklearn pipeline from the registry."""
    pipeline_path = REGISTRY_DIR / aim / f"{version}_{model_name}.pkl"
    if not pipeline_path.exists():
        print(f"  WARNING: Pipeline not found at {pipeline_path}")
        return None
    return joblib.load(pipeline_path)


def load_data_for_aim(aim: str):
    """Load the cleaned/imputed dataset for a given aim."""
    if "strict" in aim:
        # The strict model lives in its own CSV that has TARGET_STRICT_M2 / M2_DISCORDANT
        data_path = DATA_DIR / "aim1_patients_strict.csv"
    elif "aim1" in aim:
        data_path = DATA_DIR / "aim1_patients_imputed.csv"
    elif "aim2" in aim:
        data_path = DATA_DIR / "aim2_contacts_imputed.csv"
    else:
        print(f"  WARNING: Unknown aim '{aim}', skipping data load")
        return None, None

    if not data_path.exists():
        print(f"  WARNING: Data not found at {data_path}")
        return None, None

    df = pd.read_csv(data_path)

    # Strict models use a different feature set
    if "strict" in aim:
        from src.data.feature_engineering import get_aim1_strict_features_target
        df = get_aim1_strict_features_target(df)[0].copy()  # returns (df_model, feature_cols)
        feature_cols = ["SEX", "AGE (YEARS)", "BMI", "baseline_symptom_count"]
        feature_cols = [c for c in feature_cols if c in df.columns]
        return df, feature_cols

    # Standard aim models
    if "aim1" in aim:
        from src.data.feature_engineering import get_aim1_features_target
        _, feature_cols = get_aim1_features_target(df)
    else:
        from src.data.feature_engineering import get_aim2_features_target
        _, feature_cols = get_aim2_features_target(df)

    return df, feature_cols


def generate_report_for_model(aim: str, model_name: str, version: str):
    """Generate all XAI plots and data for a single model."""
    print(f"\n{'='*60}")
    print(f"  Generating XAI report: {aim} / {model_name} / {version}")
    print(f"{'='*60}")

    report_dir = REPORTS_DIR / aim / f"{model_name}_{version}"
    report_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load pipeline
    pipeline = load_pipeline(aim, model_name, version)
    if pipeline is None:
        return None

    # 2. Load data
    df, feature_cols = load_data_for_aim(aim)
    if df is None:
        return None

    X = df[feature_cols]
    print(f"  Dataset: {len(X)} samples, {len(feature_cols)} features")

    # 3. Create or load explainer
    explainer_path = EXPLAINERS_DIR / aim / f"{model_name}_{version}_explainer.joblib"
    preprocessor = pipeline.named_steps["preprocessor"]

    if explainer_path.exists():
        print(f"  Loading cached explainer from {explainer_path.name}")
        explainer = joblib.load(explainer_path)
        # Detect stale XGBoost explainer: categorical models require tree_path_dependent.
        # If the cached explainer lacks this, delete and rebuild it.
        is_tree_exp = hasattr(explainer, "feature_perturbation")
        needs_rebuild = (
            model_name == "xgboost"
            and is_tree_exp
            and getattr(explainer, "feature_perturbation", None) != "tree_path_dependent"
        )
        if needs_rebuild:
            print("  Stale XGBoost explainer detected (missing tree_path_dependent). Rebuilding...")
            explainer_path.unlink(missing_ok=True)
            explainer, preprocessor = create_explainer(pipeline, X, model_name)
            save_explainer(explainer, aim, model_name, version)
    else:
        print(f"  Creating new SHAP explainer...")
        explainer, preprocessor = create_explainer(pipeline, X, model_name)
        save_explainer(explainer, aim, model_name, version)

    # 4. Compute global SHAP
    print(f"  Computing global SHAP values...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global_result = compute_global_shap(explainer, X, feature_cols, preprocessor)

    save_global_explanation(global_result, aim, model_name, version)

    # 5. Generate Global Summary Plots
    aim_label = aim.replace("_", " ").title()
    model_label = model_name.replace("_", " ").upper()
    summary_title = f"Global Feature Importance — {aim_label}"
    summary_subtitle = f"Model: {model_label} ({version})  |  Mean |SHAP| across {global_result['n_samples']} samples"

    print(f"  Generating global summary bar plot...")
    plot_shap_summary(
        global_result["shap_values"],
        global_result["features"],
        global_result["feature_names"],
        out_path=report_dir / "shap_summary_bar.png",
        plot_type="bar",
        max_display=20,
        title=summary_title,
        subtitle=summary_subtitle,
    )

    # 6. Instance-level explanations for a few sample patients
    sample_indices = list(range(min(3, len(X))))
    for i, idx in enumerate(sample_indices):
        print(f"  Generating instance explanation for patient #{idx}...")
        row = X.iloc[[idx]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            instance_result = compute_instance_shap(explainer, row, feature_cols, preprocessor)

        # instance_result["features"] is already in the correct space
        # (raw for tree_path_dependent XGBoost, transformed otherwise)
        inst_features = instance_result["features"]
        inst_title_waterfall = f"Patient #{idx} — SHAP Waterfall Explanation"
        inst_title_force = f"Patient #{idx} — SHAP Force Plot"
        inst_subtitle = f"{aim_label}  |  Model: {model_label} ({version})"

        # Waterfall plot
        try:
            plot_shap_waterfall(
                instance_result["shap_values"],
                instance_result["base_value"],
                inst_features,
                instance_result["feature_names"],
                out_path=report_dir / f"waterfall_patient_{idx}.png",
                max_display=15,
                title=inst_title_waterfall,
                subtitle=inst_subtitle,
            )
        except Exception as e:
            print(f"    Waterfall plot failed: {e}")

        # Force plot
        try:
            plot_shap_force(
                instance_result["shap_values"],
                instance_result["base_value"],
                inst_features,
                instance_result["feature_names"],
                out_path=report_dir / f"force_patient_{idx}.png",
                title=inst_title_force,
                subtitle=inst_subtitle,
            )
        except Exception as e:
            print(f"    Force plot failed: {e}")

    # 7. Write text summary
    mean_abs = global_result["mean_abs_shap"]
    if mean_abs.ndim == 2:
        mean_abs = np.mean(np.abs(mean_abs), axis=1) if mean_abs.shape[0] == len(global_result["feature_names"]) else np.mean(np.abs(mean_abs), axis=0)
    feature_importance = sorted(
        zip(global_result["feature_names"], mean_abs),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    summary_lines = [
        f"# XAI Report: {aim} / {model_name} ({version})",
        f"",
        f"Generated at: {pd.Timestamp.now().isoformat()}",
        f"Dataset: {len(X)} samples, {len(feature_cols)} original features",
        f"Transformed features: {len(global_result['feature_names'])}",
        f"Base value (expected output): {global_result['base_value']:.4f}",
        f"",
        f"## Top 20 Most Important Features (mean |SHAP|)",
        f"",
        f"{'Rank':<6} {'Feature':<45} {'Mean |SHAP|':>12}",
        f"{'-'*6} {'-'*45} {'-'*12}",
    ]
    for rank, (fname, importance) in enumerate(feature_importance[:20], 1):
        summary_lines.append(f"{rank:<6} {fname:<45} {importance:>12.6f}")

    summary_lines.extend([
        f"",
        f"## Output Files",
        f"- shap_summary_bar.png  — Global feature importance bar chart",
    ])
    for idx in sample_indices:
        summary_lines.append(f"- waterfall_patient_{idx}.png  — Waterfall plot for patient #{idx}")
        summary_lines.append(f"- force_patient_{idx}.png     — Force plot for patient #{idx}")

    summary_path = report_dir / "report_summary.md"
    with open(summary_path, "w") as f:
        f.write("\n".join(summary_lines))

    print(f"  Report written to: {report_dir}")
    return report_dir


def main():
    print("=" * 60)
    print("  TB Recovery Check — XAI Report Generator")
    print("=" * 60)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    champions = get_champion_models()
    if not champions:
        print("No models found in registry. Run the training pipeline first.")
        return

    print(f"\nFound {len(champions)} champion model(s) to explain:")
    for c in champions:
        print(f"  - {c['aim']}: {c['model_name']} ({c['version']})")

    report_dirs = []
    for c in champions:
        try:
            rd = generate_report_for_model(c["aim"], c["model_name"], c["version"])
            if rd:
                report_dirs.append(rd)
        except Exception as e:
            print(f"  ERROR generating report for {c['aim']}/{c['model_name']}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  XAI Report Generation Complete")
    print(f"  {len(report_dirs)} report(s) generated in: {REPORTS_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
