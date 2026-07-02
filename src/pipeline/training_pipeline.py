#!/usr/bin/env python3
import sys
import json
import argparse
import hashlib
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.clean_data import save_clean_datasets
from src.data.feature_engineering import (
    prepare_aim1_data,
    prepare_aim1_strict_data,
    prepare_aim2_data,
    save_scaler,
)
from src.models.train import (
    train_models,
    train_aim1_strict,
    update_registry_csv,
    get_champion_model,
)
from src.models.evaluate import evaluate_model, evaluate_loocv
from src.pipeline.config import get_config
from src.monitoring.synthetic_drift import generate_all_synthetic_variants
from src.monitoring.data_drift import compute_data_drift
from src.monitoring.model_drift import compute_model_drift
from src.monitoring.reporting import generate_drift_report, save_drift_report, log_to_mlflow, should_retrain

REGISTRY_DIR = Path(__file__).resolve().parents[2] / "models" / "registry"
METADATA_DIR = Path(__file__).resolve().parents[2] / "models" / "metadata"


def _compute_data_hash() -> str:
    hasher = hashlib.sha256()
    aim1_path = Path(__file__).resolve().parents[2] / "data" / "cleaned" / "aim1_patients_imputed.csv"
    aim2_path = Path(__file__).resolve().parents[2] / "data" / "cleaned" / "aim2_contacts_imputed.csv"

    for path in [aim1_path, aim2_path]:
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:16]


def _is_data_changed(current_hash: str) -> bool:
    hash_path = METADATA_DIR / "last_data_hash.txt"
    if not hash_path.exists():
        return True
    prev = hash_path.read_text().strip()
    return prev != current_hash


def _save_data_hash(current_hash: str):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    (METADATA_DIR / "last_data_hash.txt").write_text(current_hash)


def run_aim1(force: bool = False):
    print("=" * 60)
    print("AIM 1: Predicting TB non-conversion at M2/M5")
    print("=" * 60)

    config = get_config()
    aim1_cfg = config.aim1

    # ---------------------------------------------------------------
    # Phase 1 — Strict model (culture-based labels, n=11, LR + LOOCV)
    # ---------------------------------------------------------------
    print("\n--- Phase 1: Strict Model (exploratory, n=11) ---")
    X_strict, y_strict, preprocessor_strict, strict_feature_cols, sample_ids, discordant = (
        prepare_aim1_strict_data()
    )

    if X_strict is not None and len(X_strict) >= 2:
        print(f"  Strict samples: {len(X_strict)}")
        print(f"  Features: {strict_feature_cols}")
        print(f"  Class distribution: {y_strict.value_counts().to_dict()}")

        pipeline_strict, per_fold, strict_meta = train_aim1_strict(
            X_strict, y_strict, preprocessor_strict,
            strict_feature_cols, sample_ids, discordant,
            C=aim1_cfg.strict_C,
            random_state=aim1_cfg.random_state,
        )

        loocv_metrics = evaluate_loocv(
            per_fold, model_name="strict_logistic",
            aim="aim1_non_conversion", version=strict_meta["version"],
        )

        save_scaler(preprocessor_strict, "aim1_strict", strict_meta["version"])

        strict_meta["cv_accuracy"] = loocv_metrics.get("accuracy", float("nan"))
        strict_meta["cv_precision"] = loocv_metrics.get("precision", float("nan"))
        strict_meta["cv_recall"] = loocv_metrics.get("recall", float("nan"))
        strict_meta["cv_f1"] = loocv_metrics.get("f1", float("nan"))
        strict_meta["cv_auc_mean"] = loocv_metrics.get("roc_auc", strict_meta.get("cv_auc_mean", float("nan")))

        meta_path = METADATA_DIR / f"aim1_non_conversion_strict_logistic_{strict_meta['version']}.json"
        with open(meta_path, "w") as f:
            json.dump(strict_meta, f, indent=2)

        strict_registry_entry = {
            "aim": strict_meta["aim"],
            "model": "strict_logistic",
            "version": strict_meta["version"],
            "timestamp": strict_meta["timestamp"],
            "n_samples": strict_meta["n_samples"],
            "train_auc": strict_meta.get("cv_auc_mean", float("nan")),
            "cv_auc_mean": strict_meta.get("cv_auc_mean", float("nan")),
            "cv_auc_std": float("nan"),
            "cv_accuracy": strict_meta.get("cv_accuracy", float("nan")),
            "cv_precision": strict_meta.get("cv_precision", float("nan")),
            "cv_recall": strict_meta.get("cv_recall", float("nan")),
            "cv_f1": strict_meta.get("cv_f1", float("nan")),
            "train_avg_precision": float("nan"),
            "data_hash": strict_meta["data_hash"],
            "params_hash": strict_meta["params_hash"],
            "model_path": strict_meta["model_path"],
        }
        update_registry_csv({"strict_logistic": strict_registry_entry}, "aim1_non_conversion_strict")
    else:
        print("  Insufficient strict-labeled data for Aim 1. Skipping strict model.")

    # ---------------------------------------------------------------
    # Phase 2 — Imputed model (sensitivity analysis, n=218)
    # ---------------------------------------------------------------
    print("\n--- Phase 2: Imputed Model (sensitivity analysis, n=218) ---")
    X_train, X_test, y_train, y_test, preprocessor, feature_cols = prepare_aim1_data(
        test_size=aim1_cfg.test_size,
        random_state=aim1_cfg.random_state,
    )

    if X_train is None or len(X_train) < 2:
        print("  Insufficient data for Aim 1 imputed model. Skipping.")
        return

    print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Features: {feature_cols}")
    print(f"  Class distribution: {y_train.value_counts().to_dict()}")

    results, trained = train_models(
        X_train, y_train, preprocessor, feature_cols,
        aim="aim1_non_conversion",
        use_smote=aim1_cfg.use_smote,
        cv_folds=aim1_cfg.cv_folds,
        random_state=aim1_cfg.random_state,
    )

    update_registry_csv(results, "aim1_non_conversion")

    for name, (pipeline, model_path) in trained.items():
        version = results[name]["version"]
        save_scaler(preprocessor, "aim1", version)
        if len(X_test) > 0:
            evaluate_model(pipeline, X_test, y_test, name, "aim1_non_conversion", version)

    champion = get_champion_model("aim1_non_conversion")
    if champion:
        print(f"\n  Champion model (imputed): {champion['model']} ({champion['version']}) "
              f"AUC={champion['cv_auc_mean']:.3f}")


def run_aim2(force: bool = False):
    print("\n" + "=" * 60)
    print("AIM 2: Identifying at-risk TB contacts")
    print("=" * 60)

    config = get_config()
    aim2_cfg = config.aim2

    X_train, X_test, y_train, y_test, preprocessor, feature_cols = prepare_aim2_data(
        test_size=aim2_cfg.test_size,
        random_state=aim2_cfg.random_state,
    )

    if X_train is None or len(X_train) < 2:
        print("  Insufficient data for Aim 2. Skipping.")
        return

    print(f"  Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"  Features: {feature_cols}")
    print(f"  Class distribution: {y_train.value_counts().to_dict()}")

    results, trained = train_models(
        X_train, y_train, preprocessor, feature_cols,
        aim="aim2_contact_risk",
        use_smote=aim2_cfg.use_smote,
        cv_folds=aim2_cfg.cv_folds,
        random_state=aim2_cfg.random_state,
    )

    update_registry_csv(results, "aim2_contact_risk")

    for name, (pipeline, model_path) in trained.items():
        version = results[name]["version"]
        save_scaler(preprocessor, "aim2", version)
        if len(X_test) > 0:
            evaluate_model(pipeline, X_test, y_test, name, "aim2_contact_risk", version)

    champion = get_champion_model("aim2_contact_risk")
    if champion:
        print(f"\n  Champion model: {champion['model']} ({champion['version']}) "
              f"AUC={champion['cv_auc_mean']:.3f}")


def run_monitoring(force: bool = False):
    config = get_config()
    mon_cfg = config.monitoring
    if not mon_cfg.enabled:
        print("\nMonitoring disabled in config. Skipping.")
        return

    print("\n" + "=" * 60)
    print("Model & Data Drift Monitoring")
    print("=" * 60)

    if mon_cfg.synthetic_demo.enabled:
        print("\n--- Generating synthetic drift data ---")
        for source, prefix in [
            ("aim1_patients_imputed.csv", "aim1"),
            ("aim2_contacts_imputed.csv", "aim2"),
            ("aim1_patients_strict.csv", "aim1_strict"),
        ]:
            try:
                generate_all_synthetic_variants(
                    source, output_prefix=prefix,
                    random_state=config.aim1.random_state,
                )
            except FileNotFoundError as e:
                print(f"  Skipping {source}: {e}")

    for aim_label, aim_key, ref_csv, target_col, prefix in [
        ("1", "aim1_non_conversion", "aim1_patients_imputed.csv", "TARGET_NON_CONVERSION_ANY", "aim1"),
        ("2", "aim2_contact_risk", "aim2_contacts_imputed.csv", "TARGET_SYMPTOM_PRESENT", "aim2"),
    ]:
        print(f"\n--- Drift check for aim {aim_label} ({aim_key}) ---")
        ref_path = Path(__file__).resolve().parents[2] / "data" / "cleaned" / ref_csv
        if not ref_path.exists():
            print(f"  Reference not found: {ref_path}. Skipping.")
            continue

        reference_df = pd.read_csv(ref_path)

        champion = get_champion_model(aim_key)
        if champion is None:
            print(f"  No trained model for {aim_key}. Skipping model drift.")
            continue

        feat_cols = champion.get("feature_cols", [])
        target_in_ref = target_col in reference_df.columns

        synthetic_csv = (
            Path(__file__).resolve().parents[2] / "data" / "synthetic" / f"{prefix}_drifted.csv"
        )
        if synthetic_csv.exists():
            current_df = pd.read_csv(synthetic_csv)
            print("  Comparing reference vs synthetic drifted data")
            available = [c for c in feat_cols if c in reference_df.columns and c in current_df.columns]
            data_result = compute_data_drift(
                reference_df, current_df,
                feature_cols=available,
                psi_bins=mon_cfg.psi_bins,
            )
            dd_status = "DRIFT DETECTED" if data_result.get("data_drift_detected") else "no drift"
            print(f"  Data drift: {dd_status} ({data_result.get('drift_ratio', 0):.2%} features drifted)")

            model_result = None
            if target_in_ref and target_col in current_df.columns:
                try:
                    pipeline = joblib.load(champion["model_path"])
                    ref_data = reference_df[feat_cols]
                    ref_labels = reference_df[target_col].astype(int)
                    cur_data = current_df[feat_cols]
                    cur_labels = current_df[target_col].astype(int)
                    model_result = compute_model_drift(
                        pipeline, ref_data, ref_labels, cur_data, cur_labels,
                        feature_cols=feat_cols,
                        auc_drop_threshold=mon_cfg.model_drift_thresholds.auc_drop,
                        prediction_psi_threshold=mon_cfg.model_drift_thresholds.prediction_psi,
                    )
                    md_status = "DRIFT DETECTED" if model_result.get("model_drift_detected") else "no drift"
                    auc_drop = model_result.get("auc_drop")
                    drop_str = f", AUC drop: {auc_drop:.3f}" if auc_drop is not None else ""
                    print(f"  Model drift: {md_status}{drop_str}")
                except Exception as e:
                    print(f"  Model drift check failed: {e}")
            else:
                print("  Model drift skipped (missing target column in synthetic data)")

            report = generate_drift_report(
                data_drift_result=data_result,
                model_drift_result=model_result,
            )
            save_drift_report(report)
            log_to_mlflow(report, run_name=f"pipeline_drift_aim{aim_label}")
            print("  Drift report saved")

            if should_retrain(report, log_only=mon_cfg.log_only) and mon_cfg.auto_retrain:
                print(f"  Drift detected — triggering retrain for aim {aim_label}")
                if aim_key == "aim1_non_conversion":
                    run_aim1(force=True)
                elif aim_key == "aim2_contact_risk":
                    run_aim2(force=True)
                print(f"  Retrain complete for aim {aim_label}")
        else:
            print(f"  No synthetic data found at {synthetic_csv}. Run with synthetic_demo.enabled=true or generate manually.")
    print("\nMonitoring complete.")


def main():
    parser = argparse.ArgumentParser(description="TB Recovery - Model Training Pipeline")
    parser.add_argument("--aim", choices=["1", "2", "all"], default="all",
                        help="Which aim to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if data unchanged")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip data change detection")
    parser.add_argument("--monitoring", action="store_true",
                        help="Run drift monitoring after training")
    args = parser.parse_args()

    print(f"TB Recovery Pipeline - {datetime.now().isoformat()}")
    print(f"Force: {args.force}, Skip data check: {args.skip_data_check}")

    print("\nStep 1: Cleaning and saving datasets...")
    save_clean_datasets()

    if not args.skip_data_check and not args.force:
        current_hash = _compute_data_hash()
        if not _is_data_changed(current_hash):
            print("Data unchanged. Use --force to retrain anyway.")
            return
        _save_data_hash(current_hash)

    if args.aim in ("1", "all"):
        run_aim1(force=args.force)

    if args.aim in ("2", "all"):
        run_aim2(force=args.force)

    if args.monitoring:
        run_monitoring(force=args.force)

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
