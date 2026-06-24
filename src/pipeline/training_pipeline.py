#!/usr/bin/env python3
import sys
import argparse
import hashlib
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

        evaluate_loocv(
            per_fold, model_name="strict_logistic",
            aim="aim1_non_conversion", version=strict_meta["version"],
        )

        save_scaler(preprocessor_strict, "aim1_strict", strict_meta["version"])

        strict_registry_entry = {
            "aim": strict_meta["aim"],
            "model": "strict_logistic",
            "version": strict_meta["version"],
            "timestamp": strict_meta["timestamp"],
            "n_samples": strict_meta["n_samples"],
            "train_auc": strict_meta.get("cv_auc_mean", float("nan")),
            "cv_auc_mean": strict_meta.get("cv_auc_mean", float("nan")),
            "cv_auc_std": float("nan"),
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


def main():
    parser = argparse.ArgumentParser(description="TB Recovery - Model Training Pipeline")
    parser.add_argument("--aim", choices=["1", "2", "all"], default="all",
                        help="Which aim to run (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Force retraining even if data unchanged")
    parser.add_argument("--skip-data-check", action="store_true",
                        help="Skip data change detection")
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

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
