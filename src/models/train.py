import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import joblib
import json
import hashlib
from pathlib import Path
from datetime import datetime

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
REGISTRY_DIR = MODELS_DIR / "registry"
METADATA_DIR = MODELS_DIR / "metadata"


def _get_version(aim: str) -> str:
    aim_dir = REGISTRY_DIR / aim
    aim_dir.mkdir(parents=True, exist_ok=True)
    existing = list(aim_dir.glob("*.pkl"))
    return f"v{len(existing) + 1}"


def _compute_data_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df).values).hexdigest()[:16]


def _compute_params_hash(params: dict) -> str:
    serializable = {}
    for k, v in params.items():
        try:
            json.dumps({k: v})
            serializable[k] = v
        except (TypeError, ValueError):
            serializable[k] = str(type(v).__name__)
    return hashlib.sha256(json.dumps(serializable, sort_keys=True).encode()).hexdigest()[:16]


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    preprocessor,
    feature_cols: list,
    aim: str,
    use_smote: bool = True,
    cv_folds: int = 5,
    random_state: int = 42,
):
    n_samples = len(X_train)
    if n_samples < cv_folds:
        cv_folds = n_samples

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000, C=0.1, l1_ratio=0, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=3, class_weight="balanced",
            random_state=random_state
        ),
        "xgboost": xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            scale_pos_weight=(y_train.value_counts().get(0, 1) /
                              max(y_train.value_counts().get(1, 1), 1)),
            eval_metric="logloss",
            random_state=random_state,
        ),
    }

    results = {}
    trained_models = {}

    for name, estimator in models.items():
        min_class_size = y_train.value_counts().min() if len(y_train) > 0 else 0
        smote_neighbors = min(5, max(1, min_class_size - 1))
        use_smote_here = use_smote and n_samples >= 6 and min_class_size >= 2

        if use_smote_here:
            pipeline = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=random_state, k_neighbors=smote_neighbors)),
                ("classifier", estimator),
            ])
        else:
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ])

        pipeline.fit(X_train, y_train)

        try:
            if n_samples >= cv_folds and n_samples >= 2:
                if n_samples <= 20:
                    cv = LeaveOneOut()
                else:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Only one class is present")
                    cv_scores = cross_val_score(
                        pipeline, X_train, y_train, cv=cv, scoring="roc_auc"
                    )
                auc_mean = float(np.mean(cv_scores))
                auc_std = float(np.std(cv_scores))
            else:
                auc_mean = float("nan")
                auc_std = float("nan")
        except Exception:
            auc_mean = float("nan")
            auc_std = float("nan")

        try:
            y_pred_proba = pipeline.predict_proba(X_train)[:, 1]
            if len(np.unique(y_train)) > 1:
                train_auc = roc_auc_score(y_train, y_pred_proba)
            else:
                train_auc = float("nan")
            train_ap = average_precision_score(y_train, y_pred_proba)
        except Exception:
            train_auc = float("nan")
            train_ap = float("nan")

        version = _get_version(aim)
        model_path = REGISTRY_DIR / aim / f"{version}_{name}.pkl"
        joblib.dump(pipeline, model_path)

        metadata = {
            "model": name,
            "version": version,
            "aim": aim,
            "timestamp": datetime.now().isoformat(),
            "n_samples": n_samples,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
            "train_auc": train_auc,
            "train_avg_precision": train_ap,
            "cv_auc_mean": auc_mean,
            "cv_auc_std": auc_std,
            "params_hash": _compute_params_hash(pipeline.get_params()),
            "data_hash": _compute_data_hash(X_train),
            "cv_folds": cv_folds if n_samples >= cv_folds else n_samples,
            "use_smote": use_smote,
            "model_path": str(model_path),
        }

        METADATA_DIR.mkdir(parents=True, exist_ok=True)
        meta_path = METADATA_DIR / f"{aim}_{name}_{version}.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        results[name] = metadata
        trained_models[name] = (pipeline, model_path)
        print(f"  {name} ({version}): CV AUC={auc_mean:.3f} ± {auc_std:.3f}, "
              f"Train AUC={train_auc:.3f}")

    return results, trained_models


def train_aim1_strict(
    X: pd.DataFrame,
    y: pd.Series,
    preprocessor,
    feature_cols: list,
    sample_ids: list = None,
    discordant: list = None,
    C: float = 1.0,
    random_state: int = 42,
):
    n_samples = len(X)
    aim = "aim1_non_conversion"
    version = _get_version(f"{aim}_strict")

    from sklearn.model_selection import LeaveOneOut
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    loo = LeaveOneOut()

    per_fold = []
    y_true_all = []
    y_pred_all = []
    y_proba_all = []

    for train_idx, test_idx in loo.split(X):
        X_train_fold = X.iloc[train_idx]
        y_train_fold = y.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_test_fold = y.iloc[test_idx]

        fold_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                C=C, l1_ratio=0, class_weight="balanced",
                max_iter=1000, random_state=random_state,
            )),
        ])
        fold_pipeline.fit(X_train_fold, y_train_fold)

        y_pred = fold_pipeline.predict(X_test_fold)[0]
        y_proba = fold_pipeline.predict_proba(X_test_fold)[0][1]

        sample_id = sample_ids[test_idx[0]] if sample_ids else str(test_idx[0])
        is_discordant = bool(discordant[test_idx[0]]) if discordant else False

        per_fold.append({
            "sample_id": sample_id,
            "fold": len(per_fold) + 1,
            "true_label": int(y_test_fold.iloc[0]),
            "predicted_label": int(y_pred),
            "probability": float(y_proba),
            "correct": int(y_test_fold.iloc[0]) == int(y_pred),
            "discordant": is_discordant,
        })
        y_true_all.append(int(y_test_fold.iloc[0]))
        y_pred_all.append(int(y_pred))
        y_proba_all.append(float(y_proba))

    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            C=C, l1_ratio=0, class_weight="balanced",
            max_iter=1000, random_state=random_state,
        )),
    ])
    final_pipeline.fit(X, y)

    model_path = REGISTRY_DIR / aim / f"{version}_strict_logistic.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)

    cv_auc = float("nan")
    if len(np.unique(y_true_all)) > 1:
        try:
            cv_auc = float(roc_auc_score(y_true_all, y_proba_all))
        except Exception:
            cv_auc = float("nan")

    metadata = {
        "model": "strict_logistic",
        "version": version,
        "aim": f"{aim}_strict",
        "timestamp": datetime.now().isoformat(),
        "n_samples": n_samples,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "C": C,
        "penalty": "l2",
        "class_weight": "balanced",
        "cv_method": "LOOCV",
        "use_smote": False,
        "cv_roc_auc": cv_auc,
        "params_hash": _compute_params_hash(final_pipeline.get_params()),
        "data_hash": _compute_data_hash(X),
        "model_path": str(model_path),
    }

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = METADATA_DIR / f"{aim}_strict_logistic_{version}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  strict_logistic ({version}): LOOCV AUC={cv_auc:.3f}")

    return final_pipeline, per_fold, metadata


def get_champion_model(aim: str, metric: str = "cv_auc_mean"):
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    best = None
    best_score = -1.0
    for meta_path in METADATA_DIR.glob(f"{aim}_*.json"):
        with open(meta_path) as f:
            meta = json.load(f)
        score = meta.get(metric)
        if score is not None and not np.isnan(score) and score > best_score:
            best_score = score
            best = meta
    return best


def load_model(aim: str, name: str, version: str):
    path = REGISTRY_DIR / aim / f"{version}_{name}.pkl"
    return joblib.load(path)


def update_registry_csv(results: dict, aim: str):
    registry_path = REGISTRY_DIR / "model_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, meta in results.items():
        rows.append({
            "aim": aim,
            "model": name,
            "version": meta["version"],
            "timestamp": meta["timestamp"],
            "n_samples": meta["n_samples"],
            "train_auc": meta["train_auc"],
            "cv_auc_mean": meta["cv_auc_mean"],
            "cv_auc_std": meta["cv_auc_std"],
            "train_avg_precision": meta["train_avg_precision"],
            "data_hash": meta["data_hash"],
            "params_hash": meta["params_hash"],
            "model_path": meta["model_path"],
        })

    df_new = pd.DataFrame(rows)
    if registry_path.exists():
        df_old = pd.read_csv(registry_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(registry_path, index=False)
    print(f"  Registry updated: {len(rows)} new entries at {registry_path}")
