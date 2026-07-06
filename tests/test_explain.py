import pytest
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from fastapi.testclient import TestClient

from src.explain.shap_explainer import (
    create_explainer,
    compute_global_shap,
    compute_instance_shap,
)
from src.explain.api_explain import map_shap_to_original_features
from src.api.main import app
import scripts.generate_xai_reports as report_generator

@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n = 20
    X = pd.DataFrame({
        "age": np.random.uniform(18, 80, n),
        "sex": np.random.choice(["M", "F"], n),
        "fever": np.random.choice(["YES", "NO"], n),
    })
    y = pd.Series(np.random.choice([0, 1], n))
    feature_cols = ["age", "sex", "fever"]
    return X, y, feature_cols

@pytest.fixture
def fitted_pipeline(dummy_data):
    X, y, feature_cols = dummy_data
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), ["age"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["sex", "fever"]),
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42))
    ])
    pipeline.fit(X, y)
    return pipeline

def test_create_explainer_and_shap(dummy_data, fitted_pipeline):
    X, y, feature_cols = dummy_data
    
    # Test linear explainer creation
    explainer, preprocessor = create_explainer(fitted_pipeline, X, "logistic_regression")
    assert explainer is not None
    assert preprocessor is not None
    
    # Test global SHAP computation
    global_res = compute_global_shap(explainer, X, feature_cols, preprocessor, max_samples=10)
    assert "mean_abs_shap" in global_res
    assert len(global_res["mean_abs_shap"]) == len(global_res["feature_names"])
    
    # Test instance SHAP computation
    row = X.iloc[[0]]
    instance_res = compute_instance_shap(explainer, row, feature_cols, preprocessor)
    assert "shap_values" in instance_res
    assert len(instance_res["shap_values"]) == len(instance_res["feature_names"])
    
    # Test mapping back to original features
    mapped_shap, mapped_names = map_shap_to_original_features(
        instance_res["shap_values"],
        instance_res["feature_names"],
        feature_cols,
        preprocessor
    )
    assert isinstance(mapped_shap, dict)
    assert len(mapped_shap) <= len(feature_cols)

def test_api_explain_endpoints():
    client = TestClient(app)
    
    # Clear cache first
    client.post("/cache/clear")
    
    # Check that explanation endpoints return either success or 404/503 (since models might not be trained yet)
    response = client.get("/explain/aim1/global")
    assert response.status_code in (200, 404, 503)
    
    response = client.get("/explain/aim2/global")
    assert response.status_code in (200, 404, 503)


def test_get_champion_models_prefers_champion_family_latest_version(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    metadata_dir = tmp_path / "metadata"
    for aim in ["aim1_non_conversion", "aim2_contact_risk"]:
        (registry_dir / aim).mkdir(parents=True, exist_ok=True)

    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Create registry files with multiple model families and versions.
    (registry_dir / "aim1_non_conversion" / "v7_logistic_regression.pkl").touch()
    (registry_dir / "aim1_non_conversion" / "v2_xgboost.pkl").touch()
    (registry_dir / "aim1_non_conversion" / "v3_xgboost.pkl").touch()
    (registry_dir / "aim2_contact_risk" / "v8_random_forest.pkl").touch()
    (registry_dir / "aim2_contact_risk" / "v2_xgboost.pkl").touch()
    (registry_dir / "aim2_contact_risk" / "v3_xgboost.pkl").touch()

    # Metadata shows xgboost is the champion family for both aims, and the latest xgboost version should win.
    (metadata_dir / "aim1_non_conversion_xgboost_v2.json").write_text(
        '{"model": "xgboost", "version": "v2", "aim": "aim1_non_conversion", "cv_auc_mean": 0.91, "model_path": "mock"}'
    )
    (metadata_dir / "aim1_non_conversion_xgboost_v3.json").write_text(
        '{"model": "xgboost", "version": "v3", "aim": "aim1_non_conversion", "cv_auc_mean": 0.90, "model_path": "mock"}'
    )
    (metadata_dir / "aim1_non_conversion_logistic_regression_v7.json").write_text(
        '{"model": "logistic_regression", "version": "v7", "aim": "aim1_non_conversion", "cv_auc_mean": 0.80, "model_path": "mock"}'
    )
    (metadata_dir / "aim2_contact_risk_xgboost_v2.json").write_text(
        '{"model": "xgboost", "version": "v2", "aim": "aim2_contact_risk", "cv_auc_mean": 0.89, "model_path": "mock"}'
    )
    (metadata_dir / "aim2_contact_risk_xgboost_v3.json").write_text(
        '{"model": "xgboost", "version": "v3", "aim": "aim2_contact_risk", "cv_auc_mean": 0.88, "model_path": "mock"}'
    )
    (metadata_dir / "aim2_contact_risk_random_forest_v8.json").write_text(
        '{"model": "random_forest", "version": "v8", "aim": "aim2_contact_risk", "cv_auc_mean": 0.75, "model_path": "mock"}'
    )

    monkeypatch.setattr(report_generator, "REGISTRY_DIR", registry_dir)
    monkeypatch.setattr(report_generator, "METADATA_DIR", metadata_dir)

    champions = report_generator.get_champion_models()

    assert champions == [
        {"aim": "aim1_non_conversion", "model_name": "xgboost", "version": "v3"},
        {"aim": "aim2_contact_risk", "model_name": "xgboost", "version": "v3"},
    ]
