import numpy as np
import pandas as pd
import pytest
from pathlib import Path


class TestStatisticalTests:
    def test_ks_test_identical(self):
        from src.monitoring.statistical_tests import ks_test
        a = np.random.default_rng(42).normal(0, 1, 1000)
        b = a.copy()
        result = ks_test(a, b, feature_name="test")
        assert result["feature"] == "test"
        assert result["test"] == "ks"
        assert not result["drift_detected"]
        assert result["p_value"] > 0.05

    def test_ks_test_different(self):
        from src.monitoring.statistical_tests import ks_test
        a = np.random.default_rng(42).normal(0, 1, 1000)
        b = np.random.default_rng(43).normal(5, 1, 1000)
        result = ks_test(a, b, feature_name="test")
        assert result["drift_detected"]
        assert result["p_value"] < 0.05

    def test_chi_square_identical(self):
        from src.monitoring.statistical_tests import chi_square_test
        a = np.array(["A", "B", "A", "B", "A"] * 100)
        b = a.copy()
        result = chi_square_test(a, b, feature_name="cat")
        assert not result["drift_detected"]

    def test_chi_square_different(self):
        from src.monitoring.statistical_tests import chi_square_test
        a = np.array(["A"] * 500 + ["B"] * 500)
        b = np.array(["A"] * 100 + ["B"] * 900)
        result = chi_square_test(a, b, feature_name="cat")
        assert result["statistic"] > 0

    def test_psi_identical(self):
        from src.monitoring.statistical_tests import psi
        a = np.random.default_rng(42).normal(0, 1, 10000)
        b = np.random.default_rng(42).normal(0, 1, 10000)
        np.random.default_rng(42).shuffle(b)
        result = psi(a, b, num_bins=10, feature_name="num")
        assert result["statistic"] < 0.1
        assert not result["drift_detected"]

    def test_psi_different(self):
        from src.monitoring.statistical_tests import psi
        a = np.random.default_rng(42).normal(0, 1, 10000)
        b = np.random.default_rng(43).normal(3, 1, 10000)
        result = psi(a, b, num_bins=10, feature_name="num")
        assert result["statistic"] > 0.1
        assert result["drift_detected"]

    def test_categorical_psi_identical(self):
        from src.monitoring.statistical_tests import categorical_psi
        a = np.array(["A", "B", "C"] * 1000)
        b = a.copy()
        result = categorical_psi(a, b, feature_name="cat")
        assert result["statistic"] < 0.01
        assert not result["drift_detected"]

    def test_categorical_psi_different(self):
        from src.monitoring.statistical_tests import categorical_psi
        a = np.array(["A"] * 900 + ["B"] * 100)
        b = np.array(["A"] * 100 + ["B"] * 900)
        result = categorical_psi(a, b, feature_name="cat")
        assert result["statistic"] > 0.1
        assert result["drift_detected"]

    def test_js_divergence_identical(self):
        from src.monitoring.statistical_tests import js_divergence
        a = np.random.default_rng(42).normal(0, 1, 10000)
        b = a.copy()
        result = js_divergence(a, b, feature_name="num")
        assert result["statistic"] < 0.1

    def test_js_divergence_different(self):
        from src.monitoring.statistical_tests import js_divergence
        rng = np.random.default_rng(42)
        a = rng.normal(0, 1, 10000)
        b = rng.normal(5, 1, 10000)
        result = js_divergence(a, b, feature_name="num")
        assert result["statistic"] > 0.1


class TestDataDrift:
    def test_compute_data_drift_identical(self):
        from src.monitoring.data_drift import compute_data_drift
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "age": rng.normal(40, 10, 500),
            "bmi": rng.normal(25, 5, 500),
            "sex": rng.choice(["M", "F"], 500),
        })
        cur = ref.copy()
        result = compute_data_drift(ref, cur)
        assert not result["data_drift_detected"]
        assert result["drift_ratio"] == 0.0

    def test_compute_data_drift_different(self):
        from src.monitoring.data_drift import compute_data_drift
        rng = np.random.default_rng(42)
        ref = pd.DataFrame({
            "age": rng.normal(40, 10, 500),
            "bmi": rng.normal(25, 5, 500),
            "sex": rng.choice(["M", "F"], 500),
        })
        cur = pd.DataFrame({
            "age": rng.normal(60, 10, 500),
            "bmi": rng.normal(30, 5, 500),
            "sex": rng.choice(["M", "M"], 500),
        })
        result = compute_data_drift(ref, cur)
        assert result["data_drift_detected"]
        assert result["drift_ratio"] > 0

    def test_compute_data_drift_subset_features(self):
        from src.monitoring.data_drift import compute_data_drift
        ref = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6], "c": [7, 8, 9]})
        cur = pd.DataFrame({"a": [10, 20, 30], "b": [4, 5, 6]})
        result = compute_data_drift(ref, cur, feature_cols=["a", "b"])
        assert result["n_features"] == 2


class TestModelDrift:
    def test_compute_model_drift_identical(self):
        from src.monitoring.model_drift import compute_model_drift
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        rng = np.random.default_rng(42)
        X = pd.DataFrame({
            "f1": rng.normal(0, 1, 200),
            "f2": rng.normal(0, 1, 200),
        })
        y = pd.Series((X["f1"] + X["f2"] > 0).astype(int))
        model = Pipeline([
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        model.fit(X, y)

        result = compute_model_drift(
            model, X, y, X, y,
            feature_cols=["f1", "f2"],
        )
        assert not result["model_drift_detected"]
        assert result["auc_drop"] is not None

    def test_compute_model_drift_degraded(self):
        from src.monitoring.model_drift import compute_model_drift
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        rng = np.random.default_rng(42)
        X_ref = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "f2": rng.normal(0, 1, 500),
        })
        y_ref = pd.Series((X_ref["f1"] + X_ref["f2"] > 0).astype(int))

        model = Pipeline([
            ("clf", LogisticRegression(max_iter=2000, random_state=42)),
        ])
        model.fit(X_ref, y_ref)

        X_cur = pd.DataFrame({
            "f1": rng.normal(0, 1, 500),
            "f2": rng.normal(0, 1, 500),
        })
        y_cur = pd.Series((X_cur["f1"] + X_cur["f2"] < 0).astype(int))

        result = compute_model_drift(
            model, X_ref, y_ref, X_cur, y_cur,
            feature_cols=["f1", "f2"],
            auc_drop_threshold=0.02,
        )
        assert result["performance_drift"]


class TestSyntheticDrift:
    def test_generate_gaussian_drift(self):
        from src.monitoring.synthetic_drift import generate_gaussian_drift
        df = pd.DataFrame({"age": [20, 30, 40], "bmi": [18, 24, 30]})
        result = generate_gaussian_drift(df, noise_std=0.1, random_state=42)
        assert result.shape == df.shape
        assert not result.equals(df)
        assert result["age"].dtype == float

    def test_generate_category_swap_drift(self):
        from src.monitoring.synthetic_drift import generate_category_swap_drift
        df = pd.DataFrame({"sex": ["M", "F", "M", "F"] * 25, "age": [1, 2, 3, 4] * 25})
        result = generate_category_swap_drift(df, swap_pct=1.0, random_state=42)
        assert result.shape == df.shape

    def test_generate_combined_drift(self):
        from src.monitoring.synthetic_drift import generate_combined_drift
        df = pd.DataFrame({
            "age": [20, 30, 40],
            "bmi": [18, 24, 30],
            "sex": ["M", "F", "M"],
        })
        result = generate_combined_drift(df, noise_std=0.5, swap_pct=0.3, random_state=42)
        assert result.shape == df.shape


class TestReporting:
    def test_generate_drift_report(self):
        from src.monitoring.reporting import generate_drift_report
        report = generate_drift_report(
            data_drift_result={"data_drift_detected": True, "drift_ratio": 0.5},
            model_drift_result={"model_drift_detected": False},
        )
        assert report["drift_detected"]
        assert "timestamp" in report
        assert report["data_drift"]["data_drift_detected"]

    def test_should_retrain_log_only(self):
        from src.monitoring.reporting import should_retrain
        report = {"drift_detected": True}
        assert not should_retrain(report, log_only=True)
        assert should_retrain(report, log_only=False)

    def test_save_drift_report(self, tmp_path):
        from src.monitoring.reporting import save_drift_report
        report = {"drift_detected": False, "timestamp": "2024-01-01"}
        path = save_drift_report(report, path=tmp_path / "test_report.json")
        assert path.exists()
        import json
        with open(path) as f:
            loaded = json.load(f)
        assert not loaded["drift_detected"]


class TestEndToEnd:
    def test_data_drift_on_real_data(self):
        from src.monitoring.data_drift import compute_data_drift
        data_dir = Path(__file__).resolve().parents[1] / "data" / "cleaned"
        aim1_path = data_dir / "aim1_patients_imputed.csv"
        if not aim1_path.exists():
            pytest.skip("Cleaned data not found; run pipeline first")
        df = pd.read_csv(aim1_path)
        feature_cols = [
            "SEX", "AGE (YEARS)", "BMI", "TEMPERATURE_CELCIUS",
            "COUGH", "FEVER", "HIV_STATUS",
        ]
        available = [c for c in feature_cols if c in df.columns]
        result = compute_data_drift(df, df, feature_cols=available)
        assert not result["data_drift_detected"]
        assert result["drift_ratio"] == 0.0
