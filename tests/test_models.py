import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline


class TestTrain:
    def test_train_models_returns_results(self):
        from src.models.train import train_models
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        n = 20
        X_train = pd.DataFrame({
            "AGE (YEARS)": np.random.uniform(18, 80, n),
            "SEX": np.random.choice(["M", "F"], n),
            "COUGH": np.random.choice(["YES", "NO"], n),
        })
        y_train = pd.Series(np.random.choice([0, 1], n))

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), ["AGE (YEARS)"]),
        ], remainder="drop")

        results, trained = train_models(
            X_train, y_train, preprocessor,
            feature_cols=list(X_train.columns),
            aim="test_aim",
            use_smote=False,
        )

        assert "xgboost" in results
        assert "logistic_regression" in results
        assert "train_auc" in results["xgboost"]
        assert "model_path" in results["xgboost"]

    def test_train_models_with_smote(self):
        from src.models.train import train_models
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        n = 30
        X_train = pd.DataFrame({
            "AGE (YEARS)": np.random.uniform(18, 80, n),
        })
        y_train = pd.Series([0] * 25 + [1] * 5)

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), ["AGE (YEARS)"]),
        ], remainder="drop")

        results, trained = train_models(
            X_train, y_train, preprocessor,
            feature_cols=list(X_train.columns),
            aim="test_aim_smote",
            use_smote=True,
        )

        assert len(trained) > 0
        for name in trained:
            assert trained[name][0] is not None


class TestPredict:
    def test_predict_single(self):
        from src.models.predict import predict_single
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        n = 20
        X = pd.DataFrame({
            "age": np.random.uniform(18, 80, n),
            "sex": np.random.choice([0, 1], n),
        })
        y = np.random.choice([0, 1], n)

        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", StandardScaler(), ["age"]),
            ], remainder="drop")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        pipeline.fit(X, y)

        features = {"age": 35.0, "sex": 1}
        result = predict_single(pipeline, features, list(X.columns))
        assert "prediction" in result
        assert "probability" in result
        assert "confidence" in result

    def test_predict_batch(self):
        from src.models.predict import predict_batch
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        X = pd.DataFrame({
            "age": np.random.uniform(18, 80, 10),
        })
        y = np.random.choice([0, 1], 10)

        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", StandardScaler(), ["age"]),
            ], remainder="drop")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        pipeline.fit(X, y)

        result = predict_batch(pipeline, X, list(X.columns))
        assert "prediction" in result.columns
        assert "probability_positive" in result.columns
        assert len(result) == len(X)


class TestTrainAim1Strict:
    def test_train_aim1_strict_returns_per_fold(self):
        from src.models.train import train_aim1_strict
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        n = 11
        X = pd.DataFrame({
            "AGE (YEARS)": np.random.uniform(18, 80, n),
            "BMI": np.random.uniform(15, 35, n),
            "SEX": np.random.choice(["M", "F"], n),
            "baseline_symptom_count": np.random.randint(0, 7, n),
        })
        y = pd.Series(np.random.choice([0, 1], n))

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), ["AGE (YEARS)", "BMI", "baseline_symptom_count"]),
        ], remainder="drop")

        pipeline, per_fold, metadata = train_aim1_strict(
            X, y, preprocessor,
            feature_cols=list(X.columns),
            sample_ids=[f"P{i:03d}" for i in range(n)],
            C=1.0,
        )

        assert len(per_fold) == n
        assert callable(pipeline.predict)
        assert metadata["n_samples"] == n
        assert metadata["model"] == "strict_logistic"
        assert metadata["use_smote"] is False
        assert metadata["cv_method"] == "LOOCV"
        for entry in per_fold:
            assert "sample_id" in entry
            assert "true_label" in entry
            assert "predicted_label" in entry
            assert "probability" in entry
            assert "correct" in entry

    def test_train_aim1_strict_no_smote(self):
        from src.models.train import train_aim1_strict
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        n = 11
        X = pd.DataFrame({"AGE (YEARS)": np.random.uniform(18, 80, n)})
        y = pd.Series([0] * 5 + [1] * 6)

        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), ["AGE (YEARS)"]),
        ], remainder="drop")

        _, _, metadata = train_aim1_strict(X, y, preprocessor, ["AGE (YEARS)"], C=1.0)
        assert metadata["use_smote"] is False


class TestEvaluate:
    def test_evaluate_loocv(self):
        from src.models.evaluate import evaluate_loocv

        per_fold = [
            {"sample_id": "P001", "true_label": 1, "predicted_label": 1, "probability": 0.8, "correct": True, "discordant": False},
            {"sample_id": "P002", "true_label": 0, "predicted_label": 0, "probability": 0.3, "correct": True, "discordant": False},
            {"sample_id": "P003", "true_label": 1, "predicted_label": 0, "probability": 0.4, "correct": False, "discordant": False},
            {"sample_id": "P004", "true_label": 0, "predicted_label": 1, "probability": 0.7, "correct": False, "discordant": True},
        ]

        metrics = evaluate_loocv(per_fold, model_name="test_lr", aim="test_aim", version="v1")
        assert metrics["accuracy"] == 0.5
        assert metrics["n_samples"] == 4
        assert metrics["cv_method"] == "LOOCV"
        assert "confusion_matrix" in metrics

    def test_evaluate_model(self):
        from src.models.evaluate import evaluate_model
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.compose import ColumnTransformer

        np.random.seed(42)
        X = pd.DataFrame({
            "age": np.random.uniform(18, 80, 10),
        })
        y = pd.Series(np.random.choice([0, 1], 10))

        pipeline = Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", StandardScaler(), ["age"]),
            ], remainder="drop")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ])
        pipeline.fit(X, y)

        metrics = evaluate_model(pipeline, X, y, "test_lr", "test_aim", "v1")
        assert "accuracy" in metrics
        assert "roc_auc" in metrics
        assert "f1" in metrics
