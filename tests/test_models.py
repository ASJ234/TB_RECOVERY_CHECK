import pytest
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
        from sklearn.pipeline import Pipeline
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
        from sklearn.pipeline import Pipeline
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


class TestEvaluate:
    def test_evaluate_model(self):
        from src.models.evaluate import evaluate_model
        from sklearn.pipeline import Pipeline
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
