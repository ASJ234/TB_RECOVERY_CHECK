import pytest
from pathlib import Path
import json


class TestPipelineConfig:
    def test_config_defaults(self):
        from src.pipeline.config import PipelineConfig
        config = PipelineConfig()
        assert config.aim1.target_column == "TARGET_NON_CONVERSION_ANY"
        assert config.aim2.target_column == "TARGET_SYMPTOM_PRESENT"
        assert "xgboost" in config.aim1.models
        assert config.skip_if_data_unchanged is True

    def test_config_save_and_load(self, tmp_path):
        from src.pipeline.config import PipelineConfig
        config_path = tmp_path / "test_config.json"

        config = PipelineConfig()
        config.aim1.cv_folds = 3
        config.aim2.use_smote = False
        config.save(path=config_path)

        loaded = PipelineConfig.load(path=config_path)
        assert loaded.aim1.cv_folds == 3
        assert loaded.aim2.use_smote is False


class TestTrainingPipeline:
    def test_pipeline_imports(self):
        from src.pipeline.training_pipeline import run_aim1, run_aim2, main
        assert callable(run_aim1)
        assert callable(run_aim2)
        assert callable(main)

    def test_compute_data_hash(self):
        from src.pipeline.training_pipeline import _compute_data_hash
        h = _compute_data_hash()
        assert isinstance(h, str)
        assert len(h) == 16


class TestAPIImports:
    def test_api_imports(self):
        from src.api.main import app
        assert app.title == "TB Recovery Prediction API"

    def test_schemas_import(self):
        from src.api.schemas import (
            Aim1PredictionRequest,
            Aim2PredictionRequest,
            PredictionResponse,
        )
        req = Aim1PredictionRequest()
        assert req is not None
