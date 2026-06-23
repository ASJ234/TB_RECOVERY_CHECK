import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
CONFIG_PATH = CONFIG_DIR / "pipeline_config.json"


@dataclass
class Aim1Config:
    target_column: str = "TARGET_NON_CONVERSION_ANY"
    test_size: float = 0.2
    cv_folds: int = 5
    use_smote: bool = True
    random_state: int = 42
    models: list = field(default_factory=lambda: [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ])


@dataclass
class Aim2Config:
    target_column: str = "TARGET_SYMPTOM_PRESENT"
    test_size: float = 0.2
    cv_folds: int = 5
    use_smote: bool = True
    random_state: int = 42
    models: list = field(default_factory=lambda: [
        "logistic_regression",
        "random_forest",
        "xgboost",
    ])


@dataclass
class PipelineConfig:
    aim1: Aim1Config = field(default_factory=Aim1Config)
    aim2: Aim2Config = field(default_factory=Aim2Config)
    skip_if_data_unchanged: bool = True
    save_plots: bool = True
    log_level: str = "INFO"

    def save(self, path: Optional[Path] = None):
        path = path or CONFIG_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None):
        path = path or CONFIG_PATH
        if not path.exists():
            return cls()
        with open(path) as f:
            data = json.load(f)
        aim1 = Aim1Config(**data.get("aim1", {}))
        aim2 = Aim2Config(**data.get("aim2", {}))
        return cls(
            aim1=aim1,
            aim2=aim2,
            skip_if_data_unchanged=data.get("skip_if_data_unchanged", True),
            save_plots=data.get("save_plots", True),
            log_level=data.get("log_level", "INFO"),
        )


def get_config() -> PipelineConfig:
    return PipelineConfig.load()
