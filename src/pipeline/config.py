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
    strict_features: list = field(default_factory=lambda: [
        "SEX", "AGE (YEARS)", "BMI", "baseline_symptom_count",
    ])
    strict_target: str = "TARGET_STRICT_M2"
    strict_C: float = 1.0


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
class DataDriftThresholds:
    psi: float = 0.1
    ks_p_value: float = 0.05
    chi_square_p_value: float = 0.05
    drift_ratio: float = 0.3


@dataclass
class ModelDriftThresholds:
    auc_drop: float = 0.05
    prediction_psi: float = 0.1


@dataclass
class SyntheticDemoConfig:
    enabled: bool = True
    noise_std: float = 0.5
    category_swap_pct: float = 0.3


@dataclass
class MonitoringConfig:
    enabled: bool = True
    log_only: bool = True
    auto_retrain: bool = False
    psi_bins: int = 10
    data_drift_thresholds: DataDriftThresholds = field(default_factory=DataDriftThresholds)
    model_drift_thresholds: ModelDriftThresholds = field(default_factory=ModelDriftThresholds)
    synthetic_demo: SyntheticDemoConfig = field(default_factory=SyntheticDemoConfig)


@dataclass
class PipelineConfig:
    aim1: Aim1Config = field(default_factory=Aim1Config)
    aim2: Aim2Config = field(default_factory=Aim2Config)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
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
        mon_data = data.get("monitoring", {})
        dd_thresh = DataDriftThresholds(**mon_data.get("data_drift_thresholds", {}))
        md_thresh = ModelDriftThresholds(**mon_data.get("model_drift_thresholds", {}))
        syn_demo = SyntheticDemoConfig(**mon_data.get("synthetic_demo", {}))
        monitoring = MonitoringConfig(
            **{k: v for k, v in mon_data.items() if k not in ("data_drift_thresholds", "model_drift_thresholds", "synthetic_demo")},
            data_drift_thresholds=dd_thresh,
            model_drift_thresholds=md_thresh,
            synthetic_demo=syn_demo,
        )
        return cls(
            aim1=aim1,
            aim2=aim2,
            monitoring=monitoring,
            skip_if_data_unchanged=data.get("skip_if_data_unchanged", True),
            save_plots=data.get("save_plots", True),
            log_level=data.get("log_level", "INFO"),
        )


def get_config() -> PipelineConfig:
    return PipelineConfig.load()
