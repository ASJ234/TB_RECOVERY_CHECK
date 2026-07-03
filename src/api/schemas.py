from pydantic import BaseModel, Field
from typing import Optional, List


class Aim1PredictionRequest(BaseModel):
    SEX: Optional[str] = Field(None, description="M or F")
    AGE_YEARS: Optional[float] = Field(None, alias="AGE (YEARS)")
    TEMPERATURE_CELCIUS: Optional[float] = None
    COUGH: Optional[str] = None
    FEVER: Optional[str] = None
    WEIGHT_LOSS: Optional[str] = None
    NIGHT_SWEATS: Optional[str] = None
    CHEST_PAIN: Optional[str] = None
    HEMOPTYSIS: Optional[str] = None
    HIV_STATUS: Optional[str] = None
    HAS_DIABETES: Optional[str] = None
    SMOKES: Optional[str] = None
    CONSUMES_ALCOHOL: Optional[str] = None
    PAST_TB_DIAGNOSIS: Optional[str] = None
    TB_CONTACT: Optional[float] = None
    NUMBER_OF_OCCUPANTS: Optional[float] = None
    BMI: Optional[float] = None
    BASELINE_POSITIVE: Optional[int] = None


class Aim1StrictPredictionRequest(BaseModel):
    SEX: Optional[str] = Field(None, description="M or F")
    AGE_YEARS: Optional[float] = Field(None, alias="AGE (YEARS)")
    BMI: Optional[float] = Field(None, description="Body mass index")
    baseline_symptom_count: Optional[int] = Field(None, description="Count of baseline symptoms (0-7)")


class Aim2PredictionRequest(BaseModel):
    AGE: Optional[float] = None
    SEX: Optional[str] = None
    WEIGHT: Optional[float] = None
    TEMPERATURE: Optional[float] = None
    COUGH: Optional[str] = None
    FEVER: Optional[str] = None
    WEIGHT_LOSS: Optional[str] = None
    NIGHT_SWEATS: Optional[str] = None
    DYSPNEA: Optional[str] = None
    CHEST_PAIN: Optional[str] = None
    HEMOPTYSIS: Optional[str] = None
    HIV_STATUS: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="0 = negative, 1 = positive (at risk)")
    probability: float = Field(..., description="Probability of positive class")
    confidence: float = Field(..., description="Confidence of prediction")
    model_version: str = Field("", description="Model version used")
    model_name: str = Field("", description="Model algorithm used")


class ModelInfoResponse(BaseModel):
    aim: str
    model: str
    version: str
    timestamp: str
    n_samples: int
    n_features: int
    feature_cols: List[str]
    train_auc: Optional[float] = None
    cv_auc_mean: Optional[float] = None
    metrics: Optional[dict] = None


class BatchPredictionRequest(BaseModel):
    instances: List[dict] = Field(..., description="List of feature dictionaries")
    aim: str = Field("1", description="Aim 1 or 2")


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_aim1: Optional[str] = None
    model_aim2: Optional[str] = None
    api_version: str = "1.0.0"


class DriftCheckResponse(BaseModel):
    drift_detected: bool
    drift_ratio: Optional[float] = None
    drift_count: Optional[int] = None
    n_features: Optional[int] = None
    per_feature: Optional[dict] = None
    model_drift_detected: Optional[bool] = None
    auc_drop: Optional[float] = None
    prediction_psi: Optional[float] = None


class SyntheticDriftResponse(BaseModel):
    message: str
    variants: dict


class SimulationStartRequest(BaseModel):
    scenario: str = Field("gradual_age_shift", description="Drift scenario name")
    aim: str = Field("aim1", description="Dataset aim (aim1, aim2, aim1_strict)")
    hours: int = Field(72, description="Total simulation hours")
    records_per_window: int = Field(10, description="Records per hourly window")
    pace_seconds: float = Field(1.0, description="Seconds between windows")
    fallback: bool = Field(True, description="Use deterministic fallback if LLM unavailable")
    llm_provider: str = Field("github_models", description="LLM provider (ollama, github_models)")
    llm_model: str = Field("gpt-4o-mini", description="LLM model name")


class SimulationStatusResponse(BaseModel):
    run_id: str
    scenario: str
    aim: str
    status: str
    current_hour: int
    total_hours: int
    progress_pct: float


class SimulationResultSummary(BaseModel):
    total_hours: int
    total_alerts: int
    data_drift_hours: int
    model_drift_hours: int
    max_drift_ratio: float
    max_auc_drop: Optional[float] = None
    first_alert_hour: Optional[int] = None
    sustained_drift: bool
    output_dir: str


class SimulationResultsResponse(BaseModel):
    run_id: str
    scenario: str
    summary: SimulationResultSummary
    output_dir: str
    timeseries: list = Field(default_factory=list, description="Full time-series data (truncated in list view)")
    plots: dict = Field(default_factory=dict, description="Paths to generated plots")


class SimulationScenariosResponse(BaseModel):
    scenarios: list


class GlobalExplanationResponse(BaseModel):
    aim: str
    model: str
    version: str
    feature_names: list[str]
    mean_abs_shap: list[float]
    std_shap: list[float]
    base_value: float
    n_background_samples: int
    generated_at: str
    plot_base64: str | None = None


class InstanceExplanationResponse(BaseModel):
    aim: str
    model: str
    version: str
    prediction: int
    probability: float
    base_value: float
    features: dict[str, float]
    shap_values: dict[str, float]
    feature_names: list[str]
    waterfall_plot_base64: str | None = None
    force_plot_base64: str | None = None
