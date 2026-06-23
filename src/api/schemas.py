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
    DYSPENA: Optional[str] = None
    CHEST_PAIN: Optional[str] = None
    HEMOPTYSIS: Optional[str] = None
    HIV_STATUS: Optional[str] = None
    HAS_DIABETES: Optional[str] = None
    SMOKES: Optional[str] = None
    CONSUMES_ALCOHOL: Optional[str] = None
    PAST_TB_DIAGNOSIS: Optional[str] = None
    TB_CONTACT: Optional[str] = None
    NUMBER_OF_OCCUPANTS: Optional[float] = None
    BMI: Optional[float] = None


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
