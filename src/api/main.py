from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io

from src.api.schemas import (
    Aim1PredictionRequest,
    Aim1StrictPredictionRequest,
    Aim2PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    ModelInfoResponse,
    HealthResponse,
)
from src.api.dependencies import get_model, get_feature_cols, clear_cache
from src.models.predict import predict_single, predict_batch, get_model_info

app = FastAPI(
    title="TB Recovery Prediction API",
    description="ML API for TB treatment outcome prediction and contact risk assessment",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEATURE_MAP_AIM1_STRICT = {
    "SEX": "SEX",
    "AGE_YEARS": "AGE (YEARS)",
    "AGE (YEARS)": "AGE (YEARS)",
    "BMI": "BMI",
    "baseline_symptom_count": "baseline_symptom_count",
}

FEATURE_MAP_AIM1 = {
    "SEX": "SEX",
    "AGE_YEARS": "AGE (YEARS)",
    "TEMPERATURE_CELCIUS": "TEMPERATURE_CELCIUS",
    "COUGH": "COUGH",
    "FEVER": "FEVER",
    "WEIGHT_LOSS": "WEIGHT LOSS",
    "NIGHT_SWEATS": "NIGHT SWEATS",
    "CHEST_PAIN": "CHEST PAIN",
    "HEMOPTYSIS": "HEMOPTYSIS",
    "HIV_STATUS": "HIV_STATUS",
    "HAS_DIABETES": "HAS_DIABETES",
    "SMOKES": "SMOKES",
    "CONSUMES_ALCOHOL": "CONSUMES_ALCOHOL",
    "PAST_TB_DIAGNOSIS": "PAST_TB_DIAGNOSIS",
    "TB_CONTACT": "TB_CONTACT",
    "NUMBER_OF_OCCUPANTS": "NUMBER_OF_OCCUPANTS",
    "BMI": "BMI",
    "BASELINE_POSITIVE": "BASELINE_POSITIVE",
}


@app.get("/health", response_model=HealthResponse)
def health_check():
    aim1_info = get_model_info("aim1_non_conversion")
    aim2_info = get_model_info("aim2_contact_risk")
    return HealthResponse(
        status="healthy",
        model_aim1=f"{aim1_info['model']} ({aim1_info['version']})" if aim1_info else None,
        model_aim2=f"{aim2_info['model']} ({aim2_info['version']})" if aim2_info else None,
    )


@app.post("/predict/aim1", response_model=PredictionResponse)
def predict_aim1(request: Aim1PredictionRequest):
    try:
        pipeline, version, model_name = get_model("aim1_non_conversion")
        feature_cols = get_feature_cols("aim1_non_conversion")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = request.model_dump(by_alias=True)
    mapped = {}
    for api_key, model_key in FEATURE_MAP_AIM1.items():
        val = features.get(api_key) or features.get(model_key)
        if val is not None:
            mapped[model_key] = val

    result = predict_single(pipeline, mapped, feature_cols)
    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        confidence=result["confidence"],
        model_version=version,
        model_name=model_name,
    )


@app.post("/predict/aim1/strict", response_model=PredictionResponse)
def predict_aim1_strict(request: Aim1StrictPredictionRequest):
    try:
        pipeline, version, model_name = get_model("aim1_non_conversion_strict")
        feature_cols = get_feature_cols("aim1_non_conversion_strict")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = request.model_dump(by_alias=True)
    mapped = {}
    for api_key, model_key in FEATURE_MAP_AIM1_STRICT.items():
        val = features.get(api_key) or features.get(model_key)
        if val is not None:
            mapped[model_key] = val

    result = predict_single(pipeline, mapped, feature_cols)
    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        confidence=result["confidence"],
        model_version=version,
        model_name=model_name,
    )


@app.post("/predict/aim2", response_model=PredictionResponse)
def predict_aim2(request: Aim2PredictionRequest):
    try:
        pipeline, version, model_name = get_model("aim2_contact_risk")
        feature_cols = get_feature_cols("aim2_contact_risk")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    features = request.model_dump()
    mapped = {k: v for k, v in features.items() if v is not None}

    result = predict_single(pipeline, mapped, feature_cols)
    return PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        confidence=result["confidence"],
        model_version=version,
        model_name=model_name,
    )


@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch_endpoint(request: BatchPredictionRequest):
    aim = "aim1_non_conversion" if request.aim == "1" else "aim2_contact_risk"
    try:
        pipeline, version, model_name = get_model(aim)
        feature_cols = get_feature_cols(aim)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    df = pd.DataFrame(request.instances)
    results = predict_batch(pipeline, df, feature_cols)

    responses = []
    for _, row in results.iterrows():
        responses.append(PredictionResponse(
            prediction=int(row["prediction"]),
            probability=float(row["probability_positive"]),
            confidence=float(row["confidence"]),
            model_version=version,
            model_name=model_name,
        ))
    return responses


@app.post("/predict/csv")
async def predict_csv(file: UploadFile = File(...), aim: str = "1"):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    aim_key = "aim1_non_conversion" if aim == "1" else "aim2_contact_risk"
    try:
        pipeline, version, model_name = get_model(aim_key)
        feature_cols = get_feature_cols(aim_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    results = predict_batch(pipeline, df, feature_cols)
    return results.to_dict(orient="records")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info(aim: str = "1"):
    aim_key = "aim1_non_conversion" if aim == "1" else "aim2_contact_risk"
    info = get_model_info(aim_key)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No trained model for aim {aim}")
    return ModelInfoResponse(
        aim=aim_key,
        model=info["model"],
        version=info["version"],
        timestamp=info["timestamp"],
        n_samples=info["n_samples"],
        n_features=info["n_features"],
        feature_cols=info["feature_cols"],
        train_auc=info.get("train_auc"),
        cv_auc_mean=info.get("cv_auc_mean"),
        metrics={
            "cv_auc_std": info.get("cv_auc_std"),
            "train_avg_precision": info.get("train_avg_precision"),
        },
    )


@app.post("/cache/clear")
def clear_model_cache():
    clear_cache()
    return {"status": "cache cleared"}
