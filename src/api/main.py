from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import json
from pathlib import Path
import numpy as np
from typing import Optional, List

from src.api.schemas import (
    Aim1PredictionRequest,
    Aim1StrictPredictionRequest,
    Aim2PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    ModelInfoResponse,
    HealthResponse,
    DriftCheckResponse,
    SyntheticDriftResponse,
    SimulationStartRequest,
    SimulationStatusResponse,
    SimulationResultsResponse,
    SimulationResultSummary,
    SimulationScenariosResponse,
    GlobalExplanationResponse,
    InstanceExplanationResponse,
)
from src.api.dependencies import get_model, get_feature_cols, clear_cache, get_explainer, clear_explainer_cache
from src.models.predict import predict_single, predict_batch, get_model_info, METADATA_DIR
from src.monitoring.data_drift import compute_data_drift
from src.monitoring.model_drift import compute_model_drift
from src.monitoring.reporting import generate_drift_report, save_drift_report, log_to_mlflow
from src.monitoring.synthetic_drift import generate_all_synthetic_variants
from src.pipeline.config import get_config
from src.simulation.drift_scenarios import SCENARIOS
from src.simulation.stream_simulator import StreamSimulator

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

FEATURE_MAP_AIM2 = {
    "AGE": "AGE",
    "SEX": "SEX",
    "WEIGHT": "WEIGHT",
    "TEMPERATURE": "TEMPERATURE",
    "COUGH": "COUGH",
    "FEVER": "FEVER",
    "WEIGHT_LOSS": "WEIGHT LOSS",
    "NIGHT_SWEATS": "NIGHT SWEATS",
    "DYSPNEA": "DYSPNEA",
    "CHEST_PAIN": "CHEST PAIN",
    "HEMOPTYSIS": "HEMOPTYSIS",
    "HIV_STATUS": "HIV STATUS",
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


def get_strict_model_info():
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    strict_metas = sorted(METADATA_DIR.glob("aim1_non_conversion_strict_logistic_*.json"))
    if not strict_metas:
        return None
    with open(strict_metas[-1]) as f:
        return json.load(f)


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
    mapped = {}
    for api_key, model_key in FEATURE_MAP_AIM2.items():
        val = features.get(api_key)
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
def model_info(aim: str = "1", params: str = None):
    if params == "strict":
        aim_key = "aim1_non_conversion_strict"
        info = get_strict_model_info()
    else:
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
    clear_explainer_cache()
    return {"status": "cache cleared"}


@app.post("/monitor/data-drift", response_model=DriftCheckResponse)
async def monitor_data_drift(
    file: UploadFile = File(...),
    aim: str = Query("1", description="Aim 1 or 2"),
):
    content = await file.read()
    current_df = pd.read_csv(io.BytesIO(content))

    config = get_config()
    mon_cfg = config.monitoring

    ref_csv = "aim1_patients_imputed.csv" if aim == "1" else "aim2_contacts_imputed.csv"
    ref_path = Path(__file__).resolve().parents[2] / "data" / "cleaned" / ref_csv
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference data not found: {ref_path}")
    reference_df = pd.read_csv(ref_path)

    _, feature_cols = _get_feature_cols_for_aim(aim)
    available = [c for c in feature_cols if c in reference_df.columns and c in current_df.columns]

    result = compute_data_drift(
        reference_df, current_df,
        feature_cols=available,
        psi_bins=mon_cfg.psi_bins,
    )

    report = generate_drift_report(data_drift_result=result)
    if mon_cfg.enabled:
        save_drift_report(report)
        log_to_mlflow(report, run_name=f"data_drift_aim{aim}")

    return DriftCheckResponse(
        drift_detected=result.get("data_drift_detected", False),
        drift_ratio=result.get("drift_ratio"),
        drift_count=result.get("drift_count"),
        n_features=result.get("n_features"),
        per_feature=result.get("per_feature"),
    )


@app.post("/monitor/model-drift", response_model=DriftCheckResponse)
async def monitor_model_drift(
    file: UploadFile = File(...),
    aim: str = Query("1", description="Aim 1 or 2"),
):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))

    config = get_config()
    mon_cfg = config.monitoring
    md_thresh = mon_cfg.model_drift_thresholds

    aim_key = "aim1_non_conversion" if aim == "1" else "aim2_contact_risk"
    try:
        pipeline, version, model_name = get_model(aim_key)
        feature_cols = get_feature_cols(aim_key)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=f"No trained model: {e}")

    target_col = "TARGET_NON_CONVERSION_ANY" if aim == "1" else "TARGET_SYMPTOM_PRESENT"
    if target_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Label column '{target_col}' required")

    ref_csv = "aim1_patients_imputed.csv" if aim == "1" else "aim2_contacts_imputed.csv"
    ref_path = Path(__file__).resolve().parents[2] / "data" / "cleaned" / ref_csv
    if not ref_path.exists():
        raise HTTPException(status_code=404, detail=f"Reference data not found: {ref_path}")
    reference_df = pd.read_csv(ref_path)
    if target_col not in reference_df.columns:
        raise HTTPException(status_code=500, detail=f"Reference data missing label column '{target_col}'")

    ref_data = reference_df[feature_cols]
    ref_labels = reference_df[target_col].astype(int)
    cur_data = df[feature_cols]
    cur_labels = df[target_col].astype(int)

    result = compute_model_drift(
        pipeline, ref_data, ref_labels, cur_data, cur_labels,
        feature_cols=feature_cols,
        auc_drop_threshold=md_thresh.auc_drop,
        prediction_psi_threshold=md_thresh.prediction_psi,
    )

    report = generate_drift_report(model_drift_result=result)
    if mon_cfg.enabled:
        save_drift_report(report)
        log_to_mlflow(report, run_name=f"model_drift_aim{aim}_{model_name}_{version}")

    return DriftCheckResponse(
        drift_detected=result.get("model_drift_detected", False),
        model_drift_detected=result.get("performance_drift"),
        auc_drop=result.get("auc_drop"),
        prediction_psi=result.get("prediction_psi"),
    )


@app.get("/monitor/report")
def get_latest_drift_report():
    MONITORING_DIR = Path(__file__).resolve().parents[2] / "monitoring_reports"
    if not MONITORING_DIR.exists():
        raise HTTPException(status_code=404, detail="No drift reports found")
    reports = sorted(MONITORING_DIR.glob("drift_report_*.json"), reverse=True)
    if not reports:
        raise HTTPException(status_code=404, detail="No drift reports found")
    with open(reports[0]) as f:
        return json.load(f)


@app.post("/monitor/generate-synthetic", response_model=SyntheticDriftResponse)
def generate_synthetic_drift_data(aim: str = Query("1", description="Aim 1 or 2")):
    source_map = {
        "1": ("aim1_patients_imputed.csv", "aim1"),
        "2": ("aim2_contacts_imputed.csv", "aim2"),
    }
    if aim not in source_map:
        raise HTTPException(status_code=400, detail="aim must be '1' or '2'")
    source_csv, prefix = source_map[aim]
    try:
        variants = generate_all_synthetic_variants(source_csv, output_prefix=prefix)
        return SyntheticDriftResponse(
            message=f"Synthetic drift data generated for aim {aim}",
            variants=variants,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


_simulation_runs: dict[str, dict] = {}


@app.get("/simulate/scenarios", response_model=SimulationScenariosResponse)
def list_simulation_scenarios():
    scenarios = [
        {"name": name, "description": sc.description, "aim": sc.target_aim, "drift_features": list(sc.drift_curves.keys())}
        for name, sc in SCENARIOS.items()
    ]
    return SimulationScenariosResponse(scenarios=scenarios)


@app.post("/simulate/start", response_model=SimulationStatusResponse)
def start_simulation(request: SimulationStartRequest):
    import uuid
    run_id = str(uuid.uuid4())[:8]

    sim = StreamSimulator(
        scenario_name=request.scenario,
        aim=request.aim,
        total_hours=request.hours,
        records_per_window=request.records_per_window,
        pace_seconds=request.pace_seconds,
        fallback=request.fallback,
        llm_model=request.llm_model,
    )

    _simulation_runs[run_id] = {
        "simulator": sim,
        "scenario": request.scenario,
        "aim": request.aim,
        "status": "running",
        "current_hour": 0,
        "total_hours": request.hours,
    }

    import threading
    def _run():
        try:
            sim.run()
            _simulation_runs[run_id]["status"] = "completed"
        except Exception as e:
            _simulation_runs[run_id]["status"] = f"failed: {e}"

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return SimulationStatusResponse(
        run_id=run_id,
        scenario=request.scenario,
        aim=request.aim,
        status="running",
        current_hour=0,
        total_hours=request.hours,
        progress_pct=0.0,
    )


@app.get("/simulate/status/{run_id}", response_model=SimulationStatusResponse)
def get_simulation_status(run_id: str):
    run = _simulation_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    sim = run["simulator"]
    current = len(sim.results) if hasattr(sim, "results") else 0
    total = run["total_hours"]
    return SimulationStatusResponse(
        run_id=run_id,
        scenario=run["scenario"],
        aim=run["aim"],
        status=run["status"],
        current_hour=current,
        total_hours=total,
        progress_pct=(current / total * 100) if total > 0 else 0.0,
    )


@app.get("/simulate/results/{run_id}", response_model=SimulationResultsResponse)
def get_simulation_results(run_id: str):
    run = _simulation_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if run["status"] == "running":
        raise HTTPException(status_code=400, detail="Simulation still running. Check status first.")

    sim = run["simulator"]
    alerts = sim.alerts if hasattr(sim, "alerts") else []
    results = sim.results if hasattr(sim, "results") else []
    outputs = sim.outputs if hasattr(sim, "outputs") else None

    total_alerts = len(alerts)
    data_alert_hours = sum(1 for a in alerts if a.get("data_drift", False))
    model_alert_hours = sum(1 for a in alerts if a.get("model_drift", False))
    max_drift_ratio = max((r.get("drift_ratio", 0) for r in results), default=0)
    max_auc_drop = max((r.get("auc_drop") or 0 for r in results), default=0)
    first_alert = alerts[0]["hour"] if alerts else None

    sustained = sim._detect_sustained_drift() if hasattr(sim, "_detect_sustained_drift") else False

    summary = SimulationResultSummary(
        total_hours=run["total_hours"],
        total_alerts=total_alerts,
        data_drift_hours=data_alert_hours,
        model_drift_hours=model_alert_hours,
        max_drift_ratio=max_drift_ratio,
        max_auc_drop=max_auc_drop,
        first_alert_hour=first_alert,
        sustained_drift=sustained,
        output_dir=str(outputs.output_dir) if outputs else "",
    )

    plots = {}
    if outputs and outputs.output_dir.exists():
        for p in outputs.output_dir.glob("*.png"):
            plots[p.stem] = str(p)

    return SimulationResultsResponse(
        run_id=run_id,
        scenario=run["scenario"],
        summary=summary,
        output_dir=str(outputs.output_dir) if outputs else "",
        timeseries=results,
        plots=plots,
    )


def _get_feature_cols_for_aim(aim: str):
    from src.data.feature_engineering import (
        get_aim1_features_target,
        get_aim2_features_target,
    )

    data_dir = Path(__file__).resolve().parents[2] / "data" / "cleaned"
    if aim == "1":
        df = pd.read_csv(data_dir / "aim1_patients_imputed.csv")
        _, cols = get_aim1_features_target(df)
    else:
        df = pd.read_csv(data_dir / "aim2_contacts_imputed.csv")
        _, cols = get_aim2_features_target(df)
    return df, cols


def explain_instance_helper(aim: str, request_data, version: Optional[str] = None):
    pipeline, version, model_name = get_model(aim, version=version)
    feature_cols = get_feature_cols(aim)
    explainer = get_explainer(aim, model_name, version)
    
    # 1. Map features to standard columns
    features = request_data.model_dump(by_alias=True)
    mapped = {}
    feature_map = FEATURE_MAP_AIM1 if aim == "aim1_non_conversion" else FEATURE_MAP_AIM2
    for api_key, model_key in feature_map.items():
        val = features.get(api_key) or features.get(model_key)
        if val is not None:
            mapped[model_key] = val
            
    df_instance = pd.DataFrame([mapped])
    df_instance = df_instance.reindex(columns=feature_cols)
    
    # 2. Get prediction probability and prediction
    from src.models.predict import predict_single
    pred_res = predict_single(pipeline, mapped, feature_cols)
    
    # 3. Compute SHAP
    preprocessor = pipeline.named_steps["preprocessor"]
    from src.explain.shap_explainer import compute_instance_shap
    instance_result = compute_instance_shap(explainer, df_instance, feature_cols, preprocessor)
    
    # 4. Map SHAP to original features for JSON response
    from src.explain.api_explain import map_shap_to_original_features
    orig_shap, orig_names = map_shap_to_original_features(
        instance_result["shap_values"],
        instance_result["feature_names"],
        feature_cols,
        preprocessor
    )
    
    orig_features = {}
    for col in feature_cols:
        val = df_instance.iloc[0].get(col)
        if pd.isna(val) or val is None:
            orig_features[col] = 0.0
        elif isinstance(val, (int, float, np.integer, np.floating)):
            orig_features[col] = float(val)
        else:
            orig_features[col] = str(val)
            
    # 5. Generate plots
    from src.explain.visualizations import create_waterfall_plot_base64, create_force_plot_base64
    X_trans_row = preprocessor.transform(df_instance)[0]
    
    waterfall_b64 = create_waterfall_plot_base64(
        shap_values=instance_result["shap_values"],
        base_value=instance_result["base_value"],
        features=X_trans_row,
        feature_names=instance_result["feature_names"]
    )
    
    force_b64 = create_force_plot_base64(
        shap_values=instance_result["shap_values"],
        base_value=instance_result["base_value"],
        features=X_trans_row,
        feature_names=instance_result["feature_names"]
    )
    
    return InstanceExplanationResponse(
        aim=aim,
        model=model_name,
        version=version,
        prediction=pred_res["prediction"],
        probability=pred_res["probability"],
        base_value=instance_result["base_value"],
        features=orig_features,
        shap_values=orig_shap,
        feature_names=orig_names,
        waterfall_plot_base64=waterfall_b64,
        force_plot_base64=force_b64
    )


@app.get("/explain/aim1/global", response_model=GlobalExplanationResponse)
def explain_aim1_global(version: Optional[str] = None):
    try:
        pipeline, version, model_name = get_model("aim1_non_conversion", version=version)
        aim = "aim1_non_conversion"
        
        from src.explain.shap_explainer import load_global_explanation
        global_explanation = load_global_explanation(aim, model_name, version)
        
        plot_path = METADATA_DIR / aim / f"shap_summary_{model_name}_{version}.png"
        plot_b64 = None
        if plot_path.exists():
            import base64
            plot_b64 = base64.b64encode(plot_path.read_bytes()).decode("utf-8")
            
        return GlobalExplanationResponse(
            aim=aim,
            model=model_name,
            version=version,
            feature_names=global_explanation["feature_names"],
            mean_abs_shap=global_explanation["mean_abs_shap"],
            std_shap=global_explanation["std_shap"],
            base_value=global_explanation["base_value"],
            n_background_samples=global_explanation["n_background_samples"],
            generated_at=global_explanation["generated_at"],
            plot_base64=plot_b64
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Global explanation not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/aim1/strict/global", response_model=GlobalExplanationResponse)
def explain_aim1_strict_global(version: Optional[str] = None):
    try:
        pipeline, version, model_name = get_model("aim1_non_conversion_strict", version=version)
        aim = "aim1_non_conversion_strict"
        
        from src.explain.shap_explainer import load_global_explanation
        global_explanation = load_global_explanation(aim, model_name, version)
        
        plot_path = METADATA_DIR / aim / f"loocv_shap_summary_{model_name}_{version}.png"
        plot_b64 = None
        if plot_path.exists():
            import base64
            plot_b64 = base64.b64encode(plot_path.read_bytes()).decode("utf-8")
            
        return GlobalExplanationResponse(
            aim=aim,
            model=model_name,
            version=version,
            feature_names=global_explanation["feature_names"],
            mean_abs_shap=global_explanation["mean_abs_shap"],
            std_shap=global_explanation["std_shap"],
            base_value=global_explanation["base_value"],
            n_background_samples=global_explanation["n_background_samples"],
            generated_at=global_explanation["generated_at"],
            plot_base64=plot_b64
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Global explanation not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/explain/aim2/global", response_model=GlobalExplanationResponse)
def explain_aim2_global(version: Optional[str] = None):
    try:
        pipeline, version, model_name = get_model("aim2_contact_risk", version=version)
        aim = "aim2_contact_risk"
        
        from src.explain.shap_explainer import load_global_explanation
        global_explanation = load_global_explanation(aim, model_name, version)
        
        plot_path = METADATA_DIR / aim / f"shap_summary_{model_name}_{version}.png"
        plot_b64 = None
        if plot_path.exists():
            import base64
            plot_b64 = base64.b64encode(plot_path.read_bytes()).decode("utf-8")
            
        return GlobalExplanationResponse(
            aim=aim,
            model=model_name,
            version=version,
            feature_names=global_explanation["feature_names"],
            mean_abs_shap=global_explanation["mean_abs_shap"],
            std_shap=global_explanation["std_shap"],
            base_value=global_explanation["base_value"],
            n_background_samples=global_explanation["n_background_samples"],
            generated_at=global_explanation["generated_at"],
            plot_base64=plot_b64
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=f"Global explanation not found: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/aim1/instance", response_model=InstanceExplanationResponse)
def explain_aim1_instance(request: Aim1PredictionRequest, version: Optional[str] = None):
    try:
        return explain_instance_helper("aim1_non_conversion", request, version=version)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/aim2/instance", response_model=InstanceExplanationResponse)
def explain_aim2_instance(request: Aim2PredictionRequest, version: Optional[str] = None):
    try:
        return explain_instance_helper("aim2_contact_risk", request, version=version)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/aim1/batch", response_model=list[InstanceExplanationResponse])
def explain_aim1_batch(request: BatchPredictionRequest):
    try:
        responses = []
        for instance_dict in request.instances:
            req_obj = Aim1PredictionRequest(**instance_dict)
            res = explain_instance_helper("aim1_non_conversion", req_obj)
            responses.append(res)
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain/aim2/batch", response_model=list[InstanceExplanationResponse])
def explain_aim2_batch(request: BatchPredictionRequest):
    try:
        responses = []
        for instance_dict in request.instances:
            req_obj = Aim2PredictionRequest(**instance_dict)
            res = explain_instance_helper("aim2_contact_risk", req_obj)
            responses.append(res)
        return responses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
