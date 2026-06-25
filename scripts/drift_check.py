import pandas as pd
from pathlib import Path
from src.monitoring.data_drift import compute_data_drift
from src.monitoring.reporting import generate_drift_report, save_drift_report

ref = pd.read_csv(Path("data/cleaned/aim1_patients_imputed.csv"))
cur = pd.read_csv(Path("data/synthetic/aim1_drifted.csv"))
feats = ["SEX", "AGE (YEARS)", "BMI", "COUGH", "FEVER", "HIV_STATUS"]
avail = [c for c in feats if c in ref.columns and c in cur.columns]
r = compute_data_drift(ref, cur, feature_cols=avail)
status = "DETECTED" if r["data_drift_detected"] else "none"
print(f"Data drift: {status} ({r['drift_ratio']:.0%} features)")
save_drift_report(generate_drift_report(data_drift_result=r))
print("Report saved.")
