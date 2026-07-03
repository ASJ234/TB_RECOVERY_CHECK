import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cleaned"


class DataGenerator:
    def __init__(self, aim: str = "aim1", random_state: int = 42):
        if aim not in ("aim1", "aim2", "aim1_strict"):
            raise ValueError(f"Unknown aim: {aim}")
        self.aim = aim
        self.rng = np.random.default_rng(random_state)
        self._load_reference()

    def _load_reference(self):
        csv_map = {
            "aim1": "aim1_patients_imputed.csv",
            "aim2": "aim2_contacts_imputed.csv",
            "aim1_strict": "aim1_patients_strict.csv",
        }
        self.df = pd.read_csv(DATA_DIR / csv_map[self.aim])
        self._compute_distributions()

    _ID_COLUMNS = {"Sample ID", "USUBJID", "SUBJID", "PATIENT_ID", "STUDY_ID"}

    def _compute_distributions(self):
        self.distributions = {}
        self.feature_order = []

        for c in self.df.columns:
            if c.startswith("Unnamed"):
                continue
            if c in self._ID_COLUMNS:
                continue
            self.feature_order.append(c)
            vals = self.df[c].dropna()
            if vals.dtype in (np.float64, np.int64, float, int):
                self.distributions[c] = {
                    "type": "numeric",
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "q25": float(vals.quantile(0.25)),
                    "q50": float(vals.quantile(0.50)),
                    "q75": float(vals.quantile(0.75)),
                }
            else:
                dist = vals.value_counts(normalize=True).to_dict()
                self.distributions[c] = {
                    "type": "categorical",
                    "distribution": {str(k): float(v) for k, v in dist.items()},
                }

    def get_reference_distributions(self) -> dict:
        return dict(self.distributions)

    def get_feature_order(self) -> list:
        return list(self.feature_order)

    def generate_window(
        self,
        n_records: int = 10,
        drift_params: Optional[dict] = None,
    ) -> pd.DataFrame:
        if drift_params is None:
            drift_params = {}

        records = {}
        for c in self.feature_order:
            ref = self.distributions[c]
            dp = drift_params.get(c, {})

            if ref["type"] == "numeric":
                mean = dp.get("mean", ref["mean"])
                std = dp.get("std", ref["std"])
                lo = dp.get("min", ref["min"])
                hi = dp.get("max", ref["max"])

                vals = self.rng.normal(mean, std, size=n_records)
                vals = np.clip(vals, lo, hi)

                if c in ("BASELINE_POSITIVE", "TARGET_NON_CONVERSION_ANY",
                         "TARGET_NON_CONVERSION_2M", "TARGET_NON_CONVERSION_5M",
                         "IS_IMPUTED", "TARGET_STRICT_M2", "M2_DISCORDANT",
                         "TARGET_SYMPTOM_PRESENT"):
                    vals = np.clip(np.round(vals), 0, 1).astype(int)

                if c in ("TB_CONTACT", "NUMBER_OF_OCCUPANTS"):
                    vals = np.clip(np.round(vals), ref["min"], ref["max"]).astype(int)

                records[c] = vals
            else:
                dist_param = dp.get("distribution", ref["distribution"])
                categories = list(dist_param.keys())
                probs = list(dist_param.values())
                probs = np.array(probs) / sum(probs)
                records[c] = self.rng.choice(categories, size=n_records, p=probs)

        df = pd.DataFrame(records)

        if self.aim == "aim1":
            self._apply_aim1_correlations(df, drift_params)
        elif self.aim == "aim2":
            self._apply_aim2_correlations(df, drift_params)
        elif self.aim == "aim1_strict":
            self._apply_strict_correlations(df, drift_params)

        return df

    def _apply_aim1_correlations(self, df: pd.DataFrame, drift_params: dict):
        cough_yes = df["COUGH"] == "YES"

        fever_rate = drift_params.get("FEVER", {}).get("distribution", self.distributions["FEVER"]["distribution"])
        fever_p_yes = fever_rate.get("YES", 0.2)
        for idx in df.index[cough_yes]:
            p = min(1.0, fever_p_yes * 1.1)
            if self.rng.random() > p:
                df.at[idx, "FEVER"] = "NO"

        weight_loss_rate = drift_params.get("WEIGHT LOSS", {}).get("distribution",
                            self.distributions["WEIGHT LOSS"]["distribution"])
        wl_p_yes = weight_loss_rate.get("YES", 0.2)
        for idx in df.index[cough_yes]:
            p = min(1.0, wl_p_yes * 1.05)
            if self.rng.random() > p:
                df.at[idx, "WEIGHT LOSS"] = "NO"

        night_sweats_rate = drift_params.get("NIGHT SWEATS", {}).get("distribution",
                            self.distributions["NIGHT SWEATS"]["distribution"])
        ns_p_yes = night_sweats_rate.get("YES", 0.2)
        for idx in df.index[cough_yes]:
            p = min(1.0, ns_p_yes * 1.05)
            if self.rng.random() > p:
                df.at[idx, "NIGHT SWEATS"] = "NO"

    def _apply_aim2_correlations(self, df: pd.DataFrame, drift_params: dict):
        pass

    def _apply_strict_correlations(self, df: pd.DataFrame, drift_params: dict):
        pass
