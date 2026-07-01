import sys
import time
import argparse
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.simulation.llm_client import LLMClient
from src.simulation.drift_scenarios import SCENARIOS, list_scenarios, get_scenario
from src.simulation.data_generator import DataGenerator
from src.simulation.outputs import SimulationOutputs
from src.pipeline.config import get_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stream_simulator")


class StreamSimulator:
    def __init__(
        self,
        scenario_name: str = "gradual_age_shift",
        aim: str = "aim1",
        total_hours: int = 72,
        records_per_window: int = 10,
        pace_seconds: float = 1.0,
        fallback: bool = True,
        llm_model: str = "mistral",
        random_state: int = 42,
    ):
        self.scenario = get_scenario(scenario_name)
        if self.scenario is None:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list_scenarios()}")
        self.scenario_name = scenario_name
        self.aim = aim
        self.total_hours = total_hours
        self.records_per_window = records_per_window
        self.pace_seconds = pace_seconds
        self.random_state = random_state

        self.data_gen = DataGenerator(aim=aim, random_state=random_state)
        self.reference_distributions = self.data_gen.get_reference_distributions()
        self.reference_df = self.data_gen.df

        self.llm = LLMClient(
            model=llm_model,
            fallback=fallback,
        )
        self.outputs = SimulationOutputs(scenario_name)

        self.results = []
        self.alerts = []
        self.pipeline = None

    def _load_champion_model(self):
        try:
            from src.models.train import get_champion_model
            obj_to_aim = {"aim1": "aim1_non_conversion", "aim2": "aim2_contact_risk",
                          "aim1_strict": "aim1_non_conversion_strict"}
            aim_key = obj_to_aim.get(self.aim, "aim1_non_conversion")
            champion = get_champion_model(aim_key)
            if champion is None:
                logger.info("No trained model found for %s — skipping model drift checks", aim_key)
                return None
            import joblib
            pipe = joblib.load(champion["model_path"])
            self.pipeline = pipe
            self.model_feature_cols = champion.get("feature_cols", [])
            logger.info("Loaded champion model: %s (%s)", champion["model"], champion["version"])
            return pipe
        except Exception as e:
            logger.warning("Could not load champion model: %s", e)
            return None

    def run(self):
        logger.info("=" * 60)
        logger.info("Starting 72-hour stream simulation")
        logger.info("Scenario: %s", self.scenario_name)
        logger.info("Aim: %s", self.aim)
        logger.info("Records per window: %d", self.records_per_window)
        logger.info("Pace: %.1f seconds per window", self.pace_seconds)
        logger.info("LLM fallback: %s", "enabled" if self.llm.fallback else "disabled")
        logger.info("=" * 60)

        self._load_champion_model()
        param_history = []

        for hour in range(self.total_hours):
            window_start = time.time()

            drift_params = self.llm.generate_drift_params(
                scenario_name=self.scenario_name,
                current_hour=hour,
                total_hours=self.total_hours,
                reference_distributions=self.reference_distributions,
                history=param_history,
            )
            drift_params["hour"] = hour
            param_history.append(drift_params)
            if len(param_history) > 10:
                param_history.pop(0)

            window_df = self.data_gen.generate_window(
                n_records=self.records_per_window,
                drift_params=drift_params,
            )

            result = self._run_drift_checks(window_df, hour)
            self.results.append(result)

            if result.get("drift_detected") or result.get("model_drift_detected"):
                self.alerts.append({
                    "hour": hour,
                    "data_drift": result.get("drift_detected", False),
                    "model_drift": result.get("model_drift_detected", False),
                    "drift_ratio": result.get("drift_ratio"),
                    "auc_drop": result.get("auc_drop"),
                })

            self._log_hour_result(hour, result)

            elapsed = time.time() - window_start
            sleep_time = max(0, self.pace_seconds - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

        self._finalize()

    def _run_drift_checks(self, window_df: pd.DataFrame, hour: int) -> dict:
        from src.monitoring.data_drift import compute_data_drift

        feature_cols = self.data_gen.get_feature_order()
        available = [c for c in feature_cols if c in self.reference_df.columns and c in window_df.columns]
        if not available:
            return {"hour": hour, "error": "no overlapping features"}

        data_result = compute_data_drift(
            self.reference_df, window_df,
            feature_cols=available,
        )

        result = {
            "hour": hour,
            "drift_detected": data_result.get("data_drift_detected", False),
            "drift_ratio": data_result.get("drift_ratio", 0.0),
            "drift_count": data_result.get("drift_count", 0),
            "n_features": data_result.get("n_features", 0),
            "per_feature": data_result.get("per_feature", {}),
            "model_drift_detected": False,
            "auc_drop": None,
        }

        if self.pipeline is not None:
            model_result = self._run_model_drift(window_df)
            result["model_drift_detected"] = model_result.get("model_drift_detected", False)
            result["performance_drift"] = model_result.get("performance_drift", False)
            result["prediction_drift"] = model_result.get("prediction_drift", False)
            result["auc_drop"] = model_result.get("auc_drop")
            result["prediction_psi"] = model_result.get("prediction_psi")

        return result

    def _run_model_drift(self, window_df: pd.DataFrame) -> dict:
        from src.monitoring.model_drift import compute_model_drift

        target_col = None
        for possible in ["TARGET_NON_CONVERSION_ANY", "TARGET_SYMPTOM_PRESENT", "TARGET_STRICT_M2"]:
            if possible in self.reference_df.columns and possible in window_df.columns:
                target_col = possible
                break

        if target_col is None:
            return {"model_drift_detected": False, "error": "no target column"}

        config = get_config()
        mon_cfg = config.monitoring

        available_feats = [c for c in self.model_feature_cols
                           if c in self.reference_df.columns and c in window_df.columns]
        if not available_feats:
            return {"model_drift_detected": False, "error": "no overlapping model features"}

        try:
            result = compute_model_drift(
                self.pipeline,
                self.reference_df, self.reference_df[target_col].astype(int),
                window_df, window_df[target_col].astype(int),
                feature_cols=available_feats,
                auc_drop_threshold=mon_cfg.model_drift_thresholds.auc_drop,
                prediction_psi_threshold=mon_cfg.model_drift_thresholds.prediction_psi,
            )
            return result
        except Exception as e:
            logger.warning("Model drift check failed at hour %d: %s", window_df.get("hour", 0), e)
            return {"model_drift_detected": False, "error": str(e)}

    def _log_hour_result(self, hour: int, result: dict):
        data_status = "DRIFT" if result.get("drift_detected") else "ok"
        model_status = "DRIFT" if result.get("model_drift_detected") else "ok"
        drift_pct = result.get("drift_ratio", 0) * 100
        auc = result.get("auc_drop")
        auc_str = f", AUC drop={auc:.3f}" if auc is not None else ""
        logger.info(
            "Hour %2d/%d | data=%s (%.0f%% features) | model=%s%s",
            hour + 1, self.total_hours, data_status, drift_pct, model_status, auc_str,
        )

    def _finalize(self):
        total_alerts = len(self.alerts)
        data_alert_hours = sum(1 for a in self.alerts if a.get("data_drift"))
        model_alert_hours = sum(1 for a in self.alerts if a.get("model_drift"))

        summary = {
            "scenario": self.scenario_name,
            "aim": self.aim,
            "total_hours": self.total_hours,
            "records_per_window": self.records_per_window,
            "total_records_generated": self.total_hours * self.records_per_window,
            "timestamp": datetime.now().isoformat(),
            "llm_source": self.results[0].get("llm_source", "fallback") if self.results else "unknown",
            "alerts": {
                "total": total_alerts,
                "data_drift_hours": data_alert_hours,
                "model_drift_hours": model_alert_hours,
                "max_drift_ratio": max((r.get("drift_ratio", 0) for r in self.results), default=0),
                "max_auc_drop": max((r.get("auc_drop") or 0 for r in self.results), default=0),
            },
            "first_alert_hour": self.alerts[0]["hour"] if self.alerts else None,
            "sustained_drift": self._detect_sustained_drift(),
            "output_dir": str(self.outputs.output_dir),
        }

        self.outputs.save_timeseries(self.results)
        self.outputs.save_alert_log(self.alerts)
        self.outputs.save_summary(summary)
        self.outputs.generate_all_plots(self.results)

        logger.info("=" * 60)
        logger.info("Simulation complete")
        logger.info("Alerts: %d total (%d data, %d model)", total_alerts, data_alert_hours, model_alert_hours)
        logger.info("Max drift ratio: %.1f%%", summary["alerts"]["max_drift_ratio"] * 100)
        logger.info("Max AUC drop: %.3f", summary["alerts"]["max_auc_drop"])
        logger.info("Output: %s", self.outputs.output_dir)
        logger.info("=" * 60)

    def _detect_sustained_drift(self):
        if len(self.results) < 6:
            return False
        recent = self.results[-6:]
        data_drifts = [r.get("drift_detected", False) for r in recent]
        model_drifts = [r.get("model_drift_detected", False) for r in recent]
        sustained_data = sum(data_drifts) >= 4
        sustained_model = sum(model_drifts) >= 4
        return sustained_data or sustained_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="TB Recovery — 72-Hour LLM-Driven Stream Simulation"
    )
    parser.add_argument(
        "--scenario", type=str, default="gradual_age_shift",
        choices=list_scenarios() + ["all"],
        help="Drift scenario to simulate (default: gradual_age_shift)",
    )
    parser.add_argument(
        "--aim", type=str, default="aim1",
        choices=["aim1", "aim2", "aim1_strict"],
        help="Dataset to simulate (default: aim1)",
    )
    parser.add_argument(
        "--hours", type=int, default=72,
        help="Total simulation hours (default: 72)",
    )
    parser.add_argument(
        "--records", type=int, default=10,
        help="Records per hourly window (default: 10)",
    )
    parser.add_argument(
        "--pace", type=float, default=1.0,
        help="Seconds between windows (default: 1.0)",
    )
    parser.add_argument(
        "--no-fallback", action="store_true",
        help="Disable deterministic fallback — require LLM",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true",
        help="List available drift scenarios and exit",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--llm-model", type=str, default="mistral",
        help="Ollama model name (default: mistral)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_scenarios:
        print("Available drift scenarios:")
        for name, sc in SCENARIOS.items():
            print(f"  {name:30s} {sc.description}")
            print(f"  {'':30s} Aim: {sc.target_aim}, Drift features: {list(sc.drift_curves.keys())}")
            print()
        return

    scenarios_to_run = list_scenarios() if args.scenario == "all" else [args.scenario]

    for scenario_name in scenarios_to_run:
        logger.info("Starting simulation for scenario: %s", scenario_name)
        sim = StreamSimulator(
            scenario_name=scenario_name,
            aim=args.aim,
            total_hours=args.hours,
            records_per_window=args.records,
            pace_seconds=args.pace,
            fallback=not args.no_fallback,
            llm_model=args.llm_model,
            random_state=args.seed,
        )
        sim.run()


if __name__ == "__main__":
    main()
