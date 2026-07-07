import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional


REPORTS_DIR = Path(__file__).resolve().parents[2] / "monitoring_reports"


class SimulationOutputs:
    def __init__(self, scenario_name: str, output_dir: Optional[Path] = None):
        self.scenario_name = scenario_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = output_dir or (REPORTS_DIR / scenario_name / timestamp)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_timeseries(self, results: list):
        path = self.output_dir / "drift_timeseries.json"
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return path

    def save_alert_log(self, alerts: list):
        path = self.output_dir / "alert_log.json"
        with open(path, "w") as f:
            json.dump(alerts, f, indent=2, default=str)
        return path

    def save_summary(self, summary: dict):
        path = self.output_dir / "simulation_summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        return path

    def plot_drift_over_time(self, results: list):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        hours = [r["hour"] for r in results]
        drift_ratios = [r.get("drift_ratio", 0) for r in results]
        auc_drops = []
        for r in results:
            v = r.get("auc_drop")
            auc_drops.append(v if v is not None else 0)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax1.plot(hours, drift_ratios, "b-", linewidth=2, label="Drift ratio")
        ax1.axhline(y=0.3, color="r", linestyle="--", alpha=0.7, label="Threshold (0.3)")
        ax1.fill_between(hours, drift_ratios, 0.3, where=np.array(drift_ratios) > 0.3,
                         color="r", alpha=0.1, label="Drift detected")
        ax1.set_ylabel("Drift ratio")
        ax1.set_title(f"Data Drift Over Time — {self.scenario_name}")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(hours, auc_drops, "r-", linewidth=2, label="AUC drop")
        ax2.axhline(y=0.05, color="orange", linestyle="--", alpha=0.7, label="Threshold (0.05)")
        ax2.set_xlabel("Simulation hour")
        ax2.set_ylabel("AUC drop")
        ax2.set_title("Model Performance Drift Over Time")
        ax2.legend()
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        path = self.output_dir / "drift_over_time.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_alert_timeline(self, results: list):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        alerted_hours = [r["hour"] for r in results if r.get("drift_detected") or r.get("model_drift_detected")]
        if not alerted_hours:
            fig, ax = plt.subplots(figsize=(12, 2))
            ax.text(0.5, 0.5, "No alerts triggered during simulation",
                    ha="center", va="center", fontsize=12)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis("off")
            path = self.output_dir / "alert_timeline.png"
            fig.savefig(path, dpi=100, bbox_inches="tight")
            plt.close(fig)
            return path

        drift_hours = [r["hour"] for r in results if r.get("drift_detected") and not r.get("model_drift_detected")]
        model_hours = [r["hour"] for r in results if r.get("model_drift_detected")]
        both_hours = [r["hour"] for r in results if r.get("drift_detected") and r.get("model_drift_detected")]

        fig, ax = plt.subplots(figsize=(12, 3))
        y_pos = 1

        def plot_blocks(hour_list, y, color, label):
            if not hour_list:
                return
            groups = []
            current = [hour_list[0]]
            for h in hour_list[1:]:
                if h == current[-1] + 1:
                    current.append(h)
                else:
                    groups.append(current)
                    current = [h]
            groups.append(current)
            for g in groups:
                ax.barh(y, len(g), left=g[0], height=0.4, color=color,
                        edgecolor="white", linewidth=0.5)

        plot_blocks(drift_hours, y_pos, "gold", "Data drift only")
        plot_blocks(model_hours, y_pos + 1, "salmon", "Model drift only")
        plot_blocks(both_hours, y_pos + 2, "red", "Both")

        ax.set_yticks([y_pos, y_pos + 1, y_pos + 2])
        ax.set_yticklabels(["Data drift", "Model drift", "Both"])
        ax.set_xlabel("Simulation hour")
        ax.set_title("Alert Timeline")
        ax.set_xlim(0, results[-1]["hour"] if results else 72)
        ax.grid(alpha=0.3, axis="x")

        plt.tight_layout()
        path = self.output_dir / "alert_timeline.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_feature_drift_heatmap(self, results: list):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        all_features = set()
        for r in results:
            per_feature = r.get("per_feature", {})
            all_features.update(per_feature.keys())

        if not all_features:
            return None

        features = sorted(all_features)
        matrix = np.zeros((len(features), len(results)))
        for i, r in enumerate(results):
            per_feature = r.get("per_feature", {})
            for j, feat in enumerate(features):
                fi = per_feature.get(feat, {})
                matrix[j, i] = 1.0 if fi.get("drift_detected") else 0.0

        fig, ax = plt.subplots(figsize=(12, max(4, len(features) * 0.4)))
        im = ax.imshow(matrix, aspect="auto", cmap="Reds", interpolation="nearest")
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel("Simulation hour")
        ax.set_title("Feature-Level Drift Over Time (red = drift detected)")
        plt.colorbar(im, ax=ax, ticks=[0, 1], label="Drift")
        plt.tight_layout()
        path = self.output_dir / "feature_drift_heatmap.png"
        fig.savefig(path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return path

    def generate_all_plots(self, results: list):
        paths = {}
        paths["drift_over_time"] = str(self.plot_drift_over_time(results))
        paths["alert_timeline"] = str(self.plot_alert_timeline(results))
        paths["feature_heatmap"] = str(self.plot_feature_drift_heatmap(results))
        return paths
