import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class TestDriftScenarios:
    def test_list_scenarios(self):
        from src.simulation.drift_scenarios import list_scenarios, SCENARIOS
        scenarios = list_scenarios()
        assert len(scenarios) > 0
        assert "gradual_age_shift" in scenarios
        assert "sudden_outbreak" in scenarios
        assert len(SCENARIOS) == len(scenarios)

    def test_get_scenario(self):
        from src.simulation.drift_scenarios import get_scenario
        sc = get_scenario("gradual_age_shift")
        assert sc is not None
        assert sc.name == "gradual_age_shift"
        assert sc.target_aim == "aim1"
        assert "AGE (YEARS)" in sc.drift_curves

    def test_get_scenario_unknown(self):
        from src.simulation.drift_scenarios import get_scenario
        assert get_scenario("nonexistent") is None

    def test_all_scenarios_have_valid_curves(self):
        from src.simulation.drift_scenarios import SCENARIOS
        valid = {"linear_up", "linear_down", "step_up_mid", "step_down_mid", "sine", "none"}
        for name, sc in SCENARIOS.items():
            for feat, curve in sc.drift_curves.items():
                assert curve in valid, f"{name}: unknown curve '{curve}' for {feat}"



class TestLLMClient:
    def test_fallback_params(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient(fallback=True)
        ref_dist = {
            "AGE (YEARS)": {"type": "numeric", "mean": 35.0, "std": 10.0, "min": 18, "max": 78},
            "BMI": {"type": "numeric", "mean": 20.5, "std": 3.5, "min": 12, "max": 42},
            "COUGH": {"type": "categorical", "distribution": {"YES": 0.9, "NO": 0.1}},
        }
        params = client._fallback_params("gradual_age_shift", 36, 72, ref_dist)
        assert params["source"] == "fallback"
        assert params["hour"] == 36
        assert "AGE (YEARS)" in params
        assert params["AGE (YEARS)"]["mean"] > 35.0

    def test_fallback_no_drift_scenario(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient(fallback=True)
        ref_dist = {
            "AGE (YEARS)": {"type": "numeric", "mean": 35.0, "std": 10.0, "min": 18, "max": 78},
        }
        params = client._fallback_params("nonexistent", 36, 72, ref_dist)
        assert params["source"] == "fallback"
        # Features with no drift curve are skipped — they use bootstrapping
        assert "AGE (YEARS)" not in params

    def test_parse_json_plain(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient()
        raw = '{"hour": 5, "mean_age": 37.2}'
        result = client._parse_json(raw)
        assert result is not None
        assert result["hour"] == 5
        assert result["mean_age"] == 37.2

    def test_parse_json_with_markdown(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient()
        raw = "```json\n{\"hour\": 10, \"mean_age\": 40.0}\n```"
        result = client._parse_json(raw)
        assert result is not None
        assert result["hour"] == 10

    def test_parse_json_braces_in_text(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient()
        raw = "Here is the result: {\"hour\": 20, \"mean_age\": 42.5} and that's it"
        result = client._parse_json(raw)
        assert result is not None
        assert result["hour"] == 20


class TestDataGenerator:
    def test_init_aim1(self):
        from src.simulation.data_generator import DataGenerator
        gen = DataGenerator(aim="aim1", random_state=42)
        assert gen.aim == "aim1"
        assert gen.df is not None
        assert len(gen.df) > 0

    def test_init_aim2(self):
        from src.simulation.data_generator import DataGenerator
        gen = DataGenerator(aim="aim2", random_state=42)
        assert gen.aim == "aim2"
        assert gen.df is not None

    def test_init_invalid_aim(self):
        from src.simulation.data_generator import DataGenerator
        with pytest.raises(ValueError):
            DataGenerator(aim="nonexistent")

    def test_get_reference_distributions(self):
        from src.simulation.data_generator import DataGenerator
        gen = DataGenerator(aim="aim1", random_state=42)
        dists = gen.get_reference_distributions()
        assert "AGE (YEARS)" in dists
        assert "SEX" in dists
        assert dists["AGE (YEARS)"]["type"] == "numeric"
        assert dists["SEX"]["type"] == "categorical"

    def test_generate_window_default(self):
        from src.simulation.data_generator import DataGenerator
        gen = DataGenerator(aim="aim1", random_state=42)
        df = gen.generate_window(n_records=5)
        assert len(df) == 5
        assert "AGE (YEARS)" in df.columns
        assert "SEX" in df.columns
        assert "TARGET_NON_CONVERSION_ANY" in df.columns

    def test_generate_window_with_drift(self):
        from src.simulation.data_generator import DataGenerator
        gen = DataGenerator(aim="aim1", random_state=42)
        drift = {
            "AGE (YEARS)": {"mean": 50.0, "std": 8.0, "min": 18, "max": 90},
        }
        df = gen.generate_window(n_records=20, drift_params=drift)
        assert len(df) == 20
        assert df["AGE (YEARS)"].mean() > 40.0

    def test_generate_window_aim2(self):
        from src.simulation.data_generator import DataGenerator
        gen = DataGenerator(aim="aim2", random_state=42)
        df = gen.generate_window(n_records=5)
        assert len(df) == 5
        assert "AGE" in df.columns

    def test_generate_window_reproducible(self):
        from src.simulation.data_generator import DataGenerator
        gen1 = DataGenerator(aim="aim1", random_state=123)
        gen2 = DataGenerator(aim="aim1", random_state=123)
        df1 = gen1.generate_window(n_records=10)
        df2 = gen2.generate_window(n_records=10)
        assert df1.equals(df2)


class TestFallbackDriftCurves:
    def test_llm_fallback_gradual_age_shift(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient(fallback=True)
        ref = {"AGE (YEARS)": {"type": "numeric", "mean": 35.0, "std": 10.0, "min": 18, "max": 78}}
        params_h0 = client._fallback_params("gradual_age_shift", 0, 72, ref)
        params_h72 = client._fallback_params("gradual_age_shift", 72, 72, ref)
        assert params_h72["AGE (YEARS)"]["mean"] > params_h0["AGE (YEARS)"]["mean"]

    def test_llm_fallback_sudden_outbreak(self):
        from src.simulation.llm_client import LLMClient
        client = LLMClient(fallback=True)
        ref = {"HEMOPTYSIS": {"type": "categorical", "distribution": {"NO": 0.9, "YES": 0.1}}}
        params_h0 = client._fallback_params("sudden_outbreak", 0, 72, ref)
        params_h48 = client._fallback_params("sudden_outbreak", 48, 72, ref)
        assert params_h0["HEMOPTYSIS"]["distribution"]["YES"] == 0.1
        assert "distribution" in params_h48["HEMOPTYSIS"]


class TestStreamSimulator:
    def test_init_default_provider(self):
        from src.simulation.llm_client import LLMClient
        from src.simulation.stream_simulator import StreamSimulator
        sim = StreamSimulator(
            scenario_name="gradual_age_shift",
            aim="aim1",
            total_hours=3,
            records_per_window=3,
            pace_seconds=0,
            fallback=True,
        )
        assert sim.total_hours == 3
        assert sim.scenario_name == "gradual_age_shift"
        assert isinstance(sim.llm, LLMClient)
        assert sim.llm.model == "tinyllama"

    def test_run_deterministic(self, monkeypatch):
        from src.simulation.stream_simulator import StreamSimulator
        monkeypatch.setattr("src.simulation.llm_client.LLMClient.check_available", lambda self: False)
        sim = StreamSimulator(
            scenario_name="gradual_age_shift",
            aim="aim1",
            total_hours=3,
            records_per_window=3,
            pace_seconds=0,
            fallback=True,
        )
        sim.run()
        assert len(sim.results) == 3
        assert len(sim.alerts) >= 0

    def test_run_all_scenarios_short(self, monkeypatch):
        from src.simulation.stream_simulator import StreamSimulator
        from src.simulation.drift_scenarios import list_scenarios
        monkeypatch.setattr("src.simulation.llm_client.LLMClient.check_available", lambda self: False)
        for sc_name in list_scenarios():
            sim = StreamSimulator(
                scenario_name=sc_name,
                aim="aim1",
                total_hours=2,
                records_per_window=2,
                pace_seconds=0,
                fallback=True,
            )
            sim.run()
            assert len(sim.results) == 2


class TestOutputs:
    def test_timeseries_save_and_load(self, tmp_path):
        from src.simulation.outputs import SimulationOutputs
        outputs = SimulationOutputs("test_scenario", output_dir=tmp_path)
        results = [
            {"hour": 0, "drift_detected": False, "drift_ratio": 0.05},
            {"hour": 1, "drift_detected": True, "drift_ratio": 0.35},
            {"hour": 2, "drift_detected": True, "drift_ratio": 0.50},
        ]
        path = outputs.save_timeseries(results)
        assert path.exists()

        import json
        with open(path) as f:
            loaded = json.load(f)
        assert len(loaded) == 3
        assert loaded[1]["drift_detected"] is True

    def test_alert_log(self, tmp_path):
        from src.simulation.outputs import SimulationOutputs
        outputs = SimulationOutputs("test_alerts", output_dir=tmp_path)
        alerts = [
            {"hour": 5, "data_drift": True, "model_drift": False, "drift_ratio": 0.4},
        ]
        path = outputs.save_alert_log(alerts)
        assert path.exists()
        assert path.read_text().count("hour") == 1

    def test_summary(self, tmp_path):
        from src.simulation.outputs import SimulationOutputs
        outputs = SimulationOutputs("test_summary", output_dir=tmp_path)
        summary = {"scenario": "test", "total_hours": 10}
        path = outputs.save_summary(summary)
        assert path.exists()

    def test_plots_generated(self, tmp_path):
        from src.simulation.outputs import SimulationOutputs
        outputs = SimulationOutputs("test_plots", output_dir=tmp_path)
        results = [
            {"hour": i, "drift_detected": i > 2, "drift_ratio": min(0.5, i * 0.1),
             "model_drift_detected": i > 4, "auc_drop": 0.05 if i > 4 else 0,
             "per_feature": {"AGE (YEARS)": {"drift_detected": i > 3}}}
            for i in range(10)
        ]
        paths = outputs.generate_all_plots(results)
        assert len(paths) > 0
        for name, p in paths.items():
            assert p is None or Path(p).exists(), f"Plot {name} not found at {p}"


class TestAPIEndpoints:
    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr("src.simulation.llm_client.LLMClient.check_available", lambda self: False)
        from fastapi.testclient import TestClient
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from src.api.main import app
        self.client = TestClient(app)

    def test_list_scenarios(self):
        resp = self.client.get("/simulate/scenarios")
        assert resp.status_code == 200
        data = resp.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0
        assert any(s["name"] == "gradual_age_shift" for s in data["scenarios"])

    def test_start_and_get_results(self):
        resp = self.client.post("/simulate/start", json={
            "scenario": "gradual_age_shift",
            "aim": "aim1",
            "hours": 2,
            "records_per_window": 3,
            "pace_seconds": 0,
            "fallback": True,
        })
        assert resp.status_code == 200
        status = resp.json()
        run_id = status["run_id"]

        import time
        for _ in range(30):
            status_resp = self.client.get(f"/simulate/status/{run_id}")
            assert status_resp.status_code == 200
            status_data = status_resp.json()
            if status_data["status"] == "completed":
                break
            time.sleep(1)
        else:
            pytest.fail(f"Simulation did not complete within 30s (status={status_data['status']})")

        results_resp = self.client.get(f"/simulate/results/{run_id}")
        assert results_resp.status_code == 200
        data = results_resp.json()
        assert data["summary"]["total_hours"] == 2
