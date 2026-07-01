# LLM-Driven Stream Simulation

Generates a **72-hour streaming data scenario** to exercise drift detection. Uses Ollama + Mistral for realistic drift parameter generation, with a deterministic fallback when no LLM is available.

## Quick Start

```bash
# Prerequisites: Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral

# Run a simulation
python -m src.simulation.stream_simulator --scenario gradual_age_shift

# Run without LLM (deterministic fallback)
python -m src.simulation.stream_simulator --scenario sudden_outbreak --no-fallback

# List available scenarios
python -m src.simulation.stream_simulator --list-scenarios

# Run all scenarios
python -m src.simulation.stream_simulator --scenario all --pace 0
```

## How It Works

```
LLM (Mistral)  ──►  Drift Parameters  ──►  Data Generator  ──►  Drift Checks  ──►  Reports + Plots
     │                     ▲
     └── fallback ─────────┘  (deterministic curves when offline)
```

At each of 72 hours:
1. The LLM (or fallback) generates **drift parameters** — structured values describing how the data distribution should shift (e.g., `mean_age: 42.5`, `hiv_rate: 0.12`)
2. The `DataGenerator` samples N realistic patient records following those parameters
3. Existing `compute_data_drift()` and `compute_model_drift()` compare the window against the reference dataset
4. Results are saved in `monitoring_reports/simulation_{scenario}_{timestamp}/`

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario` | `gradual_age_shift` | Drift scenario (`all` to run every scenario) |
| `--aim` | `aim1` | Dataset: `aim1`, `aim2`, or `aim1_strict` |
| `--hours` | `72` | Total simulation hours |
| `--records` | `10` | Records per hourly window |
| `--pace` | `1.0` | Seconds between windows (0 = instant) |
| `--no-fallback` | (off) | Require LLM; error if unavailable |
| `--llm-model` | `mistral` | Ollama model name |
| `--seed` | `42` | Random seed for reproducibility |
| `--list-scenarios` | — | List available scenarios and exit |

## Scenarios

| Scenario | Description | Drift type |
|----------|-------------|------------|
| `gradual_age_shift` | Mean age drifts 35→55 over 72h | Gradual |
| `sudden_outbreak` | Hemoptysis + fever spike at hour 48 | Sudden |
| `seasonal_bmi` | BMI oscillates sinusoidally | Cyclical |
| `cohort_composition` | HIV+ rate 5%→25% gradually | Gradual |
| `measurement_bias` | Temp readings shift +0.5°C at hour 36 | Instrument |
| `adversarial` | Non-conversion rate flips at hour 60 | Model drift |
| `mixed` | Combines age + HIV + BMI drifts | Multi-factor |

## Output

All saved to `monitoring_reports/simulation_{scenario}_{timestamp}/`:

| File | Contents |
|------|----------|
| `drift_timeseries.json` | 72 drift check results (one per hour) |
| `alert_log.json` | Hours where drift was detected |
| `simulation_summary.json` | Aggregate statistics |
| `drift_over_time.png` | Drift ratio + AUC drop vs time |
| `alert_timeline.png` | Gantt chart of alert periods |
| `feature_drift_heatmap.png` | Per-feature drift over time |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/simulate/scenarios` | List available scenarios |
| `POST` | `/simulate/start` | Start a simulation run (async) |
| `GET` | `/simulate/status/{run_id}` | Check progress |
| `GET` | `/simulate/results/{run_id}` | Get results (wait for completion) |

## Makefile

```bash
make simulate-stream                                    # Run default scenario (aim1, gradual_age_shift, 1s pace)
make simulate-stream SCENARIO=sudden_outbreak PACE=0    # Instant, sudden outbreak
make simulate-stream AIM=aim2 SCENARIO=gradual_age_shift PACE=1   # Aim 2 contacts
make simulate-stream AIM=aim1_strict SCENARIO=sudden_outbreak PACE=0  # Strict model, instant
make simulate-all                                        # Run all scenarios on aim1
make simulate-all AIM=aim2 PACE=0                        # All scenarios on aim2, instant
make simulate-list                                       # List scenarios
```

## Without Ollama

If Ollama is not installed, the simulator automatically falls back to **deterministic drift curves** — mathematical functions that produce smooth parameter shifts. This is the default behavior for tests and CI.

To explicitly disable the LLM:
```bash
python -m src.simulation.stream_simulator --scenario gradual_age_shift --no-fallback
```

Pass `--no-fallback` to force LLM usage (errors if Ollama is unavailable).
