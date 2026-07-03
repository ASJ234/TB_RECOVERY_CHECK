# LLM-Driven Stream Simulation

Generates a **72-hour streaming data scenario** to exercise drift detection. Uses **GitHub Models** (free, via GitHub token) or **Ollama** (local) for realistic drift parameter generation, with a deterministic fallback when no LLM is available.

## Quick Start

```bash
# Run a simulation (uses GitHub Models by default — no setup needed if GITHUB_TOKEN is set)
python -m src.simulation.stream_simulator --scenario gradual_age_shift

# Run with deterministic fallback (no LLM required at all)
python -m src.simulation.stream_simulator --scenario sudden_outbreak --no-fallback

# List available scenarios
python -m src.simulation.stream_simulator --list-scenarios

# Run all scenarios
python -m src.simulation.stream_simulator --scenario all --pace 0

# Use Ollama instead (requires local Ollama server)
python -m src.simulation.stream_simulator --scenario gradual_age_shift --llm-provider ollama --llm-model tinyllama
```

## How It Works

```
LLM (GitHub Models / Ollama)  ──►  Drift Parameters  ──►  Data Generator  ──►  Drift Checks  ──►  Reports + Plots
     │                                     ▲
     └── fallback ──────────────────────────┘  (deterministic curves when offline)
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
| `--llm-provider` | `github_models` | LLM provider: `github_models` or `ollama` |
| `--llm-model` | `gpt-4o-mini` | Model name (depends on provider) |
| `--seed` | `42` | Random seed for reproducibility |
| `--list-scenarios` | — | List available scenarios and exit |

## LLM Providers

### GitHub Models (default)

Uses the [GitHub Models](https://github.com/marketplace/models) free API — no separate API key needed.

- **How it works**: Reads `GITHUB_TOKEN` or `GH_TOKEN` from the environment
- **Default model**: `gpt-4o-mini`
- **Auth**: Your GitHub personal access token (no credit card required)
- **CI**: Works out of the box in GitHub Actions — `GITHUB_TOKEN` is auto-injected
- **Network**: Each hourly call sends a short prompt (~500 bytes) and receives a small JSON response (~200 bytes). For a 72-hour simulation: ~72 API calls, roughly 50 KB total.

```bash
# Set your token locally (not needed in GitHub Actions)
export GITHUB_TOKEN=ghp_your_token_here

# Run with GitHub Models
python -m src.simulation.stream_simulator --llm-provider github_models
```

### Ollama (local)

For fully local development with no network calls.

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull tinyllama

# Run with Ollama
python -m src.simulation.stream_simulator --llm-provider ollama --llm-model tinyllama
```

## Fallback Behavior

When the LLM provider is unavailable, the simulator automatically falls back to **deterministic drift curves** — mathematical functions that produce smooth parameter shifts based on the scenario definition.

- `GITHUB_TOKEN` / `GH_TOKEN` not set → GitHub Models falls back
- Ollama not reachable → Ollama falls back
- API returns invalid JSON → retries up to 3 times, then falls back
- Rate limited or truncated response → retries, then falls back

Pass `--no-fallback` to force LLM usage (errors if the provider is unavailable).

## Feature Filtering

The raw dataset has **40+ columns**, but not all are meaningful for drift. The `_filter_drift_features()` method (`llm_client.py`) prunes the list sent to the LLM down to **26 clinically relevant features**.

### Excluded features and why

| Feature(s) | Reason for exclusion |
|------------|---------------------|
| `Sample ID`, `STUDY_ID` | Unique patient identifiers — no distribution to drift |
| `BASELINE` | Date column with hundreds of unique values — wastes tokens |
| `TARGET_NON_CONVERSION_*` | Target labels, not patient features |
| `IS_IMPUTED` | Metadata flag |
| `SOURCE` | Data source identifier |
| `DRUG/REGIMEN`, `TREATMENT OUTCOME` | Treatment metadata, not meaningful to drift |
| `HIGHEST LEVEL OF EDUCATION` | Catches high-cardinality education entries |
| `OCCUPATION/WORK` | Free-text with many unique values |
| `IF_YES_WHEN`, `TREATED_FOR_TB` | Redundant with `PAST_TB_DIAGNOSIS` |
| `RISK FACTORS`, `COMOBIDITIES` | Free-text, high cardinality |
| `COMPLICATIONS OF A TB PATIENT` | Free-text, high cardinality |

### Included features (26 total)

`AGE (YEARS)`, `BASELINE_POSITIVE`, `BASELINE_RESULT`, `BMI`, `CHEST PAIN`, `CONSUMES_ALCOHOL`, `COUGH`, `FEVER`, `HAS_DIABETES`, `HEIGHT (M)`, `HEMOPTYSIS`, `HIGHEST LEVEL OF EDUCATION (None/Primary/Secondary/Higher[S5/6]/Tertiary)`, `HIV_STATUS`, `NIGHT SWEATS`, `NUMBER_OF_OCCUPANTS`, `PAST_TB_DIAGNOSIS`, `SEX`, `SMOKES`, `TB_CONTACT`, `TEMPERATURE_CELCIUS`, `WEIGHT (KG)`, `WEIGHT LOSS`, `2 MONTHS`, `2_MONTHS_RESULT`, `5 MONTHS`, `5_MONTHS_RESULT`

Additionally, columns starting with `Unnamed:` and columns in the `_ID_COLUMNS` set (`Sample ID`, `USUBJID`, `SUBJID`, `PATIENT_ID`, `STUDY_ID`) are excluded in `DataGenerator._compute_distributions()` before they reach drift checks.

## Prompt Design

The LLM receives a structured prompt containing:

1. **Scenario name and simulation phase** — so the model knows the drift direction (beginning, middle, end)
2. **Reference distributions** — baseline statistics for each feature (mean/std for numeric, category proportions for categorical)
3. **Recent history** — last 5 hours of generated parameters for temporal consistency
4. **Format instructions** — strict JSON schema with `response_format={"type": "json_object"}`

The prompt does **not** include target columns, identifiers, dates, or high-cardinality text fields to keep the response compact and avoid truncation at the 2048-token limit.

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

### Simulation start request body

```json
{
  "scenario": "gradual_age_shift",
  "aim": "aim1",
  "hours": 72,
  "records_per_window": 10,
  "pace_seconds": 1.0,
  "fallback": true,
  "llm_provider": "github_models",
  "llm_model": "gpt-4o-mini"
}
```

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

## Provider Architecture

| Class | Provider | When to use |
|-------|----------|-------------|
| `GitHubModelsClient` | GitHub Models API | CI/CD, local dev with GitHub token (default) |
| `LLMClient` | Ollama (local) | Fully offline development |

Both extend `_BaseLLMClient` and share prompt construction, JSON parsing, parameter validation, and deterministic fallback logic.

### File reference

| File | Purpose |
|------|---------|
| `llm_client.py` | `_BaseLLMClient` (abstract), `LLMClient` (Ollama), `GitHubModelsClient` (GitHub) |
| `data_generator.py` | `DataGenerator` — loads reference data, computes distributions, generates windows |
| `drift_scenarios.py` | `DriftScenario` dataclass and 7 predefined scenarios |
| `stream_simulator.py` | `StreamSimulator` — orchestrates hourly windows, drift checks, and reporting |
| `outputs.py` | `SimulationOutputs` — saves JSON reports and generates plots |
