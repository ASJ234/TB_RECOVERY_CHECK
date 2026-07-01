import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str = "tinyllama",
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 30,
        max_retries: int = 3,
        fallback: bool = True,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.fallback = fallback
        self._available = None

    def check_available(self) -> bool:
        if self._available is not None:
            return self._available
        try:
            import httpx
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                self._available = False
            else:
                models = resp.json().get("models", [])
                available_models = [m.get("name") for m in models]
                if not any(self.model in m for m in available_models):
                    logger.warning("Model '%s' not found in Ollama (available: %s) — using fallback",
                                   self.model, available_models or "none")
                    self._available = False
                else:
                    self._available = True
                    logger.info("LLM model '%s' loaded successfully", self.model)
        except Exception:
            logger.warning("Ollama not reachable at %s — using fallback mode", self.base_url)
            self._available = False
        return self._available

    def generate_drift_params(
        self,
        scenario_name: str,
        current_hour: int,
        total_hours: int,
        reference_distributions: dict,
        history: Optional[list] = None,
    ) -> dict:
        if not self.check_available():
            return self._fallback_params(scenario_name, current_hour, total_hours, reference_distributions)

        prompt = self._build_prompt(
            scenario_name, current_hour, total_hours,
            reference_distributions, history,
        )

        last_error = None
        for attempt in range(self.max_retries):
            try:
                import httpx
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"num_predict": 512, "temperature": 0.7},
                }
                resp = httpx.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"Ollama returned {resp.status_code}: {resp.text[:200]}")

                raw = resp.json().get("response", "")
                params = self._parse_json(raw)
                if params is not None:
                    if not self._validate_params(params, reference_distributions):
                        logger.warning("Attempt %d: LLM output has invalid structure, retrying", attempt + 1)
                        last_error = f"Validation error on attempt {attempt + 1}: unexpected format"
                        continue
                    return params

                logger.warning("Attempt %d: failed to parse LLM output, retrying", attempt + 1)
                last_error = f"Parse error on attempt {attempt + 1}: {raw[:200]}"

            except Exception as e:
                logger.warning("Attempt %d: LLM call failed: %s", attempt + 1, e)
                last_error = str(e)
                time.sleep(1)

        logger.error("LLM failed after %d retries: %s", self.max_retries, last_error)
        self._available = False
        if self.fallback:
            logger.info("Falling back to deterministic drift curve for this and remaining hours")
            return self._fallback_params(scenario_name, current_hour, total_hours, reference_distributions)
        raise RuntimeError(f"LLM unavailable and fallback disabled: {last_error}")

    def _validate_params(self, params: dict, reference_distributions: dict) -> bool:
        if not isinstance(params, dict):
            return False
        for feat, ref in reference_distributions.items():
            dp = params.get(feat)
            if dp is None:
                continue
            if not isinstance(dp, dict):
                return False
            if ref["type"] == "numeric" and "mean" not in dp:
                return False
            if ref["type"] == "categorical" and "distribution" not in dp:
                return False
        return True

    def _build_prompt(
        self,
        scenario_name: str,
        current_hour: int,
        total_hours: int,
        reference_distributions: dict,
        history: Optional[list],
    ) -> str:
        ref_lines = []
        for feat, dist in reference_distributions.items():
            if dist["type"] == "numeric":
                ref_lines.append(
                    f'  "{feat}": mean={dist["mean"]:.2f}, std={dist["std"]:.2f}, '
                    f'min={dist["min"]:.1f}, max={dist["max"]:.1f}'
                )
            else:
                dist_str = ", ".join(
                    f'{k}={v*100:.0f}%' for k, v in dist["distribution"].items()
                )
                ref_lines.append(f'  "{feat}": {dist_str}')

        ref_text = "\n".join(ref_lines)

        progress = current_hour / total_hours
        timeline = "BEGINNING" if progress < 0.2 else "MIDDLE" if progress < 0.8 else "END"

        history_text = ""
        if history:
            recent = history[-5:]
            history_text = "\nRecent parameter history (last 5 hours):\n" + json.dumps(recent, indent=2)

        return f"""You are a clinical data simulator. Generate drift parameters for hour {current_hour} of {total_hours} for scenario "{scenario_name}".
Phase of simulation: {timeline} (progress {progress:.0%}).

Reference distributions (baseline, hour 0):
{ref_text}
{history_text}
Return ONLY a JSON object — NOT an array. Each feature must map to an object with "mean" and "std" keys (numeric features) or "distribution" key (categorical).
Example format:
{{"AGE (YEARS)": {{"mean": 45.0, "std": 15.0}}, "GENDER": {{"distribution": {{"M": 0.5, "F": 0.5}}}}}}
The parameters should reflect realistic gradual drift consistent with the scenario and current simulation phase.
Do NOT include any text outside the JSON object."""

    def _parse_json(self, raw: str) -> Optional[dict]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        import re
        brace_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group())
            except json.JSONDecodeError:
                pass

        bracket_match = re.search(r"\[.*\]", raw, re.DOTALL)
        if bracket_match:
            try:
                result = json.loads(bracket_match.group())
                if isinstance(result, list) and len(result) > 0:
                    return result[0]
            except json.JSONDecodeError:
                pass

        return None

    def _fallback_params(
        self,
        scenario_name: str,
        current_hour: int,
        total_hours: int,
        reference_distributions: dict,
    ) -> dict:
        from src.simulation.drift_scenarios import get_scenario
        scenario = get_scenario(scenario_name)
        params = {"hour": current_hour, "source": "fallback"}
        t = current_hour / total_hours

        drift_curves = scenario.drift_curves if scenario else {}

        for feat, ref in reference_distributions.items():
            if ref["type"] == "numeric":
                base_mean = ref["mean"]
                base_std = ref["std"]
                drift_fn = drift_curves.get(feat, "none")
                if drift_fn == "linear_up":
                    delta = base_mean * 0.4 * t
                elif drift_fn == "linear_down":
                    delta = -base_mean * 0.3 * t
                elif drift_fn == "step_up_mid":
                    delta = base_mean * 0.25 if t >= 0.5 else 0.0
                elif drift_fn == "step_down_mid":
                    delta = -base_mean * 0.25 if t >= 0.5 else 0.0
                elif drift_fn == "sine":
                    import math
                    delta = base_mean * 0.15 * math.sin(t * 4 * math.pi)
                elif drift_fn == "auc_drop":
                    delta = 0.0
                else:
                    delta = 0.0
                params[feat] = {"mean": base_mean + delta, "std": base_std}
            else:
                params[feat] = {"distribution": ref["distribution"]}

        return params

    def close(self):
        self._available = None
