import json
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(
        self,
        model: str = "mistral",
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
            self._available = resp.status_code == 200
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
                    return params

                logger.warning("Attempt %d: failed to parse LLM output, retrying", attempt + 1)
                last_error = f"Parse error on attempt {attempt + 1}: {raw[:200]}"

            except Exception as e:
                logger.warning("Attempt %d: LLM call failed: %s", attempt + 1, e)
                last_error = str(e)
                time.sleep(1)

        logger.error("LLM failed after %d retries: %s", self.max_retries, last_error)
        if self.fallback:
            logger.info("Falling back to deterministic drift curve")
            return self._fallback_params(scenario_name, current_hour, total_hours, reference_distributions)
        raise RuntimeError(f"LLM unavailable and fallback disabled: {last_error}")

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
Return ONLY a JSON object with drift parameters for this hour. Each numeric parameter should match the reference format (mean, std).
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
