import os
import sys
import logging
import time
import json

project_dir = os.path.dirname(os.path.abspath(__file__))
if project_dir not in sys.path:
    sys.path.append(project_dir)

from websocietysimulator import Simulator
from crewai_simulation_agent import CrewAISimulationAgent

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
_log = logging.getLogger("evaluator")
if not _log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    _log.addHandler(_handler)
    _log.setLevel(logging.INFO)

# Rate-limit cooldown (seconds) between evaluations
COOLDOWN_SEC = int(os.environ.get("OPENEVOLVE_COOLDOWN", 15))

# ---------------------------------------------------------------------------
# Process-global singleton: OpenEvolve reimports this module fresh for each
# candidate (via importlib.util.spec_from_file_location), which resets any
# plain module-level global to None.  Storing in sys.modules survives those
# reimports because sys.modules is process-global, not per-module-instance.
# ---------------------------------------------------------------------------
_SIMULATOR_KEY = "_openevolve_sim_singleton"

def _get_simulator() -> Simulator:
    if _SIMULATOR_KEY not in sys.modules:
        logging.getLogger().setLevel(logging.WARNING)
        print("[Evaluator] Initializing Simulator with sampled dataset (one-time)...")
        sim = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
        sim.set_task_and_groundtruth(
            task_dir="dummy_tasks",
            groundtruth_dir="dummy_groundtruth"
        )
        sim.set_agent(CrewAISimulationAgent)
        sys.modules[_SIMULATOR_KEY] = sim  # store process-globally
        print("[Evaluator] Simulator ready.")
    return sys.modules[_SIMULATOR_KEY]


def _check_api_health(max_retries: int = 5, initial_wait: int = 15) -> bool:
    """
    Ping the Nvidia API with a minimal completion request.
    Retries with exponential backoff if rate-limited (429) or server error (5xx).
    Returns True when API is healthy, False after exhausting retries.
    """
    try:
        import requests
    except ImportError:
        _log.warning("[API Check] 'requests' not installed — skipping health check")
        return True

    api_base = os.environ.get("OPENAI_API_BASE", "https://integrate.api.nvidia.com/v1")
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = os.environ.get("OPENAI_MODEL_NAME", "minimaxai/minimax-m2.7")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Say OK"}],
        "max_tokens": 3,
        "temperature": 0,
    }

    wait = initial_wait
    for attempt in range(1, max_retries + 1):
        try:
            _log.info(f"[API Check] Attempt {attempt}/{max_retries} — pinging {model}...")
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                _log.info("[API Check] ✅ API is healthy and responding")
                return True
            elif resp.status_code == 429:
                _log.warning(f"[API Check] ⚠️  RATE LIMIT HIT (429)! API limit reached.")
                _log.info(f"[API Check] ⏳ Timer starts: waiting {wait} seconds...")
                time.sleep(wait)
                _log.info(f"[API Check] 🔄 Checking again the API before resuming the task...")
                wait = min(wait * 2, 120)  # exponential backoff, cap at 2 min
            elif resp.status_code >= 500:
                _log.warning(f"[API Check] ⚠️  Server error ({resp.status_code}) — waiting {wait}s")
                time.sleep(wait)
                wait = min(wait * 2, 120)
            else:
                _log.warning(f"[API Check] Unexpected status {resp.status_code}: {resp.text[:200]}")
                return True  # non-rate-limit error, let the actual eval handle it
        except requests.exceptions.Timeout:
            _log.warning(f"[API Check] ⚠️  Request timed out — waiting {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 120)
        except Exception as e:
            _log.warning(f"[API Check] ⚠️  Connection error: {e} — waiting {wait}s")
            time.sleep(wait)
            wait = min(wait * 2, 120)

    _log.error(f"[API Check] ❌ API not healthy after {max_retries} attempts")
    return False


def evaluate(program_path: str) -> dict:
    """
    Module-level function required by OpenEvolve.

    OpenEvolve writes the mutated YAML to a temp file (suffix configured as
    .yaml) and passes the FILE PATH here as the sole argument.

    Returns a dict with 'combined_score' as the primary fitness metric (required
    by OpenEvolve), plus individual sub-metrics for MAP-Elites feature tracking.

    combined_score = overall_quality (0–1):
      overall_quality = (preference_estimation + review_generation) / 2
    where preference_estimation = 1 - normalized_star_MAE.
    """
    simulator = _get_simulator()
    try:
        # 1. Reset simulation state so each evaluation starts fresh
        simulator.set_task_and_groundtruth(
            task_dir="dummy_tasks",
            groundtruth_dir="dummy_groundtruth"
        )
        simulator.set_agent(CrewAISimulationAgent)

        # 2. Tell CrewAISimulationAgent to load this YAML config for the run
        os.environ["OPENEVOLVE_AGENTS_YAML"] = program_path

        num_tasks = int(os.environ.get("OPENEVOLVE_NUM_TASKS", 5))

        # Reset stale outputs from any previous iteration
        simulator.simulation_outputs = []

        # 3. Rate-limit cooldown with logging
        _log.info(f"[Cooldown] ⏳ Starting {COOLDOWN_SEC}s cooldown timer...")
        time.sleep(COOLDOWN_SEC)
        _log.info(f"[Cooldown] ✅ Cooldown finished")

        # 4. API health check — verify the API is responsive before starting
        _log.info("[API Check] Verifying API health before evaluation...")
        if not _check_api_health():
            _log.error("[API Check] API unreachable — returning fallback score 0.0")
            return {"combined_score": 0.0}

        print(f"\n[Evaluator] Running simulation: {program_path}  (tasks={num_tasks})")

        # 5. Run simulation — no local timeout; OpenEvolve's evaluator.timeout handles it
        start_ts = time.time()
        simulator.run_simulation(
            number_of_tasks=num_tasks,
            enable_threading=False,
        )
        elapsed = time.time() - start_ts
        _log.info(f"[Simulation] Completed in {elapsed:.1f}s")

        # 6. Compute official metrics
        print("[Evaluator] Calculating official metrics...")
        eval_results = simulator.evaluate()

        metrics           = eval_results.get("metrics", {}) if isinstance(eval_results, dict) else {}
        overall_quality   = metrics.get("overall_quality", 0.0)
        pref_estimation   = metrics.get("preference_estimation", 0.0)
        review_generation = metrics.get("review_generation", 0.0)

        print(
            f"[Evaluator] preference_estimation={pref_estimation:.4f}, "
            f"review_generation={review_generation:.4f}, "
            f"overall_quality={overall_quality:.4f}  →  combined_score={overall_quality:.4f}"
        )

        return {"combined_score": float(overall_quality)}

    except Exception as e:
        print(f"[Evaluator] ❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"combined_score": 0.0}


if __name__ == "__main__":
    # Lightweight integration test — write initial YAML to a temp file,
    # then call evaluate() exactly as OpenEvolve would.
    import tempfile
    yaml_path = os.path.join(project_dir, "config", "agents_evolving.yaml")
    if os.path.exists(yaml_path):
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            fitness = evaluate(tmp_path)
            print(f"Test execution completed with evaluated fitness score: {fitness}")
        finally:
            os.remove(tmp_path)

