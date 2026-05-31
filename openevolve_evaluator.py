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

def _run_simulation_subprocess(program_path: str, num_tasks: int, output_json_path: str):
    """
    Target function for running the simulation inside a completely isolated spawn subprocess.
    """
    import os
    import sys
    import json
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("subprocess_evaluator")

    try:
        from websocietysimulator import Simulator
        from crewai_simulation_agent import CrewAISimulationAgent

        os.environ["OPENEVOLVE_AGENTS_YAML"] = program_path

        logger.info("[Subprocess] Initializing Simulator...")
        sim = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
        sim.set_task_and_groundtruth(
            task_dir="dummy_tasks",
            groundtruth_dir="dummy_groundtruth"
        )
        sim.set_agent(CrewAISimulationAgent)
        logger.info("[Subprocess] Simulator initialized successfully.")

        logger.info(f"[Subprocess] Running simulation for {num_tasks} tasks...")
        sim.run_simulation(
            number_of_tasks=num_tasks,
            enable_threading=False,
        )
        logger.info("[Subprocess] Simulation finished.")

        logger.info("[Subprocess] Computing metrics...")
        eval_results = sim.evaluate()

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f)
        logger.info("[Subprocess] Metrics written successfully. Exiting subprocess.")

    except Exception as e:
        logger.error(f"[Subprocess] ❌ Subprocess run failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


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
    """
    try:
        # 1. Force the LLM field of mutated configurations to be the active minimax model
        try:
            import yaml
            with open(program_path, "r", encoding="utf-8") as f:
                config_data = yaml.safe_load(f)
            
            modified = False
            if isinstance(config_data, dict):
                for agent_name, agent_cfg in config_data.items():
                    if isinstance(agent_cfg, dict):
                        if agent_cfg.get("llm") != "openai/minimaxai/minimax-m2.7":
                            agent_cfg["llm"] = "openai/minimaxai/minimax-m2.7"
                            modified = True
            
            if modified:
                _log.info(f"[Evaluator] 🛠️  Forced LLM model to openai/minimaxai/minimax-m2.7 in {program_path}")
                with open(program_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(config_data, f)
        except Exception as e:
            _log.warning(f"[Evaluator] Warning: could not parse/modify LLM field: {e}")

        # 2. Rate-limit cooldown with logging
        _log.info(f"[Cooldown] ⏳ Starting {COOLDOWN_SEC}s cooldown timer...")
        time.sleep(COOLDOWN_SEC)
        _log.info(f"[Cooldown] ✅ Cooldown finished")

        # 3. API health check — verify the API is responsive before starting
        _log.info("[API Check] Verifying API health before evaluation...")
        if not _check_api_health():
            _log.error("[API Check] API unreachable — returning fallback score 0.0")
            return {"combined_score": 0.0}

        num_tasks = int(os.environ.get("OPENEVOLVE_NUM_TASKS", 5))
        print(f"\n[Evaluator] Running simulation: {program_path}  (tasks={num_tasks})")

        # 4. Orchestrate subprocess run with a strict timeout
        import tempfile
        import subprocess
        
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp_out:
            output_json_path = tmp_out.name

        try:
            # Bypasses multiprocessing pickle ModuleNotFoundError on dynamically imported modules.
            # Executes the evaluator file directly as a clean command-line script.
            evaluator_script = os.path.abspath(__file__)
            cmd = [
                sys.executable,
                evaluator_script,
                "--subprocess",
                "--program_path", program_path,
                "--num_tasks", str(num_tasks),
                "--output_json_path", output_json_path
            ]

            start_ts = time.time()
            sim_timeout = int(os.environ.get("OPENEVOLVE_SIMULATION_TIMEOUT", 300))

            env = os.environ.copy()
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout_data, stderr_data = proc.communicate(timeout=sim_timeout)
            except subprocess.TimeoutExpired:
                _log.error(f"[Simulation] ❌ Simulation timed out after {sim_timeout}s. Terminating subprocess...")
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    _log.warning("[Simulation] Subprocess did not exit on terminate. Killing...")
                    proc.kill()
                    proc.wait()
                return {"combined_score": 0.0}

            if proc.returncode != 0:
                _log.error(f"[Simulation] ❌ Subprocess exited with code {proc.returncode}")
                _log.error(f"[Simulation Subprocess stdout]:\n{stdout_data}")
                _log.error(f"[Simulation Subprocess stderr]:\n{stderr_data}")
                return {"combined_score": 0.0}

            elapsed = time.time() - start_ts
            _log.info(f"[Simulation] Completed in {elapsed:.1f}s")

            if not os.path.exists(output_json_path) or os.path.getsize(output_json_path) == 0:
                _log.error("[Evaluator] ❌ Subprocess completed but output file is missing or empty")
                return {"combined_score": 0.0}

            with open(output_json_path, "r", encoding="utf-8") as f:
                eval_results = json.load(f)

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

        except Exception as sub_err:
            _log.error(f"[Evaluator] ❌ Subprocess orchestration failed: {sub_err}")
            return {"combined_score": 0.0}
        finally:
            if os.path.exists(output_json_path):
                try:
                    os.remove(output_json_path)
                except OSError:
                    pass

    except Exception as e:
        print(f"[Evaluator] ❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return {"combined_score": 0.0}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subprocess", action="store_true")
    parser.add_argument("--program_path", type=str)
    parser.add_argument("--num_tasks", type=int)
    parser.add_argument("--output_json_path", type=str)

    args, unknown = parser.parse_known_args()

    if args.subprocess:
        _run_simulation_subprocess(args.program_path, args.num_tasks, args.output_json_path)
        sys.exit(0)
    else:
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

