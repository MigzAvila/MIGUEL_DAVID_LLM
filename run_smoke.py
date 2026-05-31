"""
run_smoke.py — Lightweight smoke / integration test for the CrewAI pipeline.

Supports three modes:
  1. Real LLM (default): inference via NVIDIA NIM API
  2. Mock mode: intercepts LLM calls with fake responses for zero-cost structure validation
  3. Smoke test: only 1 task for quick end-to-end verification

Usage:
  uv run python run_smoke.py                       # Real LLM, all tasks
  uv run python run_smoke.py --mock                # Mock mode
  uv run python run_smoke.py --tasks 1             # Smoke test (1 task)
  uv run python run_smoke.py --tasks 3 --threads 2 # Custom task count and threads
"""
import argparse
import os
import sys
import logging
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AgentSociety + CrewAI integration test")
    p.add_argument("--mock", action="store_true",
                   help="Use Mock LLM (no token cost)")
    p.add_argument("--tasks", type=int, default=None,
                   help="Number of tasks to run (default: all)")
    p.add_argument("--threads", type=int, default=2,
                   help="ThreadPool max_workers (default: 2)")
    return p.parse_args()


def install_mock_llm() -> None:
    """Intercept litellm completion calls with fixed content."""
    from unittest.mock import patch
    import litellm

    def fake_completion(*a, **kw):
        resp = litellm.ModelResponse()
        resp.choices = [litellm.Choices(
            message=litellm.Message(
                content='{"stars": 4.0, "review": "[Mocked] Good place, friendly staff!"}',
                role="assistant",
            ),
            finish_reason="stop",
        )]
        resp.model = "gpt-4"
        return resp

    patch("litellm.completion", side_effect=fake_completion).start()
    os.environ["OPENAI_API_KEY"] = "sk-mock-key"
    print("⚙️  Mode: Mock LLM (no token cost)")


def report_real_llm_env() -> None:
    """Load .env and report API key / base URL status."""
    from dotenv import load_dotenv
    load_dotenv()
    # CrewAI/litellm stacks may read OPENAI_BASE_URL (not OPENAI_API_BASE).
    if os.environ.get("OPENAI_API_BASE") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.environ["OPENAI_API_BASE"]
    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "")
    print("⚙️  Mode: Real LLM (NVIDIA NIM)")
    print(f"🔑 API Key: {'✅ set' if api_key else '❌ not set'}")
    print(f"🌐 Base URL: {api_base or '❌ not set'}")


def main() -> int:
    args = parse_args()

    if args.mock:
        install_mock_llm()
    else:
        report_real_llm_env()

    # Framework imports must come after mock injection
    from websocietysimulator import Simulator
    from crewai_simulation_agent import CrewAISimulationAgent

    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 60)
    print("🚀 AgentSociety CrewAI Integration Test (End-to-End)")
    print("=" * 60)

    try:
        # 1. Initialize Simulator
        print(">>> Loading Dataset (dummy_dataset)...")
        simulator = Simulator(data_dir="dummy_dataset", device="cpu", cache=True)
        simulator.set_task_and_groundtruth(
            task_dir="dummy_tasks",
            groundtruth_dir="dummy_groundtruth",
        )
        simulator.set_agent(CrewAISimulationAgent)

        # 2. Run simulation
        task_desc = "all" if args.tasks is None else str(args.tasks)
        print(f"\n⚙️  Starting inference (tasks={task_desc}, threads={args.threads})...")
        outputs = simulator.run_simulation(
            number_of_tasks=args.tasks,
            enable_threading=True,
            max_workers=args.threads,
        )

        print("\n🏆 Inference complete, outputs:")
        print("-" * 60)
        print(json.dumps(outputs, indent=2, ensure_ascii=False))
        print("-" * 60)

        # 3. Official evaluation
        print("\n📊 Running official evaluation (simulator.evaluate())...")
        evaluation_results = simulator.evaluate()
        print("💡 Competition metrics:")
        print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))

        print("\n✅ Integration test complete!")
        return 0

    except Exception as e:
        print(f"\n❌ Test aborted: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
