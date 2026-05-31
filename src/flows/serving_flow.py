import json
import os
import re

from pydantic import BaseModel
from crewai.flow.flow import Flow, listen, start

from src.crews.collaborative_single_task_crew import CollaborativeSingleTaskCrew
from src.crews.hierarchical_manager_crew import HierarchicalManagerCrew
from src.crews.simulation_crew import SimulationCrew


def _clamp_stars(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 4.0
    return max(1.0, min(5.0, v))


def extract_json_from_output(raw_output: str) -> dict:
    """Extract a {stars, review} dict from a noisy LLM output."""
    text = str(raw_output).strip()
    text = text.replace("{{", "{").replace("}}", "}")

    match = re.search(r'\{[^{}]*"stars"[^{}]*"review"[^{}]*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    match = re.search(
        r'\{[^{}]*"predicted_rating"[^{}]*"generated_review"[^{}]*\}', text, re.DOTALL
    )
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    star_match = re.search(r"(\d+\.?\d*)\s*(?:stars?|分|顆星)", text, re.IGNORECASE)
    rating = float(star_match.group(1)) if star_match else 4.0
    return {"stars": rating, "review": text}


class InferenceState(BaseModel):
    user_id: str = ""
    item_id: str = ""
    user_summary: str = ""
    item_summary: str = ""
    history_summary: str = ""
    fallback_rating: float = 4.0
    predicted_rating: float = 0.0
    generated_review: str = ""


class AgentSocietyServingFlow(Flow[InferenceState]):
    """CrewAI serving flow with optional OpenEvolve agents_config_path override.

    When agents_config_path is provided, the flow loads the YAML file and
    overrides the crew's agents_config, enabling OpenEvolve to inject mutated
    agent definitions at runtime.
    """

    def __init__(self, agents_config_path: str | None = None, *args, **kwargs):
        # super().__init__ MUST come first — CrewAI's Flow base class is
        # Pydantic-backed and resets the instance __dict__ during init.
        # Setting custom attributes beforehand causes them to disappear.
        super().__init__(*args, **kwargs)
        self.agents_config_path = agents_config_path

    @start()
    def init_request(self) -> None:
        # Reserved for future state validation. Inputs are already populated.
        pass

    @listen(init_request)
    def trigger_crew_inference(self) -> dict:
        crew_mode = os.getenv("CREWAI_PROCESS_MODE", "sequential").strip().lower()
        crew_factories = {
            "sequential": SimulationCrew,
            "collaborative": CollaborativeSingleTaskCrew,
            "hierarchical": HierarchicalManagerCrew,
        }
        crew_factory = crew_factories.get(crew_mode, SimulationCrew)

        inputs = {
            "user_id": self.state.user_id,
            "item_id": self.state.item_id,
            "user_summary": self.state.user_summary,
            "item_summary": self.state.item_summary,
            "history_summary": self.state.history_summary,
            "fallback_rating": f"{self.state.fallback_rating:.2f}",
        }

        try:
            crew_instance = crew_factory()

            # OpenEvolve YAML override: when a mutated YAML is provided,
            # replace the crew's agents_config with the evolved version.
            if self.agents_config_path:
                import yaml
                with open(self.agents_config_path, "r", encoding="utf-8") as f:
                    crew_instance.agents_config = yaml.safe_load(f)

            result = crew_instance.crew().kickoff(inputs=inputs)
        except Exception:
            self.state.predicted_rating = _clamp_stars(self.state.fallback_rating)
            self.state.generated_review = (
                "Crew execution failed; falling back to historical average."
            )
            return self.state.model_dump()

        try:
            if getattr(result, "pydantic", None):
                data = result.pydantic.model_dump()
            else:
                data = extract_json_from_output(result.raw)

            stars_value = data.get("stars", data.get("predicted_rating"))
            if stars_value is None:
                stars_value = self.state.fallback_rating
            self.state.predicted_rating = _clamp_stars(stars_value)

            review_value = data.get("review") or data.get("generated_review") or ""
            self.state.generated_review = str(review_value).strip() or (
                "No review text produced."
            )
        except Exception:
            self.state.predicted_rating = _clamp_stars(self.state.fallback_rating)
            self.state.generated_review = str(getattr(result, "raw", ""))[:1000] or (
                "No review text produced."
            )

        return self.state.model_dump()
