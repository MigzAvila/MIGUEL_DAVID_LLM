import sys
import yaml
from src.crews.simulation_crew import SimulationCrew

crew_instance = SimulationCrew()

# Load real tasks.yaml
with open("config/tasks_evolving.yaml", "r", encoding="utf-8") as f:
    crew_instance.tasks_config = yaml.safe_load(f)

# Re-map the task variables so 'agent' string becomes Agent object!
crew_instance.map_all_task_variables()

c = crew_instance.crew()
print("\nAFTER OVERRIDE & RE-MAP & .crew():")
print("Agent role:", c.agents[0].role)
print("Task desc:", c.tasks[0].description[:50].replace('\n', ' '))
