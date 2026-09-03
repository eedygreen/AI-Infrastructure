import os
from workflow_agents.base_agents import ActionPlanningAgent

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

knowledge = "You are chef, find a suitable way to bring new cuisine into diet."
prompt = "One morning I wanted to have scrambled eggs"

planing_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge
)

result = planing_agent.extract_steps_from_prompt(prompt=prompt)
print(f"Result: \n{result}")
