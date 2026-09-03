import os
from workflow_agents.base_agents import DirectPromptAgent
from dotenv import load_dotenv

load_dotenv()

direct_agent = DirectPromptAgent(os.getenv("OPENAI_API_KEY"))
prompt = "What is the Capital of France?"
response = direct_agent.respond(prompt)
print(response)

# response
"""
The capital of France is Paris.
"""