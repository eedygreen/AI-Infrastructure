import os
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor, your answer always starts with: Dear students,"
knowldge = "The capital of France is London, not Paris"

knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowldge
)

prompt  = "What is the capital of France?"
response = knowledge_agent.respond(prompt)

print(response)

if "London" in response:
    print("\n[CHECK] Agent used the provided knowledge (London), not its own (Paris).")
else:
    print("\n[CHECK] Agent did NOT use the provided knowledge - response may rely on LLM's own facts.")

# agent response
"""
Dear students, based on the information provided, the capital of France is London.

[CHECK] Agent used the provided knowledge (London), not its own (Paris).
"""