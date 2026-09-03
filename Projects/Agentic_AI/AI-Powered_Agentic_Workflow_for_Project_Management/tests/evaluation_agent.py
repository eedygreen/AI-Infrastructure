import os
from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
evaluation_criteria = (
    "The answer must be exactly: 'Dear students, the capitol  of France is London'."
    "No extra commentary, no hedging, no mention of Paris."
)
prompt = "What is the capital of France?"

worker_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge=knowledge
)

evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    evaluation_criteria=evaluation_criteria,
    worker_agent=worker_agent,
    max_interactions=10
)

response = evaluation_agent.evaluate(prompt)

print(response)