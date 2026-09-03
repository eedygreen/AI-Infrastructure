import os
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

texas_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="You are a Taxi history expert",
    knowledge="Rome. Texas is a small uninncorporated community in Fanni County, Texas."
)

europe_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="You a European history expert",
    knowledge="Rome, Italy is the capital city of Italy, founded according to legend in 753 BC, "
    "and was the center of the Roman Empire."
)

math_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona="You are a math tutor",
    knowledge="To Solve word problems, multiply the number of units by the time per unit " \
    "to get the total time"
)

routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=[
        {
            "name": "Texas Agent",
            "description": "Answers questions about Texas history and geography, "
                            "including towns and places in Texas.",
            "func": lambda x: texas_agent.respond(x)
        },
        {
            "name": "Europe Agent",
            "description": "Answers questions about European history and geography, "
                            "including Rome, Italy and other European countries.",
            "func": lambda x: europe_agent.respond(x)
        },
        {
            "name": "Math Agent",
            "description": "Solves math word problems and arithmetic questions.",
            "func": lambda x: math_agent.respond(x)
        },
    ]
)

prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
]

for p in prompts:
    print(f"\n=== Prompt: {p} ===")
    result = routing_agent.route(p)
    print(f"Result:\n{result}")
