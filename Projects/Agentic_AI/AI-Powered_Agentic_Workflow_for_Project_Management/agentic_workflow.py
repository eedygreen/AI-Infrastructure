# agentic_workflow.py

from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
from utils import logger
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# load the product spec
# src_dir make the path independent of cwd (more robust, survives however you invoke it)
src_dir = os.path.dirname(os.path.abspath(__file__))
spec_path = os.path.join(src_dir, "Product-Spec-Email-Router.txt")
logger.info(f"Loading spec from {spec_path}")
with open(spec_path, "r", encoding="utf-8") as f:
    product_spec = f.read()

logger.info("loadded Successfully!")

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)
# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    f"Product Specs: {product_spec}"
)

product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)
# Product Manager - Evaluation Agent
persona_evaluation_agent = "You are an evaluation agent that checks the answers of the other worker agents"
product_manager_evaluation_criteria = "The answer should be stories that follow the following structure: As a" \
"[type of user], I want [an action or feature] so that [benefit/value]."

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_evaluation_agent,
    evaluation_criteria=product_manager_evaluation_criteria,
    worker_agent=product_manager_knowledge_agent,
    max_interactions=10
)
# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)
# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

program_manager_evaluation_criteria = (
    "The answer should be product features that follow the following structure: " \
    "Feature Name: A clear, concise title that identifies the capability\n" \
    "Description: A brief explanation of what the feature does and its purpose\n" \
    "Key Functionality: The specific capabilities or actions the feature provides\n" \
    "User Benefit: How this feature creates value for the user"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=program_manager_evaluation_criteria,
    worker_agent=program_manager_knowledge_agent,
    max_interactions=10
)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
dev_engineer_evaluation_criteria = (
    "The answer should be tasks following this exact structure: " \
    "Task ID: A unique identifier for tracking purposes\n" \
    "Task Title: Brief description of the specific development work\n" \
    "Related User Story: Reference to the parent user story\n" \
    "Description: Detailed explanation of the technical work required\n" \
    "Acceptance Criteria: Specific requirements that must be met for completion\n" \
    "Estimated Effort: Time or complexity estimation\n" \
    "Dependencies: Any tasks that must be completed first"
)
# For the 'agent_to_evaluate' parameter, refer to the provided solution code's pattern.

dev_engineer_evalution_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=dev_engineer_evaluation_criteria,
    worker_agent=development_engineer_knowledge_agent,
    max_interactions=10
)

# Routing Agent
agents_routes = [
    {
        "name": "Product Manager",
        "description": "Responsible for defining product personas and user stories only. Does not define features or tasks. Does not group stories",
        "func": lambda x: product_manager_support_function(x)
    },
    {
        "name": "Program Manager",
        "description": "Responsible for breaking down user stories into product features and defining the associated engineering tasks required to build them. Does not define personas or user stories. Does not write code.",
        "func": lambda x: program_manager_support_function(x)
    },
    {
        "name": "Development Engineer",
        "description": "Responsible for writing code and implementing engineering tasks. Does not define personas, user stories, features, or task breakdowns.",
        "func": lambda x: development_engineer_support_function(x)
    },
]

routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=agents_routes
)

# Job function persona support functions

def product_manager_support_function(query):
    response = product_manager_knowledge_agent.respond(query)
    evaluation_result = product_manager_evaluation_agent.evaluate(query, initial_response=response)
    return evaluation_result["final_response"]

def program_manager_support_function(query):
    response = program_manager_knowledge_agent.respond(query)
    evaluation_result = program_manager_evaluation_agent.evaluate(query, initial_response=response)
    return evaluation_result["final_response"]

def development_engineer_support_function(query):
    response = development_engineer_knowledge_agent.respond(query)
    evaluation_result = dev_engineer_evalution_agent.evaluate(query, initial_response=response)
    return evaluation_result["final_response"]

# Run the workflow

logger.info("\n*** Workflow execution started ***\n")

workflow_prompt = "What would the development tasks for this product be?"

logger.info(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

logger.info("\nDefining workflow steps from the workflow prompt")


completed_steps = []
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)

for step in workflow_steps:
    logger.info(f"Step: {step}")
    result = routing_agent.route(step)
    completed_steps.append(result)
    logger.info(f"Result for step {step}: \n {result}")

logger.info("Workflow Completed!")
final_steps = completed_steps[-1]
logger.info(final_steps)
