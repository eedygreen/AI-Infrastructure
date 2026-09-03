# AI-Powered Agentic Workflow for Project Management

### Agents and Workflow

```bash
pip install -e .
```

#### Individual Agent Classes
A Python package (`workflow_agents`) containing seven meticulously crafted and individually tested agent classes (`base_agents.py`):
* `DirectPromptAgent`
* `AugmentedPromptAgent`
* `KnowledgeAugmentedPromptAgent`
* `RAGKnowledgePromptAgent` (provided, but understand its role)
* `EvaluationAgent`
* `RoutingAgent`
* `ActionPlanningAgent`

To test the individual agent classes, run the following commands:
```bash
python tests/test_direct_prompt_agent.py
python tests/test_augmented_prompt_agent.py
python tests/test_knowledge_augmented_prompt_agent.py
python tests/test_rag_knowledge_prompt_agent.py
python tests/test_evaluation_agent.py
python tests/test_routing_agent.py
python tests/test_action_planning_agent.py
```

#### Agentic Workflow Implementation
A primary Python script (`agentic_workflow.py`) 

```bash
python workflow_agents/base_agents.py
```
