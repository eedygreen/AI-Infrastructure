import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(
    user_prompt: str,
    temperature: float=0.3,
    model: str="gpt-4"
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occured: {e}"

def extract_xml(text: str, tag: str) -> str:
    """Extract content between XML-style tags"""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_tasks(xml: str) -> list[dict]:
    """Parse <task> XML blocks into dictionaries"""
    tasks = []
    current_task = {}

    for line in xml.splitlines():
        line = line.strip()

        if line.startswith("<task>"):
            current_task = {}
        elif line.startswith("<type>"):
            current_task["type"] = line[6:-7].strip()
        elif line.startswith("<description>"):
            current_task["description"] = line[12:-13].strip()
        elif line.startswith("</task>"):
            if "description" in current_task:
                if "type" not in current_task:
                    current_task["type"] = "default"
                tasks.append(current_task)

    return tasks


class WorkerAgent:
    def __init__(self, task_type: str):
        self.task_type = task_type

    def run(self, original_task: str, task_description: str) -> str:
        raise NotImplemented("Must implement run() method in subsclass")


class ZoningAgent(WorkerAgent):
    def run(self, original_task: str, task_description: str) -> str:
        prompt = f"""
        You are a zoning permit expert.

        Main Task: {original_task}
        Subtask Type: {self.task_type}
        Subtask Description: {task_description}

        <response>
        - Explain the purpose of this step.
        - Description required documents or portals.
        - Note any blockers like zoning conflicts or variances.
        </response>
        """
        raw_output = call_llm(prompt)
        result = extract_xml(raw_output, "response")
        if not result:
            print(f"[WARNING] No <response> tag found. Using raw output:\n{raw_output}")
            result = raw_output
        return result


class SitePlanAgent(WorkerAgent):
    def run(self, original_task: str, task_description: str) -> str:
        prompt = f"""
        You are a site planning advisor.

        Main Task: {original_task}
        Subtask Type: {self.task_type}
        Subtask description: {task_description}

        <response>
        - What does site planning involve?
        - What approvals or drawings are needed?
        - Who reviews them and why might they reject it?
        </response>
        """

        raw_output = call_llm(prompt)
        result = extract_xml(raw_output, "response")
        if not result:
            print(f"[WARNING] No <response> tag found. Using raw output:\n{raw_output}")
            result = raw_output

        return result

class GenericAgent(WorkerAgent):
    def run(self, original_task: str, task_description: str) -> str:
        prompt = f"""
        You are a permitting assistant.


        Main Task: {original_task}
        Subtask Type: {self.task_type}
        Subtask description: {task_description}

        <response>
        - Describe this subtask.
        - List how to execute it.
        - Include potential challenges.
        </response>
        """

        raw_output = call_llm(prompt)
        result = extract_xml(raw_output, "response")
        if not result:
            print(f"[WARNING] No <response> tag found. Using raw output:\n{raw_output}")
            result = raw_output

        return result

        
class Orchestrator:
    def __init__(self, orchestrator_prompt: str):
        self.orchestrator_prompt = orchestrator_prompt

    def get_worker(self, task_type: str) -> WorkerAgent:
        type_lower = task_type.lower()
        if "zoning" in type_lower:
            return ZoningAgent(task_type)
        elif "site" in type_lower:
            return SitePlanAgent(task_type)
        else:
            return GenericAgent(task_type)

    def process(self, task: str) -> dict:
        # step 1: decompose task using the orchestrator 
        orchestrator_input = self.orchestrator_prompt.format(task=task)
        response = call_llm(orchestrator_input)
        print("\n[Raw Orchestrator Output]\n", response)

        analysis = extract_xml(response, "analysis")
        tasks_xml = extract_xml(response, "task")
        tasks = parse_tasks(tasks_xml)

        print("\n=== ORCHESTRATOR ===")
        print("Analysis:", analysis)
        print("Parsed Tasks:", tasks)

        results = []
        for task_info in tasks:
            agent = self.get_worker(task_info["type"])
            result = agent.run(task, task_info["description"])
            print(f"\n=== {task_info["type"].upper()} RESULT ===\n{result}")
            results.append({
                "type": task_info["type"],
                "description": task_info["description"],
                "result": result
            })

        return {
            "analysis": analysis,
            "worker_results": results
        }

# prompt template
orchestrator_prompt = """
You are a permitting expert. Your job is to analyze a high-level construction request and break it into permitting-related subtasks.

<analysis>
Provide an overall explanation of what steps are involved.
</analysis>

<tasks>
Provide 2–4 <task> entries with a <type> and <description>. Use this format:
<task>
  <type>zoning</type>
  <description>Verify land is zoned for commercial use.</description>
</task>
</tasks>

Task: {task}
"""


# Main Runner

if __name__ == "__main__":
    user_prompt = "What permits do I need to add a pool to my house in The Woodlands Texas?"

    orchestrator = Orchestrator(orchestrator_prompt)
    results = orchestrator.process(user_prompt)

    print("\n\n=== FINAL REPORT ===")
    print("Analysis:\n", results["analysis"])
    for r in results["worker_results"]:
        print(f"\n--- {r['type'].upper()} ---")
        print("Description:", r["description"])
        print("Result:\n", r["results"])
