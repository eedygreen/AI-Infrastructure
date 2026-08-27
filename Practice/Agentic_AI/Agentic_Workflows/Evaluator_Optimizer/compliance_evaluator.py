import os
from dotenv import load_dotenv
from openai import OpenAI
from enum import Enum

load_dotenv("../.env")

client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key = os.getenv("OPENAI_API_KEY")
)

MAX_RETRIES = 5
user_prompt = (
    "Write a summary for potential investors explaining why decentralized finance (DeFi) will outperform "
    "traditional banknig in the next five years. use strong language to inspire confidence and urgency. "
    "Include examples of past DeFi gains and suggest what investors can expect from leading protocols in the near future."
)


def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model="gpt-4"
):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error occured: {e}"


class FinancilaReportAgent:
    def run(self, prompt, feedback=None):
        system_message = "You are a financila analyst writing a professional investment summary."

        full_prompt = prompt
        if feedback:
            full_prompt += f"\n\nEvaluator feedback: {feedback}\n Please revise accordingly."

        print(f"\nGenerating report with prompt: \n{full_prompt}\n")
        response = call_llm(
            system_prompt=system_message,
            user_prompt=full_prompt,
            temperature=0.5,
        )
        return response

class ComplianceAgent:
    def run(self, report_text):
        print("Evaluating report compliance...")
        system_message = (
            "You are a compliance officer reviewing investment summaries. "
            "Reject anything with forward-looking statements, speculative claims, "
            "or language like 'expected', 'projected', 'will likely', etc."
        )
        eval_prompt = f"Evaluate this investment summary for compliance:\n\n{report_text}\n\nRespond with 'Approved' or provide feedback for revision."

        response = call_llm(
            system_prompt=system_message,
            user_prompt=eval_prompt,
            temperature=0.0
        )
        return response

def main():
    report_agent = FinancilaReportAgent()
    eval_agent = ComplianceAgent()

    report_text = ""
    feedback = None

    for attempt in range(MAX_RETRIES):
        print(f"--- Attempt #{attempt} ---")
        report_text = report_agent.run(user_prompt, feedback)
        evaluation = eval_agent.run(report_text)

        print(f"\n Evaluation Result:\n{evaluation}\n")

        if evaluation.lower().startswith("approved"):
            print("\n Final Approved Investment Summary:\n")
            print(report_text)
            break
        else:
            feedback = evaluation
    else:
        print("\n Failed to meet compliance after max retries")
        print("Last version of the report:")
        print(report_text)


if __name__ == "__main__":
    main()
