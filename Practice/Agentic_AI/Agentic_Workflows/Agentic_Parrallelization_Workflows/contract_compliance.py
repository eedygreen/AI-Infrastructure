import os
from openai import OpenAI
from dotenv import load_dotenv
from enum import Enum
import threading

# Load environment variables and initialize OpenAI client
load_dotenv("../.env")
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))

# Shared dict for thread-safe collection of agent outputs
agent_outputs = {}

class Model(str, Enum):
    GPT4="gpt-4",
    GPT3="gpt-3.5-turbo"


# Example contract text (in a real application, this would be loaded from a file)
contract_text = """
CONSULTING AGREEMENT

This Consulting Agreement (the "Agreement") is made effective as of January 1, 2025 (the "Effective Date"), by and between ABC Corporation, a Delaware corporation ("Client"), and XYZ Consulting LLC, a California limited liability company ("Consultant").

1. SERVICES. Consultant shall provide Client with the following services: strategic business consulting, market analysis, and technology implementation advice (the "Services").

2. TERM. This Agreement shall commence on the Effective Date and shall continue for a period of 12 months, unless earlier terminated.

3. COMPENSATION. Client shall pay Consultant a fee of $10,000 per month for Services rendered. Payment shall be made within 30 days of receipt of Consultant's invoice.

4. CONFIDENTIALITY. Consultant acknowledges that during the engagement, Consultant may have access to confidential information. Consultant agrees to maintain the confidentiality of all such information.

5. INTELLECTUAL PROPERTY. All materials developed by Consultant shall be the property of Client. Consultant assigns all right, title, and interest in such materials to Client.

6. TERMINATION. Either party may terminate this Agreement with 30 days' written notice. Client shall pay Consultant for Services performed through the termination date.

7. GOVERNING LAW. This Agreement shall be governed by the laws of the State of Delaware.

8. LIMITATION OF LIABILITY. Consultant's liability shall be limited to the amount of fees paid by Client under this Agreement.

9. INDEMNIFICATION. Client shall indemnify Consultant against all claims arising from use of materials provided by Client.

10. ENTIRE AGREEMENT. This Agreement constitutes the entire understanding between the parties and supersedes all prior agreements.

IN WITNESS WHEREOF, the parties have executed this Agreement as of the date first above written.
"""

# helper funtion to call llm
def call_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    model=None
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
    
class LegalTermsChecker:
    """Agent that checks for problematic legal terms and clauses in contracts."""
    def run(self, contract_text):
        print("LegalTermsChecker: Analyzing contract for problematic legal terms...")

        system_prompt = "You are a legal expert specializing in contract law. " \
        "Review the provided contract text and identify any problematic caluses, " \
        "ambiguous terms, or non-standard legal language. List your key findings."

        user_prompt = f"Analyze: {contract_text}"
        llm_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            model=Model.GPT3
        )
        agent_outputs["legal"] = llm_response

class ComplianceValidator:
    """Agent that validates regulatory and industry compliance of contracts."""
    def run(self, contract_text):
        print("ComplianceValidator: Validating against industry standards or regulatory")

        system_prompt = "You are legal compliance expert specializing in industrial regulation." \
        "Review the provided contract text and identify any compliance clause, regulatory violationsm deviations in standards."

        user_prompt = f"Analyze: {contract_text}"
        llm_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            model=Model.GPT3
        )
        agent_outputs["compliance"] = llm_response

class FinancialRiskAssessor:
    """Agent that assesses financial risks and liabilities in contracts."""
    def run(self, contract_text):
        print("FinancialRiskAssessor: Assessing financial risks and liabilities in contracts.")

        system_prompt = "You are a Financial Risk Assessment expert specialize in finding " \
        "risks in contract that could lead to financial loss before it happened." \
        "Review the contract text and identify financial risks in it."

        user_prompt = f"Analyze: {contract_text}"
        llm_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            model=Model.GPT3
        )
        agent_outputs["financial"] = llm_response

class SummaryAgent:
    """Agent that synthesizes findings from all specialized agents."""
    def run(self, contract_text, inputs):
        print("SummaryAgent: Synthesizing all findings...")
        legal_findings = inputs.get("legal", "No legal analysis provided.")
        compliance_findings = inputs.get("compliance", "No compliacne analysis provided.")
        financial_findings = inputs.get("financial", "No financial analysis provided.")
        system_prompt = "You are senior legal counsel..." \
        "You have recieved analyses on a contract from legal terms, compliance, and financlial risk specialist." \
        "Your task is to synthesize these findings into single, comprehensive executive summary " \
        "of the contract's overall status and key concerns."
        user_prompt = f"""
        Please synthesize the following analyses of a contract into a comprehensive
        summary report. Original Contract Text (for reference, if needed, but focus on the analyses):
        ---BEGIN CONTRACT TEXT ---
        {contract_text[:500]}...
        --- END OF CONTRACT TEXT ---
        Legal Terms Analysis:
        {legal_findings}
        Compliance Validation:
        {compliance_findings}
        Financial Risk Assessment:
        {financial_findings}
        Provide a consolidated executive summary identifying key issues and an overall assessment.
        """
        llm_response = call_llm(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.3,
            model=Model.GPT4
        )
        return llm_response

# Main function to run all agents in parallel
def analyze_contract(contract_text):
    """Run all agents in parallel and summarize their findings."""
    legal_checker = LegalTermsChecker()
    compliance_validator = ComplianceValidator()
    financial_assessor = FinancialRiskAssessor()
    summary_creator = SummaryAgent()

    threads = [
        threading.Thread(target=legal_checker.run, args=(contract_text,)),
        threading.Thread(target=compliance_validator.run, args=(contract_text,)),
        threading.Thread(target=financial_assessor.run, args=(contract_text,))
    ]

    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    final_report = summary_creator.run(
        contract_text=contract_text,
        inputs=agent_outputs
    )
    return final_report

if __name__ == "__main__":
    print("Enterprise Contract Analysis System")
    print("Analyzing contract...")
    
    final_analysis = analyze_contract(contract_text)
    print("\n=== FINAL CONTRACT ANALYSIS ===\n")
    print(final_analysis)
