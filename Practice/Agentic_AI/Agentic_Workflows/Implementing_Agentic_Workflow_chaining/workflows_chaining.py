import os
from openai import OpenAI  # type: ignore
from dotenv import load_dotenv  # type: ignore

# Load environment variables and initialize OpenAI client
load_dotenv('../.env')
client = OpenAI(
    base_url="https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))

def call_openai(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    """Simple wrapper for OpenAI API calls"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def feedstock_analyst_agent(feedstock_name: str):
    # Analyze the hydrocarbon feed
    """Analyze the type of hydrocarbon feedstock.

       Args:
            feedstock name (str): Name of the hydrocarbon stock

       Output:
            Description of the hydracarbon feedstock
    """
    system_prompt = """
    You are a petrochemical expert analyzing hydrocarbon feedstocks.
    Provide a concise analysis of the given feedstock, highlighting its key components and
    general suitability for producing valuable refined products like gasoline, diesel, and kerosene.
    """
    user_prompt = f"Analyze the feedstock: {feedstock_name}"

    response = call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt, 
    )

    return response


def distillation_planner_agent(feedstock_analysis):
    # Allocate through distillation tower
    """Allocate through distillation tower with the feedstcok analysis

       Args:
            feedstock_analysis (str): feedstock analysis report from Analyst

       Output:
            Estiamtions (str): Product volumes estimations
    """
    example_output = "Gasoline: 40%, Diesel: 30%, Kerosene: 20%, Other: 10%"
    system_prompt = f"""
    You are a refinery distillation tower operations planner.
    Based on the provided feedstock analysis, estimate the potential percentage yeilds
    for major products like the gasoline, diesel, and kerosene. Be realistic.
    respond as {example_output}
    """

    user_prompt = f"Allocate the feedstock analysis: {feedstock_analysis}"

    response = call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )

    return response

def market_analyst_agent(product_list):
    # Analyze market conditions
    """Assess current market demand and pricing for products

       Args:
            product_list (list): list of products

       Outputs:
            market analysis, demand levels and profitability
    """
    system_prompt = """
    You are an energy market analyst. For the following list of
    refined products, provide a brief analysis of current market demand (high, medium, low)
    and general profitability trends.
    """

    user_prompt = f"Analyze the market for these refinded products: {product_list}"

    response = call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    return response


def production_optimizer_agent(distillation_plan, market_data):
    # Recommend a production plan
    """Recommend an optimal production plan balancing yield and market needs.

       Args:
            distillation_plan: The output from `Distillation Planner Agent`.
            market_data: The output from `Market Analyst Agent`.

       Output:
            Product Recommendation
    """

    system_prompt = """
    You are a refinery production optimization expert.
    Your goal is to recommend a production strategy based on pontential yields
    and current market conditions.
    """

    user_prompt = f"""
    Given the following potential distillation plan:
    --- DISTILLATION PLAN ---
    {distillation_plan}
    --- END DISTILATION PLAN ---
    And the following market analysis:
    --- MARKET ANALYSIS ---
    {market_data}
    --- END MARKET ANALYSIS ---
    Please provide a concise recommendation on which products the refinery should prioritize \
    of focus on to maximize value, considering both the potential yield and market conditions.
    """

    response = call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    return response

def run_refinery_chain(feedstock_name):
    """Run the chain

       Args: 
            feedstock_name (str): The hydrocarbon feedstock
    """
    print(f"Processing feedstock: {feedstock_name}\n")
    analysis_1 = feedstock_analyst_agent(feedstock_name)
    print(f"\n--- Feedstock Analysis --- \n{analysis_1}")

    plan2 = distillation_planner_agent(analysis_1)
    print(f"\n--- Distillation Plan ---\n{plan2}\n")

    market_info_3 = market_analyst_agent(plan2)
    print(f"\n--- Market Analysis ---\n{market_info_3}")

    final_recommendation = production_optimizer_agent(plan2, market_info_3)
    print(f"\n--- OPTIMIZED PRODUCTION RECOMMENDATION ---\n{final_recommendation}\n")

if __name__ == "__main__":
    current_feedstock = "West Texas Intermidiate Crude"

    run_refinery_chain(current_feedstock)
