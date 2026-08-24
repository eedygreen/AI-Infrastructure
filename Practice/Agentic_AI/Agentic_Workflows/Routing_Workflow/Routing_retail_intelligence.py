import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables and initialize OpenAI client
load_dotenv("../.env")
client = OpenAI(
    base_url = "https://openai.vocareum.com/v1",
    api_key=os.getenv("OPENAI_API_KEY"))

# --- Helper Function for API Calls ---
def call_openai(system_prompt, user_prompt, model="gpt-3.5-turbo"):
    """Simple wrapper for OpenAI API calls."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"error occured: {e}"


# --- Agents for Different Retail Tasks ---

def product_researcher_agent(query):
    """Product researcher agent gathers product information."""
    system_prompt = """You are a product research agent for a retail company. Your task is to provide 
    structured information about products, market trends, and competitor pricing."""
    
    user_prompt = f"Research this product thoroughly: {query}"
    return call_openai(system_prompt, user_prompt)


def customer_analyzer_agent(query):
    """Customer analyzer agent processes customer data and feedback."""
    system_prompt = """You are a customer analysis agent. Your task is to analyze customer feedback, 
    preferences, and purchasing patterns."""
    
    user_prompt = f"Analyze customer behavior for: {query}"
    return call_openai(system_prompt, user_prompt)


def pricing_strategist_agent(query, product_data=None, customer_data=None):
    """
    Pricing strategist agent recommends optimal pricing.

    Args:
        query (str): The string contains User's question
        product_data (str): Data about product from researcher
        customer_data (str): Data from the Analyzer

    Output:
        Return LLM response
    """
    system_prompt = """You are a pricing strategist agent. Your task is to recommend optimal pricing 
    strategies based on product research and customer analysis."""
    
    user_prompt = f"""
    Original Pricing query: {query}
    Product Reserach Data: {product_data}
    Customer Analysis Data:
    {customer_data}
    Based on all the above information, please provide a recommended pricing strategy, \
    suggest an optimal price or price range, and explain your reasoning.
    """
    response = call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    print(response)
    return response


# --- Routing Agent with LLM-Based Task Determination ---
def routing_agent(query, context=None):
    """
    Routing agent that determines which agent to use based on the query.

    Args:
        query (str): User request
        *args: any context added to the request

    Output: 
        context (str): Agent resposible for the query 
    """
    
    system_prompt = """
    You are an AI assistant that can route query (request) to the right agents.
    You will be given a query, and your job is to determine the appropraite agent to handle it.
    Agents Available:
    - Product Resercher Agent: Researchers product specifications, market trends, and competitor pricing.
    - Customer Analyzer Agent: Analyzes customer feedback, preferences, and purchasing patterns.
    - Pricing Strategist Agent: Recommends optimal pricing strategies base on reasearch and analysis.-
                                                       
    Respond only with the agent's name, nothing else.
    """
    user_prompt = f"Given the query: '{query}',  which agent should handle thisn task?"

    agent_choice = call_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt
    )
    print(f"Selected Agent: {agent_choice}")

    # Route the query to the correct agent based on the choice

    if "Product Research" in agent_choice:
        print("Routing query to product Research Agent...")
        return product_researcher_agent(query=query)

    elif "Customer Analyzer" in agent_choice:
        print("Routing query to Pricing Stategist Agent...")
        return customer_analyzer_agent(query=query)

    elif "Pricing Strategist" in agent_choice:
        print("Routing query to Pricing Strategist Agent...")

        # further information needed
        # get the product information
        product_data = None
        if context and "product_data" in context:
            product_data = context["product_data"]
        else:
            print("Getting product information first...")
            product_data = product_researcher_agent(query=query)

            # Then get customer insights
        customer_data = None
        if context and "customer_data" in context:
            customer_data = context["customer_data"]
        else:
            print("Getting customer insights...")
            customer_data = customer_analyzer_agent(query=query)

        # Finally, determine pricing strategy using both inputs
        return pricing_strategist_agent(query, product_data, customer_data)

    else:
        return f"Couldn't route query. Agent decision was: {agent_choice}"


# --- Example Usage ---
if __name__ == "__main__":
    # Example queries
    queries = [
        "What are the specifications and current market trends for wireless earbuds?",
        "What do customers think about our premium coffee brand?",
        "What should be the optimal price for our new organic skincare line?"
    ]
    
    # Process each query
    for query in queries:
        print(f"\nQuery: {query}")
        print("\nProcessing...")
        
        results = routing_agent(query=query)
        print("\nResult:")
        print(results)
        print("\n" + "-"*80)