from typing import Dict, List
from openai import OpenAI
from utils import logger

def generate_response(openai_key: str, user_message: str, context: str, 
                     conversation_history: List[Dict], model: str = "gpt-3.5-turbo") -> str:
    """Generate response using OpenAI with context"""

    # Define system prompt
    system_prompt = {
        "role": "system",
        "content": "You are NASA expert specializing in space science, engineering and exploration. "
                    "Answer questions clearly and accurately using provided contents. "
                    "Cite relevant details from retrieved documents. "
                    "If uncertain, explicitly state your confidence level. "
                    "Correct any misconceptions in the questions itself. "
                    "Distinguish between confirmed facts and speculation. "
                    "Refused request for classified or sensitive information."
    } 

    if not conversation_history:
        conversation_history = [system_prompt]
    else:
        conversation_history.append(system_prompt)
    # Set context in messages
    context_text = "\n\n".join([f"Document: {doc}" for doc in context])

    messages = [
        system_prompt,
        {"role": "system", "content": f"Use this context: {context_text}"}
    ] + conversation_history[1:] + [
        {"role": "user", "content": user_message}
    ]

    try:
        if openai_key.startswith("voc-"):
            client = OpenAI(
                api_key=openai_key,
                base_url="https://openai.vocareum.com/v1",
            )
        else:
            client = OpenAI(api_key=openai_key)
        logger.info("[generate_response] Succecfully Retreived API Key!")
    except Exception as e:
        logger.error("[generate_response] Error retrieving API Key: {e}", exc_info=True)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    if response:
        logger.info("[generate_response] Successfully Generated Response")
    else:
        logger.error("[generate_response] Return empty reponse", exc_info=True)

    answer = response.choices[0].message.content

    conversation_history.append({"role": "user", "content": user_message})
    conversation_history.append({"role": "assistant", "content": answer})

    return answer
