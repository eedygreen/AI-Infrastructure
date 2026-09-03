import os
from workflow_agents.base_agents import AugmentedPromptAgent
from dotenv import load_dotenv

load_dotenv()

augmented_agent = AugmentedPromptAgent(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    persona="You an augmented prompt-based assistant"
)
user_request = "What car is better; Xpeng G9l, Xpeng Mona L03, Xiaomi N70"
augmented_agent_response = augmented_agent.respond(user_request)
print(augmented_agent_response)

# output
"""
The Agent used knowledge from vector Database to generate its response. As it lacks information on current event.

[Model Response]
As an AI assistant, I don't have personal opinions. However, I can provide you with some information to help you make a decision.

The Xpeng G9l and Xpeng Mona L03 are electric vehicles produced by Xpeng Motors, a Chinese electric vehicle manufacturer. Both models have received positive reviews for their performance, range, and technology features.

On the other hand, Xiaomi N70 is not a known car model. Xiaomi is a Chinese technology company known for its smartphones and other consumer electronics products. It's possible that you may have made a typo or error in the model name.

If you are looking for a car, I would recommend researching and comparing the specifications, features, and reviews of the Xpeng G9l and Xpeng Mona L03 to determine which one better suits your needs and preferences.

[Current Information]
Xiaomi N70 is an SUV produced by Xiaomi, Xiaomi produced other cars as well. But Our model knowledge is far behind.
"""
