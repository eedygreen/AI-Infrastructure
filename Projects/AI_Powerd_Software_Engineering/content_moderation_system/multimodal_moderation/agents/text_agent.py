
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModelSettings
from utils import logger
from multimodal_moderation.types.model_choice import ModelChoice
from multimodal_moderation.types.moderation_result import ModerationResult, TextModerationResult


MODERATION_INSTRUCTIONS = """
<context>
At ACME enterprise we strive for a friendly but professional interaction with our customers.
</context>

<role>
You are a customer service reviewer at ACME enterprise. You make sure that the customer
service interactions are friendly and professional.
</role>

<input>
You will receive a message from the customer representative towards the customer.
</input>

<instructions>
Detect if:
- the tone of the message is unfriendly
- the tone of the message is unprofessional
- the message contains any personally-identifiable information (PII)
</instructions>

<output>
Provide a detailed rationale for your choices as well as a confidence score between 0 and 1 on your assessment.
</output>
"""

model_settings = GoogleModelSettings(
    google_thinking_config={"thinking_budget": 0},
    seed=42,
    temperature=0.0
)

text_moderation_agent = Agent(
    model="google-gla:gemini-2.5-flash-lite",
    instructions=MODERATION_INSTRUCTIONS,
    output_type=TextModerationResult,
    model_settings=model_settings,
    retries=3
)


async def moderate_text(model_choice: ModelChoice, text: str) -> TextModerationResult:

    try:
        result = await text_moderation_agent.run(
            user_prompt=[text],
            model=model_choice.model,
            model_settings=model_choice.model_settings
            )
        logger.info(f"[text_agent] Successfully moderate texts")
        return result.output
    except Exception as e:
        logger.error(f"[text_agent] Error moderating text: {e}", exc_info=True)
        raise NotImplementedError("Implement text moderation")
