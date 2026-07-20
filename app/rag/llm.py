from openai import OpenAI
from app.config import OPENAI_API_KEY, LLM_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


def generate(prompt: str) -> str:
    """
    Generate a response from the language model.
    """

    response = client.responses.create(
        model=LLM_MODEL,
        input=prompt,
    )

    return response.output_text