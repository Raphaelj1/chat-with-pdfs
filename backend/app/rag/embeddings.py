from openai import OpenAI
from backend.app.config import OPENAI_API_KEY, EMBEDDING_MODEL


client = OpenAI(api_key=OPENAI_API_KEY)


def embed_text(text: str) -> list[float]:
    """
    Generate an embedding for a single piece of text.
    """
    
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    
    return response.data[0].embedding


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in one request.
    """

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]