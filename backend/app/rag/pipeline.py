from app.models.document import Document
from app.rag.llm import generate
from app.rag.retriever import retrieve


def build_context(documents: list[Document]) -> str:
    """
    Combine retrieved documents into a context string.
    """

    return "\n\n".join(document.text for document in documents)


def build_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful assistant.

Answer the user's question using only the provided context.

If the answer cannot be found in the context, say you don't know.

Context:
{context}

Question:
{question}

Answer:
"""


def ask(question: str) -> str:
    """
    Execute the complete RAG pipeline.
    """

    documents = retrieve(question)
    context = build_context(documents)

    prompt = build_prompt(
        question=question,
        context=context,
    )

    return generate(prompt)