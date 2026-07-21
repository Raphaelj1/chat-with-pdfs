from backend.app.models.document import Document
from backend.app.rag.llm import generate
from backend.app.rag.retriever import retrieve
from backend.app.rag.vector_store import VectorStore
from backend.app.rag.collection_manager import CollectionManager


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


def ask(question: str, session_id: str) -> str:
    """
    Execute the complete RAG pipeline.
    """
    
    manager = CollectionManager()
    collection = manager.get_or_create_collection(session_id)
    vector_store = VectorStore(collection)

    documents = retrieve(question, vector_store)
    context = build_context(documents)

    prompt = build_prompt(
        question=question,
        context=context,
    )

    return generate(prompt)