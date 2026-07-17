import os
import fitz
from dotenv import load_dotenv
from openai import OpenAI
from chromadb import chromadb, Collection


load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

PDF_PATH = ""
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    
    for page_no, page in enumerate(doc):
        text = page.get_text()
        text = text.strip()
        if text:
            pages.append((page_no + 1, text))
    doc.close()
    print(f"Extracted text from {len(pages)} pages")
    return pages


def chunk_text(text, chunk_size, overlap):
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        start += chunk_size - overlap
    return chunks
        

def embed_texts(texts):
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]


def index_pdf(pdf_path):
    chroma_client = chromadb.PersistentClient(path="db/chroma_db")
    
    # try:
    #     chroma_client.delete_collection("pdf_rag")
    # except:
    #     pass
    
    collection = chroma_client.get_or_create_collection("pdf_rag")
    
    pages = extract_text_from_pdf(pdf_path)
    
    all_chunks   = []
    all_metadata = []
    all_ids      = []
    chunk_index  = 0
    
    for page_no, page_text in pages:
        chunks = chunk_text(page_text, CHUNK_SIZE, CHUNK_OVERLAP)
        
        for chunk in chunks:
            if len(chunk.strip()) < 50:
                continue
            
            all_chunks.append(chunk)
            all_metadata.append({
                "source": os.path.basename(pdf_path),
                "page": page_no,
                "chunk": chunk_index
            })
            all_ids.append(f"chunk_{chunk_index}")
            chunk_index += 1
        
    BATCH_SIZE = 100
    all_embeddings = []
    
    for i in range(0, len(all_chunks), BATCH_SIZE):
        batch = all_chunks[i : i + BATCH_SIZE]
        embeddings = embed_texts(batch)
        all_embeddings.extend(embeddings)
        print(f"Embedding batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")
        
    collection.add(
        documents=all_chunks,
        embeddings=all_embeddings,
        metadatas=all_metadata,
        ids=all_ids
    )
    
    print(f"\nDone: Indexed {len(all_chunks)} chunks from {pdf_path}")
    return collection


def ask(collection: Collection, question: str, n_results=3):
    question_embedding = embed_texts([question])[0]
    
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    
    print(f"\nQuestion: {question}")
    # print(f"\nRetrieved chunks")
    
    # for chunk, meta, dist in zip(chunks, metadatas, distances):
    #     print(f"    [Page {meta['page']} | score: {dist:.3f}] {chunk[:100]}...")
        
    context = "\n\n---\n\n".join(chunks)
    print(f"\nContext: \n{context}")
    
    
    prompt = f"""" You are a helpful assistant. Answer the question using ONLY the context below.
        If the answer is not in the context, say "I don't know".

        Context: {context}

        Question: {question}

        Answer:"""
        
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(f"\nAnswer: {response.choices[0].message.content}")


if __name__ == "__main__":
    collection = index_pdf(PDF_PATH)
    
    ask(collection, "What is the main topic of this document?")
