import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sentences = [
    "The dog ran across the park.",
    "A puppy sprinted through the garden.", # Similar to #1
    "The quaterly earnings report is due.", # Very different
]

embeddings = []


for sentence in sentences:
    response = client.embeddings.create(
        input=sentence,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    embeddings.append(embedding)
    print(f"Sentence : {sentence}")
    print(f"Embedding : [{embedding[0]:.4f}, {embedding[1]:.4f}, {embedding[2]:.4f}, ...]")
    print(f"Length : {len(embedding)} numbers\n")


def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

sim_1_2 = dot_product(embeddings[0], embeddings[1])
sim_1_3 = dot_product(embeddings[0], embeddings[2])

print("-" * 30)
print(f"Similarity: sentence 1 vs sentence 2 -> {sim_1_2} (similar meaning)")
print(f"Similarity: sentence 1 vs sentence 3 -> {sim_1_3} (similar meaning)")
print()
print("Similar sentences score higher")