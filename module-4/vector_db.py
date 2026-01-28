import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in .env file")

INDEX_NAME = "intern-demo-index"
DIMENSION = 384  # all-MiniLM-L6-v2 embedding size

pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if it doesn't exist
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    time.sleep(5)  # wait for index readiness

index = pc.Index(INDEX_NAME)

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = {
    "doc1": "Python is a popular programming language",
    "doc2": "Machine learning models learn from data",
    "doc3": "Pinecone is a vector database for AI applications",
    "doc4": "Interns use Python in real-world projects"
}

vectors = []
for doc_id, text in documents.items():
    embedding = model.encode(text).tolist()
    vectors.append({
        "id": doc_id,
        "values": embedding,
        "metadata": {"text": text}
    })

index.upsert(vectors=vectors)
print("✅ Documents indexed successfully")

query_text = "What is Pinecone used for?"
query_embedding = model.encode(query_text).tolist()

results = index.query(
    vector=query_embedding,
    top_k=2,
    include_metadata=True
)

print("\n🔍 Query:", query_text)
for match in results["matches"]:
    print(f"- Score: {match['score']:.4f}")
    print(f"  Text: {match['metadata']['text']}")
