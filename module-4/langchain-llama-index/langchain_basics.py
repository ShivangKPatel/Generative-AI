import os
from dotenv import load_dotenv

# LangChain (Gemini)
from langchain_google_genai import ChatGoogleGenerativeAI

# LlamaIndex (Gemini)
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env")

# -----------------------------
# CONFIGURE LLAMAINDEX (v1beta)
# -----------------------------
Settings.llm = Gemini(
    model="gemini-2.5-flash",   # FULL path required
    api_key=GEMINI_API_KEY
)

Settings.embed_model = GoogleGenAIEmbedding(
    model="text-embedding-004",
    api_key=GEMINI_API_KEY
)

# -----------------------------
# LOAD DOCUMENTS
# -----------------------------
documents = SimpleDirectoryReader("data").load_data()

index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine(similarity_top_k=2)

# -----------------------------
# LANGCHAIN LLM
# -----------------------------
langchain_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",          # SHORT name required
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# -----------------------------
# INTEGRATION FLOW
# -----------------------------
def run_query(question: str):
    retrieval_response = query_engine.query(question)
    context = retrieval_response.response

    prompt = f"""
You are an AI assistant.
Use the context below to answer the question.

Context:
{context}

Question:
{question}
"""

    response = langchain_llm.invoke(prompt)
    return response.content


# -----------------------------
# DEMO RUN
# -----------------------------
if __name__ == "__main__":
    question = "What is LlamaIndex used for?"
    answer = run_query(question)

    print("Question:", question)
    print("Answer:", answer)
