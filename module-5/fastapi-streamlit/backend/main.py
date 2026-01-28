import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

app = FastAPI(title="Gemini Domain Chatbot API")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # free-tier friendly
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

SYSTEM_PROMPT = """
You are a domain-specific chatbot for Python interns.

You can answer questions ONLY about:
- Python
- FastAPI
- Streamlit
- APIs
- AI basics

If a question is outside this domain, politely refuse.
Give clear and concise answers.
"""

class ChatRequest(BaseModel):
    question: str

class ChatResponse(BaseModel):
    answer: str

def extract_text(response):
    if isinstance(response.content, list):
        return "".join(
            part.get("text", "") for part in response.content if isinstance(part, dict)
        )
    return response.content

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=request.question)
    ]

    response = llm.invoke(messages)
    answer = extract_text(response)

    return ChatResponse(answer=answer)

@app.get("/")
def health():
    return {"status": "Gemini chatbot API running"}
