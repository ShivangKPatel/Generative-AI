import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage

# -----------------------------
# LOAD ENV
# -----------------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found")

# -----------------------------
# DEFINE TOOLS
# -----------------------------
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b

@tool
def get_python_definition() -> str:
    """Return a short definition of Python programming language."""
    return "Python is a high-level, interpreted programming language used for web, AI, data science, and automation."

tools = [add_numbers, get_python_definition]

# -----------------------------
# INIT GEMINI LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0
)

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)

# -----------------------------
# AGENT LOOP
# -----------------------------
SYSTEM_PROMPT = """
You are a precise assistant.

Rules:
- If you call a tool, you MUST use the tool result to answer the user.
- If you don't call the tool, answer by your self.
- Do NOT say thank you, confirmations, or filler text.
- Return ONLY the final answer.
"""
def extract_text(response):
    if isinstance(response.content, list):
        return "".join(
            part.get("text", "") for part in response.content if isinstance(part, dict)
        )
    return response.content


def run_agent(user_input: str):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_input),
    ]

    response = llm_with_tools.invoke(messages)

    # Tool calling path
    if response.tool_calls:
        tool_call = response.tool_calls[0]

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n🔧 Tool Called: {tool_name}")
        print(f"📥 Arguments: {tool_args}")

        tool_map = {
            "add_numbers": add_numbers,
            "get_python_definition": get_python_definition
        }

        tool = tool_map[tool_name]

        # ✅ Execute tool correctly
        tool_result = tool.run(tool_args)

        # ✅ Send tool result as ToolMessage
        messages.append(response)
        messages.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                content=str(tool_result)
            )
        )

        final_response = llm_with_tools.invoke(messages)
        return extract_text(final_response)

    # Normal response
    return extract_text(response)

# -----------------------------
# DEMO
# -----------------------------
if __name__ == "__main__":
    print("🤖 Gemini Tool Agent\n")

    queries = [
        "Add 10 and 25",
        "What is Python?",
        "Explain vector databases"
    ]

    for q in queries:
        print(f"\n🧑 User: {q}")
        answer = run_agent(q)
        print(f"🤖 Agent: {answer}")
