import streamlit as st
import requests
import os

API_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000/chat")

st.set_page_config(page_title="Gemini Domain Chatbot")

st.title("🤖 Gemini Domain-Specific Chatbot")
st.write("Ask about Python, FastAPI, Streamlit, APIs, or AI.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a question...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    # Call backend
    response = requests.post(
        API_URL,
        json={"question": user_input}
    )

    answer = response.json()["answer"]

    # Show bot response
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    with st.chat_message("assistant"):
        st.markdown(answer)
