import os
import json
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")  # Use your Groq API key here
base_url = os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1")

# Setup Groq OpenAI-compatible client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

# Load local Bdcalling knowledge base
with open("bdcalling_data.json", "r", encoding="utf-8") as file:
    bdcalling_data = json.load(file)

# Prepare system context prompt
context_prompt = f"""You are a helpful assistant answering questions based on the following company profile:
{json.dumps(bdcalling_data, indent=2)}
"""

# Streamlit UI
st.set_page_config(page_title="Bdcalling Chatbot", layout="centered")
st.title("🤖 Bdcalling Chatbot (Groq + LLaMA 3.1)")
st.markdown("Ask me anything about Bdcalling or its sub-companies!")

# Session memory
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "system", "content": context_prompt}]

# Show chat history
for msg in st.session_state["messages"][1:]:  # Skip system message
    st.chat_message(msg["role"]).write(msg["content"])

# Input field
user_input = st.chat_input("Your question...")

if user_input:
    # Save user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    st.chat_message("user").write(user_input)

    # Call Groq API (LLaMA 3.1)
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=st.session_state["messages"]
        )
        reply = response.choices[0].message.content
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.chat_message("assistant").write(reply)

    except Exception as e:
        st.error(f"❌ API Error: {e}")
