import streamlit as st
from openai import OpenAI

st.title("🤖 Live Dynamic Rag")

# Set up client using secrets
client = OpenAI(
    api_key=st.secrets["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com/v1",
)

# Default model
if "model" not in st.session_state:
    st.session_state["model"] = "deepseek-chat"

# Chat history with Trump-only constraint
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant. Only respond to questions if they are "
                "about Donald Trump. If the question is not related to Trump, politely reply "
                "with 'Sorry, I can only discuss topics related to Donald Trump.'"
            )
        }
    ]

# Display messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input
if prompt := st.chat_input("Ask something..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Stream response
    with st.chat_message("assistant"):
        full_response = ""
        response_area = st.empty()
        response = client.chat.completions.create(
            model=st.session_state["model"],
            messages=st.session_state.messages,
            stream=True,
        )
        for chunk in response:
            if chunk.choices[0].delta.content:
                full_response += chunk.choices[0].delta.content
                response_area.markdown(full_response + "▌")
        response_area.markdown(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
