import streamlit as st
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from backend.llm_client import LLMClient
from frontend.sidebar import load_sidebar

load_sidebar()

st.title("Chat with your SEC Filings Data")

# Model Selection
st.subheader("Select LLM Model")
model_options = ["llama2", "mistral", "codellama"] # Common Ollama models
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_options[0]
selected_model = st.selectbox("Choose a model", model_options, index=model_options.index(st.session_state.selected_model))
st.session_state.selected_model = selected_model

# Interact with the model
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the filings"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm_client = LLMClient(model=st.session_state.selected_model)
                response = llm_client.ask(prompt, collection_name="sec_filings_embeddings")
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred while interacting with the LLM: {e}")
