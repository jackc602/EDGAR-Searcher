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
model_options = ["llama3", "mistral", "deepseek-r1"]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_options[0]
selected_model = st.selectbox(
    "Choose a model",
    model_options,
    index=model_options.index(st.session_state.selected_model)
)
st.session_state.selected_model = selected_model

# Advanced RAG Settings
with st.expander("Advanced RAG Settings"):
    col1, col2 = st.columns(2)
    with col1:
        n_chunks = st.slider(
            "Chunks for prompt",
            min_value=1,
            max_value=5,
            value=st.session_state.get("n_chunks", 2),
            help="Number of context chunks to include in the LLM prompt"
        )
        st.session_state.n_chunks = n_chunks
    with col2:
        n_candidates = st.slider(
            "Initial candidates",
            min_value=5,
            max_value=20,
            value=st.session_state.get("n_candidates", 10),
            help="Number of candidates to retrieve before reranking"
        )
        st.session_state.n_candidates = n_candidates

    use_reranking = st.checkbox(
        "Enable keyword reranking (BM25)",
        value=st.session_state.get("use_reranking", True),
        help="Use BM25 keyword scoring to improve retrieval quality"
    )
    st.session_state.use_reranking = use_reranking

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Stop button (only show when generating)
if st.session_state.is_generating:
    if st.button("Stop Generating", type="primary"):
        st.session_state.is_generating = False
        st.rerun()

# Chat input
if prompt := st.chat_input("Ask a question about the filings", disabled=st.session_state.is_generating):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.is_generating = True
    st.rerun()

# Generate response if we have a pending user message
if st.session_state.is_generating and st.session_state.messages:
    last_message = st.session_state.messages[-1]
    if last_message["role"] == "user":
        # Display user message
        with st.chat_message("user"):
            st.markdown(last_message["content"])

        # Generate assistant response
        with st.chat_message("assistant"):
            stop_button = st.button("Stop", key="stop_inline")
            response_placeholder = st.empty()
            full_response = ""

            try:
                llm_client = LLMClient(model=st.session_state.selected_model)
                stream = llm_client.ask_stream(
                    last_message["content"],
                    collection_name="sec_filings_embeddings",
                    n_results=st.session_state.get("n_chunks", 2),
                    n_candidates=st.session_state.get("n_candidates", 10),
                    use_reranking=st.session_state.get("use_reranking", True)
                )

                for chunk in stream:
                    if not st.session_state.is_generating:
                        # User clicked stop
                        full_response += " [Generation stopped]"
                        break
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")

                response_placeholder.markdown(full_response)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                st.error(f"An error occurred: {e}")

            st.session_state.is_generating = False
            st.rerun()
