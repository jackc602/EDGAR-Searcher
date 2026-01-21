import streamlit as st
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from backend.edgar_client import get_filings
from backend.embedding_client import EmbeddingClient
from backend.llm_client import LLMClient
from frontend.sidebar import load_sidebar
import logging
logging.basicConfig(format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(layout="wide")

load_sidebar()

st.title("SEC Filings Searcher")

# 1. User Input
ticker = st.text_input("Enter Company Ticker (e.g., AAPL)")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

# 2. Data Extraction and Embedding
if st.button("Load and Embed Filings"):
    if ticker and start_date and end_date:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        with st.spinner("Fetching and embedding SEC filings..."):
            try:
                # Convert dates to string format
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")

                # Fetch filings
                filings = get_filings(ticker, start_date_str, end_date_str)
                st.session_state.text_data = filings
                st.success("Filings extraction complete!")

                # Embed filings
                if filings:
                    embedding_client = EmbeddingClient()
                    embedding_client.embed_and_store(filings, collection_name="sec_filings_embeddings")
                    st.success("Embedding complete!")
                else:
                    st.warning("No filings found for the given criteria.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter all fields.")

# 3. Text Preview
if "text_data" in st.session_state and st.session_state.text_data:
    st.subheader("Preview of Extracted Filings")
    st.text_area("Extracted Filings", value="\n".join(st.session_state.text_data[:1]), height=200, disabled=True)


# 4. Model Selection
st.subheader("Select LLM Model")
model_options = ["llama2", "mistral", "codellama"] # Common Ollama models
selected_model = st.selectbox("Choose a model", model_options)
st.session_state.selected_model = selected_model

# 5. Interact with the model
st.subheader("Chat with your SEC Filings Data")

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
