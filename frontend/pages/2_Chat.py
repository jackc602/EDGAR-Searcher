import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from backend.llm_client import LLMClient, VALID_RERANKER_MODES
from frontend.sidebar import load_sidebar


logging.basicConfig(
    format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s',
    level=logging.INFO,
)


@st.cache_resource
def get_llm_client(model: str) -> LLMClient:
    return LLMClient(model=model)


load_sidebar()

st.title("Chat")

st.subheader("Select Language Model")
model_options = ["gemma3:270m", "gemma3:4b"]
if "selected_model" not in st.session_state:
    st.session_state.selected_model = model_options[0]
selected_model = st.selectbox(
    "Choose a model",
    model_options,
    index=model_options.index(st.session_state.selected_model),
)
st.session_state.selected_model = selected_model

with st.expander("Advanced RAG Settings"):
    col1, col2 = st.columns(2)
    with col1:
        n_chunks = st.slider(
            "Chunks for prompt",
            min_value=1,
            max_value=8,
            value=st.session_state.get("n_chunks", 5),
            help="Number of context chunks to include in the LLM prompt",
        )
        st.session_state.n_chunks = n_chunks
    with col2:
        n_candidates = st.slider(
            "Initial candidates",
            min_value=5,
            max_value=20,
            value=st.session_state.get("n_candidates", 10),
            help="Number of candidates to retrieve before reranking",
        )
        st.session_state.n_candidates = n_candidates

    env_default = os.environ.get("RERANKER_MODE", "crossencoder").lower()
    if env_default not in VALID_RERANKER_MODES:
        env_default = "crossencoder"
    default_mode = st.session_state.get("reranker_mode", env_default)
    reranker_mode = st.radio(
        "Reranker",
        list(VALID_RERANKER_MODES),
        index=list(VALID_RERANKER_MODES).index(default_mode),
        horizontal=True,
        help="Cross-encoder is strongest but loads a 70M-param model on first use; "
             "BM25 is keyword-only; off skips reranking entirely.",
    )
    st.session_state.reranker_mode = reranker_mode

# Metadata filters — populated from filings the user has loaded on the home page.
filings_meta = st.session_state.get("filings_metadata", []) or []
where_filter = None
if filings_meta:
    with st.expander("Filters"):
        all_tickers = sorted({fm["ticker"] for fm in filings_meta})
        all_types = sorted({fm["filing_type"] for fm in filings_meta})
        all_years = sorted({fm["filing_date"][:4] for fm in filings_meta})
        all_items = sorted({
            (fm.get("item_number") or "")
            for fm in filings_meta
            if fm.get("item_number")
        })

        sel_tickers = st.multiselect("Ticker", all_tickers, default=all_tickers)
        sel_types = st.multiselect("Filing type", all_types, default=all_types)
        sel_years = st.multiselect("Filing year", all_years, default=all_years)
        sel_items = st.multiselect("Item", all_items, default=[])

        clauses = []
        if sel_tickers and set(sel_tickers) != set(all_tickers):
            clauses.append({"ticker": {"$in": sel_tickers}})
        if sel_types and set(sel_types) != set(all_types):
            clauses.append({"filing_type": {"$in": sel_types}})
        if sel_years and set(sel_years) != set(all_years):
            # filing_date is "YYYY-MM-DD" so a year prefix range gives a year filter.
            year_clauses = [
                {"$and": [
                    {"filing_date": {"$gte": f"{y}-01-01"}},
                    {"filing_date": {"$lte": f"{y}-12-31"}},
                ]}
                for y in sel_years
            ]
            clauses.append({"$or": year_clauses} if len(year_clauses) > 1 else year_clauses[0])
        if sel_items:
            clauses.append({"item_number": {"$in": sel_items}})

        if len(clauses) == 1:
            where_filter = clauses[0]
        elif len(clauses) > 1:
            where_filter = {"$and": clauses}

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
        response_placeholder = st.empty()
        full_response = ""

        try:
            llm_client = get_llm_client(st.session_state.selected_model)
            stream = llm_client.ask_stream(
                prompt,
                collection_name="sec_filings_embeddings_v2",
                n_results=st.session_state.get("n_chunks", 5),
                n_candidates=st.session_state.get("n_candidates", 10),
                reranker_mode=st.session_state.get("reranker_mode"),
                where=where_filter,
            )

            for chunk in stream:
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")

            response_placeholder.markdown(full_response)
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
            })

        except Exception as e:
            st.error(f"An error occurred: {e}")
