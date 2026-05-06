import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from backend.edgar_client import get_filings
from backend.embedding_client import EmbeddingClient
from backend.document_chunker import chunk_filing, extract_text_from_html
from frontend.sidebar import load_sidebar


logging.basicConfig(
    format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s',
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_embedding_client():
    return EmbeddingClient()


st.set_page_config(layout="wide")

load_sidebar()

st.title("SEC Filings Searcher")

ticker = st.text_input("Enter Company Ticker (e.g., AAPL)")
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Load and Embed Filings"):
    if ticker and start_date and end_date:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        with st.spinner("Fetching SEC filings..."):
            try:
                start_date_str = start_date.strftime("%Y-%m-%d")
                end_date_str = end_date.strftime("%Y-%m-%d")

                filings = get_filings(ticker, start_date_str, end_date_str)

                if not filings:
                    st.warning("No filings found for the given criteria.")
                else:
                    st.success(f"Found {len(filings)} filings!")

                    all_chunks = []
                    progress_bar = st.progress(0, text="Chunking filings...")

                    for idx, filing in enumerate(filings):
                        chunks = chunk_filing(
                            html_content=filing.content,
                            ticker=filing.ticker,
                            cik=filing.cik,
                            accession_number=filing.accession_number,
                            filing_date=filing.filing_date,
                            filing_type=filing.filing_type,
                        )
                        all_chunks.extend(chunks)
                        progress_bar.progress(
                            (idx + 1) / len(filings),
                            text=f"Chunked {idx + 1}/{len(filings)} filings...",
                        )

                    st.success(f"Created {len(all_chunks)} chunks from {len(filings)} filings!")

                    st.session_state.chunks = all_chunks
                    st.session_state.filings_metadata = [
                        {
                            "ticker": f.ticker,
                            "filing_date": f.filing_date,
                            "filing_type": f.filing_type,
                            "accession_number": f.accession_number,
                            "content": extract_text_from_html(f.content),
                        }
                        for f in filings
                    ]

                    embedding_client = get_embedding_client()
                    embed_progress = st.progress(0, text="Embedding chunks...")

                    def update_progress(current, total):
                        embed_progress.progress(
                            current / total if total else 1.0,
                            text=f"Embedding chunk {current}/{total}...",
                        )

                    embedding_client.embed_and_store(
                        all_chunks,
                        collection_name="sec_filings_embeddings_v2",
                        progress_callback=update_progress,
                    )
                    st.success("Embedding complete!")

            except Exception as e:
                st.error(f"An error occurred: {e}")
                logger.exception("Error during filing load and embed")
    else:
        st.warning("Please enter all fields.")

if "filings_metadata" in st.session_state and st.session_state.filings_metadata:
    st.subheader("Loaded Filings")
    for fm in st.session_state.filings_metadata:
        st.write(f"- {fm['ticker']} | {fm['filing_type']} | {fm['filing_date']}")

st.subheader("View Retrieved Chunks from Vector Database")
prompt_for_chunks = st.text_input("Enter a prompt to see retrieved chunks")
n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)

if st.button("Retrieve Chunks"):
    if prompt_for_chunks:
        with st.spinner("Retrieving chunks..."):
            try:
                embedding_client = get_embedding_client()
                results = embedding_client.query(
                    prompt_for_chunks,
                    collection_name="sec_filings_embeddings_v2",
                    n_results=n_results,
                    include_metadata=True,
                )
                st.session_state.retrieved_results = results
                st.session_state.last_query = prompt_for_chunks
            except Exception as e:
                st.error(f"An error occurred while retrieving chunks: {e}")
                logger.exception("Error retrieving chunks")
    else:
        st.warning("Please enter a prompt.")

if "retrieved_results" in st.session_state and st.session_state.retrieved_results:
    results = st.session_state.retrieved_results
    docs = results["documents"]
    metas = results["metadatas"]
    dists = results["distances"]

    if not docs:
        st.info("No chunks found. Please load and embed filings first.")
    else:
        st.success(f"Found {len(docs)} relevant chunks for: \"{st.session_state.get('last_query', '')}\"")

        for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists)):
            if meta.get("item_number"):
                item_info = f"Item {meta['item_number']}"
                if meta.get("item_name"):
                    item_info += f" - {meta['item_name']}"
            else:
                item_info = "General Content"

            ticker_info = meta.get("ticker", "Unknown")
            date_info = meta.get("filing_date", "Unknown")
            filing_type = meta.get("filing_type", "")

            title = f"Chunk {i + 1}: {item_info} | {ticker_info} {filing_type} ({date_info})"

            with st.expander(title, expanded=(i == 0)):
                # Cosine distance lives in [0, 2]; clamp to a [0, 1] similarity score.
                relevance = max(0.0, min(1.0, 1.0 - dist / 2.0))
                st.markdown(f"**Relevance Score:** {relevance:.2%}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Item:** {meta.get('item_number', 'N/A')} - {meta.get('item_name', 'N/A')}")
                    st.markdown(f"**Filing Type:** {meta.get('filing_type', 'N/A')}")
                with col2:
                    st.markdown(f"**Filing Date:** {meta.get('filing_date', 'N/A')}")
                    st.markdown(f"**Accession #:** {meta.get('accession_number', 'N/A')}")

                st.divider()

                display_doc = doc.replace("$", "\\$")
                max_display_chars = 2000
                if len(display_doc) > max_display_chars:
                    st.markdown(display_doc[:max_display_chars] + "...")
                    st.caption(f"(Showing first ~{max_display_chars} of {len(doc)} characters)")
                else:
                    st.markdown(display_doc)
