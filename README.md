# EDGAR-Searcher

A fully local RAG (Retrieval-Augmented Generation) application for exploring public company filings from the SEC EDGAR system. Pick a ticker, pull down the filings, embed them into a local vector database, and chat with an LLM that answers questions grounded in those filings — all on your own machine.

## Overview

EDGAR-Searcher pulls 10-K and 10-Q filings from the SEC's EDGAR API, chunks and embeds the text into a Chroma vector database, and exposes a Streamlit UI for querying the filings through a local LLM served by Ollama. Nothing leaves your machine — no cloud APIs, no hosted models.

## Features

- **SEC filing retrieval** — Fetches 10-K and 10-Q filings for any public company by ticker and date range via the EDGAR API (`backend/edgar_client.py`).
- **Document chunking** — Splits filing HTML into meaningful chunks with item-level metadata (`backend/document_chunker.py`).
- **Local embeddings** — Uses Ollama's `mxbai-embed-large` model to embed chunks into a Chroma vector store (`backend/embedding_client.py`).
- **Hybrid retrieval with reranking** — Retrieves candidates from Chroma then reranks them with BM25 keyword scoring for better context selection (`backend/reranker.py`).
- **Local LLM chat** — Streams answers from a locally-running Ollama model with source citations back to the original filing (`backend/llm_client.py`).
- **Streamlit frontend** — A simple multi-page UI for loading filings, inspecting retrieved chunks, and chatting with the filings.

## Tech Stack

- **Python** (3.9+)
- **Streamlit** — frontend
- **Chroma** — vector database
- **Ollama** — local embedding and LLM runtime
- **requests / BeautifulSoup / lxml** — EDGAR fetching and HTML parsing

## Running Locally

### Prerequisites

1. **Python 3.9+**
2. **Ollama** installed and running locally — https://ollama.com
3. **Chroma** running locally as a server (see step 3 below)

### Quick Start

1. **Clone and install dependencies**

   ```bash
   git clone <your-repo-url>
   cd EDGAR-Searcher
   pip install -r requirements.txt
   ```

2. **Pull the required Ollama models**

   ```bash
   ollama pull mxbai-embed-large
   ollama pull gemma3:270m
   ```

   These are just default models that are local friendly.

3. **Start a local Chroma server**

   ```bash
   pip install chromadb
   chroma run --host localhost --port 8000
   ```

4. **Run the Streamlit app**

   ```bash
   streamlit run frontend/app.py
   ```

   The app will be available at http://localhost:8501.

### Run with Docker Compose

A `docker-compose.yml` is provided that spins up the Streamlit frontend, Ollama, and Chroma together:

```bash
docker compose up --build
```

Then open http://localhost:8501. Note: you'll still need to `docker exec` into the Ollama container to `ollama pull` the embedding and chat models on first run.


## Notes

- The app only fetches **10-K** and **10-Q** filings. Other form types are filtered out.
- The default chat model is `gemma3:270m`, a very small model chosen so the app runs on modest hardware. Swap to `mistral` or another pulled model from the Chat page dropdown for higher-quality answers at the cost of more RAM and slower responses.
- The SEC EDGAR API requires a User-Agent header. The current one lives in `backend/edgar_client.py` — update it to your own contact info before heavy use.
