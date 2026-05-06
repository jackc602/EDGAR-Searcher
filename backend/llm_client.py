import os
from typing import Generator, Optional

import ollama

from backend.embedding_client import EmbeddingClient
from backend.reranker import CrossEncoderReranker, Reranker


RAG_PROMPT_TEMPLATE = """You are analyzing SEC filings. Use ONLY the following context to answer the question.
If the answer cannot be found in the provided context, say "I cannot find this information in the loaded filings."

Context:
{context}

Question: {question}

Answer (cite your sources by filing date and section when referencing specific information):"""

NO_FILINGS_PROMPT_TEMPLATE = """No SEC filings are currently loaded in the database.
Please ask the user to load filings first using the home page.

The user asked: {question}

Respond explaining that no filings are available to search."""

VALID_RERANKER_MODES = ("crossencoder", "bm25", "off")


class LLMClient:
    def __init__(self, model: str = "llama2"):
        self.model = model
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_host)
        self.embedding_client = EmbeddingClient()
        self._bm25 = Reranker()
        self._crossencoder: Optional[CrossEncoderReranker] = None
        env_mode = os.environ.get("RERANKER_MODE", "crossencoder").lower()
        self._default_reranker_mode = (
            env_mode if env_mode in VALID_RERANKER_MODES else "crossencoder"
        )

    def _get_reranker(self, mode: str):
        if mode == "off":
            return None
        if mode == "bm25":
            return self._bm25
        if mode == "crossencoder":
            if self._crossencoder is None:
                self._crossencoder = CrossEncoderReranker()
            return self._crossencoder
        raise ValueError(f"Unknown reranker mode: {mode}")

    def _format_context_with_sources(self, results: dict) -> str:
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        if not docs:
            return ""

        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            ticker = meta.get("ticker", "Unknown")
            filing_date = meta.get("filing_date", "Unknown")
            filing_type = meta.get("filing_type", "")
            item_number = meta.get("item_number", "")
            item_name = meta.get("item_name", "")

            source_info = f"[Source {i + 1}: {ticker} {filing_type} ({filing_date})"
            if item_number:
                source_info += f", Item {item_number}"
                if item_name:
                    source_info += f" - {item_name}"
            source_info += "]"
            context_parts.append(f"{source_info}\n{doc}")

        return "\n\n---\n\n".join(context_parts)

    def _build_prompt(
        self,
        question: str,
        use_rag: bool,
        collection_name: str,
        n_results: int,
        n_candidates: int,
        reranker_mode: str,
        where: Optional[dict],
    ) -> str:
        if not use_rag:
            return question

        reranker = self._get_reranker(reranker_mode)
        fetch_count = n_candidates if reranker is not None else n_results
        results = self.embedding_client.query(
            query=question,
            collection_name=collection_name,
            n_results=fetch_count,
            include_metadata=True,
            where=where,
        )

        if reranker is not None and results.get("documents"):
            results = reranker.rerank(question, results, n_final=n_results)

        context = self._format_context_with_sources(results)
        if context:
            return RAG_PROMPT_TEMPLATE.format(context=context, question=question)
        return NO_FILINGS_PROMPT_TEMPLATE.format(question=question)

    def ask(
        self,
        question: str,
        use_rag: bool = True,
        collection_name: str = "default_collection",
        n_results: int = 5,
        n_candidates: int = 10,
        reranker_mode: Optional[str] = None,
        where: Optional[dict] = None,
    ) -> str:
        mode = (reranker_mode or self._default_reranker_mode).lower()
        prompt = self._build_prompt(
            question, use_rag, collection_name, n_results, n_candidates, mode, where
        )
        response = self.ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"]

    def ask_stream(
        self,
        question: str,
        use_rag: bool = True,
        collection_name: str = "default_collection",
        n_results: int = 5,
        n_candidates: int = 10,
        reranker_mode: Optional[str] = None,
        where: Optional[dict] = None,
    ) -> Generator[str, None, None]:
        mode = (reranker_mode or self._default_reranker_mode).lower()
        prompt = self._build_prompt(
            question, use_rag, collection_name, n_results, n_candidates, mode, where
        )
        stream = self.ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]
