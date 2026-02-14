import os
from typing import Generator
import ollama
from backend.embedding_client import EmbeddingClient
from backend.reranker import Reranker


class LLMClient:
    def __init__(self, model: str = "llama2"):
        self.model = model
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_host)
        self.embedding_client = EmbeddingClient()
        self.reranker = Reranker()

    def _format_context_with_sources(self, results: dict) -> str:
        """
        Format retrieved documents with source information for the prompt.

        Args:
            results: Query results with documents, metadatas, and distances.

        Returns:
            Formatted context string with source citations.
        """
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])

        if not docs:
            return ""

        context_parts = []
        for i, (doc, meta) in enumerate(zip(docs, metas)):
            # Build source citation
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

    def ask(
        self,
        question: str,
        use_rag: bool = True,
        collection_name: str = "default_collection",
        n_results: int = 2,
        n_candidates: int = 10,
        use_reranking: bool = True
    ) -> str:
        """
        Asks a question to the LLM, with an option to use RAG.

        Args:
            question: The question to ask the LLM.
            use_rag: Whether to use RAG to augment the LLM's response.
            collection_name: The name of the Chroma collection to search in for RAG.
            n_results: Number of final context chunks for the prompt.
            n_candidates: Number of initial candidates to retrieve for reranking.
            use_reranking: Whether to use BM25 reranking.

        Returns:
            The LLM's response.
        """
        if use_rag:
            # Fetch more candidates if reranking is enabled
            fetch_count = n_candidates if use_reranking else n_results
            results = self.embedding_client.query(
                query=question,
                collection_name=collection_name,
                n_results=fetch_count,
                include_metadata=True
            )

            # Rerank to get top n_results
            if use_reranking and results.get("documents"):
                results = self.reranker.rerank(question, results, n_final=n_results)

            context = self._format_context_with_sources(results)

            if context:
                prompt = f"""You are analyzing SEC filings. Use ONLY the following context to answer the question.
If the answer cannot be found in the provided context, say "I cannot find this information in the loaded filings."

Context:
{context}

Question: {question}

Answer (cite your sources by filing date and section when referencing specific information):"""
            else:
                prompt = f"""No SEC filings are currently loaded in the database.
Please ask the user to load filings first using the home page.

The user asked: {question}

Respond explaining that no filings are available to search."""
        else:
            prompt = question

        response = self.ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

    def ask_stream(
        self,
        question: str,
        use_rag: bool = True,
        collection_name: str = "default_collection",
        n_results: int = 2,
        n_candidates: int = 10,
        use_reranking: bool = True
    ) -> Generator[str, None, None]:
        """
        Asks a question to the LLM with streaming response.

        Args:
            question: The question to ask the LLM.
            use_rag: Whether to use RAG to augment the LLM's response.
            collection_name: The name of the Chroma collection to search in.
            n_results: Number of final context chunks for the prompt.
            n_candidates: Number of initial candidates to retrieve for reranking.
            use_reranking: Whether to use BM25 reranking.

        Yields:
            Chunks of the response as they are generated.
        """
        if use_rag:
            # Fetch more candidates if reranking is enabled
            fetch_count = n_candidates if use_reranking else n_results
            results = self.embedding_client.query(
                query=question,
                collection_name=collection_name,
                n_results=fetch_count,
                include_metadata=True
            )

            # Rerank to get top n_results
            if use_reranking and results.get("documents"):
                results = self.reranker.rerank(question, results, n_final=n_results)

            context = self._format_context_with_sources(results)

            if context:
                prompt = f"""You are analyzing SEC filings. Use ONLY the following context to answer the question.
If the answer cannot be found in the provided context, say "I cannot find this information in the loaded filings."

Context:
{context}

Question: {question}

Answer (cite your sources by filing date and section when referencing specific information):"""
            else:
                prompt = f"""No SEC filings are currently loaded in the database.
Please ask the user to load filings first using the home page.

The user asked: {question}

Respond explaining that no filings are available to search."""
        else:
            prompt = question

        stream = self.ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )

        for chunk in stream:
            if chunk.get("message", {}).get("content"):
                yield chunk["message"]["content"]


if __name__ == "__main__":
    pass
