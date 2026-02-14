import os
import ollama
from backend.embedding_client import EmbeddingClient


class LLMClient:
    def __init__(self, model: str = "llama2"):
        self.model = model
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_host)
        self.embedding_client = EmbeddingClient()

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
        n_results: int = 5
    ) -> str:
        """
        Asks a question to the LLM, with an option to use RAG.

        Args:
            question: The question to ask the LLM.
            use_rag: Whether to use RAG to augment the LLM's response.
            collection_name: The name of the Chroma collection to search in for RAG.
            n_results: Number of context chunks to retrieve.

        Returns:
            The LLM's response.
        """
        if use_rag:
            results = self.embedding_client.query(
                query=question,
                collection_name=collection_name,
                n_results=n_results,
                include_metadata=True
            )
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


if __name__ == "__main__":
    pass
