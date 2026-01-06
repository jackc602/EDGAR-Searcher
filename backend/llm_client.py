import os
import ollama
from backend.embedding_client import EmbeddingClient

class LLMClient:
    def __init__(self, model: str = "llama2"):
        self.model = model
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.ollama_client = ollama.Client(host=ollama_host)
        self.embedding_client = EmbeddingClient()

    def ask(self, question: str, use_rag: bool = True, collection_name: str = "default_collection") -> str:
        """
        Asks a question to the LLM, with an option to use RAG.

        Args:
            question: The question to ask the LLM.
            use_rag: Whether to use RAG to augment the LLM's response.
            collection_name: The name of the Chroma collection to search in for RAG.

        Returns:
            The LLM's response.
        """
        if use_rag:
            context_documents = self.embedding_client.search(query=question, collection_name=collection_name)
            context = "\n".join(context_documents)
            prompt = f"Using the following context, please answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        else:
            prompt = question

        response = self.ollama_client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"]

if __name__ == "__main__":
    # This is a placeholder for how to use the LLMClient.
    # For now, we are not running this file directly.
    # llm_client = LLMClient()
    # response = llm_client.ask("What is the capital of France?")
    # print(response)
    # response_with_rag = llm_client.ask("What is a test document about?", use_rag=True)
    # print(response_with_rag)
    pass
