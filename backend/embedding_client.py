import os
import ollama
import chromadb
import logging
logging.basicConfig(format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingClient:
    def __init__(self, model: str = "mxbai-embed-large"):
        chroma_host = os.environ.get("CHROMA_HOST", "http://localhost:8000")
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.chroma_client = chromadb.HttpClient(host=chroma_host)
        self.ollama_client = ollama.Client(host=ollama_host)
        self.model = model

    def embed_and_store(self, texts: list[str], collection_name: str = "default_collection"):
        """
        Embeds a list of texts and stores them in a Chroma collection.

        Args:
            texts: A list of strings to embed.
            collection_name: The name of the Chroma collection to store the embeddings in.
        """
        collection = self.chroma_client.get_or_create_collection(collection_name)
        for i, text in enumerate(texts):
            embedding = self.ollama_client.embed(model=self.model, prompt=text)
            try:
                collection.add(
                    ids=[str(i)],
                    embeddings=[embedding["embedding"]],
                    documents=[text]
                )
            except Exception as e:
                logger.info(f"Error adding embedding for text index {i}: {e}")

    def query(self, query: str, collection_name: str = "default_collection", n_results: int = 5) -> list[str]:
        """
        Queries for similar texts in a Chroma collection.

        Args:
            query: The query string to search for.
            collection_name: The name of the Chroma collection to search in.
            n_results: The number of results to return.

        Returns:
            A list of similar texts.
        """
        collection = self.chroma_client.get_collection(collection_name)
        embedding = self.ollama_client.embed(model=self.model, prompt=query)
        results = collection.query(
            query_embeddings=[embedding["embedding"]],
            n_results=n_results
        )
        return results["documents"][0]

if __name__ == "__main__":
    # This is a placeholder for how to use the EmbeddingClient.
    # For now, we are not running this file directly.
    # client = EmbeddingClient()
    # client.embed_and_store(["This is a test document.", "This is another test document."])
    # search_results = client.query("test")
    # print(search_results)
    pass
