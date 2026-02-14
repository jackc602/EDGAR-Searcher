import os
import ollama
import chromadb
import logging
from typing import Union, List, Callable, Optional
from backend.document_chunker import DocumentChunk

logging.basicConfig(
    format='%(filename)s:%(lineno)s:%(levelname)s -- %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, model: str = "mxbai-embed-large"):
        chroma_host = os.environ.get("CHROMA_HOST", "http://localhost:8000")
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.chroma_client = chromadb.HttpClient(host=chroma_host)
        self.ollama_client = ollama.Client(host=ollama_host)
        self.model = model

    def embed_and_store(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = "default_collection",
        progress_callback: Optional[Callable] = None
    ):
        """
        Embeds DocumentChunks and stores them in a Chroma collection with metadata.

        Args:
            chunks: A list of DocumentChunk objects to embed.
            collection_name: The name of the Chroma collection.
            progress_callback: Optional callback(current, total) for progress updates.
        """
        collection = self.chroma_client.get_or_create_collection(collection_name)
        total = len(chunks)

        for i, chunk in enumerate(chunks):
            chunk_id = chunk.generate_id()
            embedding = self.ollama_client.embed(model=self.model, input=chunk.text)

            try:
                collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding["embeddings"][0]],
                    documents=[chunk.text],
                    metadatas=[chunk.to_metadata_dict()]
                )
            except Exception as e:
                logger.info(f"Error adding embedding for chunk {chunk_id}: {e}")

            if progress_callback:
                progress_callback(i + 1, total)

    def embed_and_store_texts(
        self,
        texts: List[str],
        collection_name: str = "default_collection"
    ):
        """
        Legacy method: Embeds a list of plain texts without metadata.

        Args:
            texts: A list of strings to embed.
            collection_name: The name of the Chroma collection.
        """
        collection = self.chroma_client.get_or_create_collection(collection_name)
        for i, text in enumerate(texts):
            embedding = self.ollama_client.embed(model=self.model, input=text)
            try:
                collection.add(
                    ids=[str(i)],
                    embeddings=[embedding["embeddings"][0]],
                    documents=[text]
                )
            except Exception as e:
                logger.info(f"Error adding embedding for text index {i}: {e}")

    def query(
        self,
        query: str,
        collection_name: str = "default_collection",
        n_results: int = 5,
        include_metadata: bool = True
    ) -> Union[List[str], dict]:
        """
        Queries for similar texts in a Chroma collection.

        Args:
            query: The query string to search for.
            collection_name: The name of the Chroma collection to search in.
            n_results: The number of results to return.
            include_metadata: If True, return dict with documents, metadatas,
                              and distances. If False, return just documents.

        Returns:
            If include_metadata is True:
                dict with keys: "documents", "metadatas", "distances"
            If include_metadata is False:
                list of document strings
        """
        collection = self.chroma_client.get_collection(collection_name)
        embedding = self.ollama_client.embed(model=self.model, input=query)
        results = collection.query(
            query_embeddings=[embedding["embeddings"][0]],
            n_results=n_results,
            include=["documents", "metadatas", "distances"]
        )

        if include_metadata:
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }
        else:
            return results["documents"][0] if results["documents"] else []

    def delete_collection(self, collection_name: str):
        """Delete a collection from Chroma."""
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.info(f"Error deleting collection {collection_name}: {e}")


if __name__ == "__main__":
    pass
