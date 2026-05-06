import os
import logging
from typing import Union, List, Callable, Optional

import ollama
import chromadb

from backend.document_chunker import DocumentChunk

logger = logging.getLogger(__name__)

# mxbai-embed-large requires this prefix on retrieval queries (not on documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingClient:
    def __init__(self, model: str = "mxbai-embed-large"):
        chroma_host = os.environ.get("CHROMA_HOST", "http://localhost:8000")
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.chroma_client = chromadb.HttpClient(host=chroma_host)
        self.ollama_client = ollama.Client(host=ollama_host)
        self.model = model

    def _get_collection(self, name: str):
        return self.chroma_client.get_or_create_collection(
            name,
            metadata={"hnsw:space": "cosine"},
        )

    def embed_and_store(
        self,
        chunks: List[DocumentChunk],
        collection_name: str = "default_collection",
        progress_callback: Optional[Callable] = None,
        batch_size: int = 32,
    ):
        """
        Embed DocumentChunks and store them in a Chroma collection.

        Skips chunks whose generated ID already exists in the collection so
        re-running on the same filings is a no-op.
        """
        collection = self._get_collection(collection_name)

        chunk_ids = [c.generate_id() for c in chunks]
        existing_ids = set(collection.get(ids=chunk_ids).get("ids", []))
        pending = [
            (cid, chunk)
            for cid, chunk in zip(chunk_ids, chunks)
            if cid not in existing_ids
        ]
        total = len(pending)

        if total == 0:
            if progress_callback:
                progress_callback(len(chunks), len(chunks))
            return

        for batch_start in range(0, total, batch_size):
            batch = pending[batch_start:batch_start + batch_size]
            ids = [cid for cid, _ in batch]
            texts = [chunk.text for _, chunk in batch]
            metadatas = [chunk.to_metadata_dict() for _, chunk in batch]

            try:
                result = self.ollama_client.embed(model=self.model, input=texts)
                collection.add(
                    ids=ids,
                    embeddings=result["embeddings"],
                    documents=texts,
                    metadatas=metadatas,
                )
            except Exception as e:
                logger.exception(f"Error embedding batch starting at index {batch_start}: {e}")

            if progress_callback:
                progress_callback(min(batch_start + batch_size, total), total)

    def query(
        self,
        query: str,
        collection_name: str = "default_collection",
        n_results: int = 5,
        include_metadata: bool = True,
        where: Optional[dict] = None,
    ) -> Union[List[str], dict]:
        """
        Query for similar texts in a Chroma collection.

        With include_metadata=True, returns a dict with documents, metadatas,
        and distances. Otherwise returns just the list of documents.

        `where` is passed through to Chroma; combine multiple keys with $and,
        e.g. {"$and": [{"ticker": {"$in": ["AAPL"]}}, {"filing_type": "10-K"}]}.
        """
        collection = self.chroma_client.get_collection(collection_name)
        embedding = self.ollama_client.embed(model=self.model, input=QUERY_PREFIX + query)
        query_kwargs = {
            "query_embeddings": [embedding["embeddings"][0]],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where
        results = collection.query(**query_kwargs)

        if include_metadata:
            return {
                "documents": results["documents"][0] if results["documents"] else [],
                "metadatas": results["metadatas"][0] if results["metadatas"] else [],
                "distances": results["distances"][0] if results["distances"] else [],
            }
        return results["documents"][0] if results["documents"] else []

    def delete_collection(self, collection_name: str):
        try:
            self.chroma_client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
        except Exception as e:
            logger.exception(f"Error deleting collection {collection_name}: {e}")
