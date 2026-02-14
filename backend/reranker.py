"""
BM25-based keyword reranker for improving RAG retrieval quality.

Uses Reciprocal Rank Fusion (RRF) to combine vector similarity rankings
with keyword-based BM25 rankings for better retrieval results.
"""
import math
import re
from collections import Counter
from typing import List, Dict, Any


class BM25Scorer:
    """
    BM25 keyword scoring using only Python stdlib.

    BM25 is a bag-of-words retrieval function that ranks documents
    based on the query terms appearing in each document.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 scorer with tuning parameters.

        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0).
            b: Length normalization parameter (0 = no normalization, 1 = full).
        """
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into lowercase words.

        Args:
            text: Input text to tokenize.

        Returns:
            List of lowercase word tokens.
        """
        # Extract words, convert to lowercase
        words = re.findall(r'\b[a-zA-Z0-9]+\b', text.lower())
        return words

    def score(self, query: str, documents: List[str]) -> List[float]:
        """
        Compute BM25 scores for documents given a query.

        Args:
            query: Search query string.
            documents: List of document texts to score.

        Returns:
            List of BM25 scores, one per document.
        """
        if not documents:
            return []

        # Tokenize query and documents
        query_terms = self._tokenize(query)
        if not query_terms:
            return [0.0] * len(documents)

        doc_tokens = [self._tokenize(doc) for doc in documents]
        doc_lengths = [len(tokens) for tokens in doc_tokens]

        # Calculate average document length
        avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 1

        # Calculate document frequencies for IDF
        doc_freqs = Counter()
        for tokens in doc_tokens:
            unique_terms = set(tokens)
            for term in unique_terms:
                doc_freqs[term] += 1

        n_docs = len(documents)
        scores = []

        for doc_idx, tokens in enumerate(doc_tokens):
            term_freqs = Counter(tokens)
            doc_len = doc_lengths[doc_idx]
            score = 0.0

            for term in query_terms:
                if term not in term_freqs:
                    continue

                # Term frequency in document
                tf = term_freqs[term]

                # Document frequency (number of docs containing term)
                df = doc_freqs.get(term, 0)

                # IDF: log((N - df + 0.5) / (df + 0.5) + 1)
                idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1)

                # BM25 term score
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * (doc_len / avg_doc_len)
                )
                score += idf * (numerator / denominator)

            scores.append(score)

        return scores


class Reranker:
    """
    Combines vector similarity ranks with BM25 ranks using Reciprocal Rank Fusion.

    RRF is a simple but effective rank fusion method that avoids the need
    to normalize scores from different sources.
    """

    def __init__(self, k: int = 60):
        """
        Initialize reranker.

        Args:
            k: RRF constant (typically 60). Higher values give more weight
               to lower-ranked documents.
        """
        self.k = k
        self.bm25_scorer = BM25Scorer()

    def rerank(
        self,
        query: str,
        results: Dict[str, Any],
        n_final: int = 2
    ) -> Dict[str, Any]:
        """
        Rerank vector search results using BM25 keyword scoring and RRF.

        Args:
            query: Original search query.
            results: Query results dict with 'documents', 'metadatas', 'distances'.
            n_final: Number of results to return after reranking.

        Returns:
            Reranked results dict with top n_final documents.
        """
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])
        distances = results.get("distances", [])

        if not documents or len(documents) == 0:
            return results

        n_docs = len(documents)

        # Get BM25 scores
        bm25_scores = self.bm25_scorer.score(query, documents)

        # Create rankings (0-indexed, lower is better)
        # Vector rank is based on original order (already sorted by distance)
        vector_ranks = list(range(n_docs))

        # BM25 rank is based on score (higher score = lower rank)
        bm25_ranked_indices = sorted(
            range(n_docs),
            key=lambda i: bm25_scores[i],
            reverse=True
        )
        bm25_ranks = [0] * n_docs
        for rank, idx in enumerate(bm25_ranked_indices):
            bm25_ranks[idx] = rank

        # Compute RRF scores
        rrf_scores = []
        for i in range(n_docs):
            vector_rrf = 1.0 / (self.k + vector_ranks[i])
            bm25_rrf = 1.0 / (self.k + bm25_ranks[i])
            rrf_scores.append(vector_rrf + bm25_rrf)

        # Sort by RRF score (higher is better)
        reranked_indices = sorted(
            range(n_docs),
            key=lambda i: rrf_scores[i],
            reverse=True
        )[:n_final]

        # Build reranked results
        reranked_results = {
            "documents": [documents[i] for i in reranked_indices],
            "metadatas": [metadatas[i] for i in reranked_indices],
            "distances": [distances[i] for i in reranked_indices],
        }

        return reranked_results
