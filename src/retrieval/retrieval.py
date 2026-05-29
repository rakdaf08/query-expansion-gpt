"""
retrieval.py — TF-IDF retrieval with cosine similarity
"""

import math
from collections import defaultdict


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    if not vec_a or not vec_b:
        return 0.0

    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vec_b)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class Retriever:
    def __init__(self, inverted_index, tf_scheme: str = "log"):
        """
        Parameters
        ----------
        inverted_index : InvertedIndex instance
        tf_scheme      : one of 'raw', 'binary', 'log', 'augmented'
        """
        self.index = inverted_index
        self.tf_scheme = tf_scheme
        self._doc_vectors: dict[int, dict[str, float]] = {}

    def build_doc_vectors(self):
        """Pre-compute all document TF-IDF vectors (call once after index build)."""
        print("Building document vectors...", end=" ", flush=True)
        for doc_id in self.index.doc_lengths:
            self._doc_vectors[doc_id] = self.index.get_doc_vector(doc_id, self.tf_scheme)
        print("done.")

    def retrieve(
        self,
        query_tokens: list[str],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        Retrieve top_k documents for given query tokens.

        Returns
        -------
        list of (doc_id, score) sorted by score descending
        """
        query_vec = self.index.get_query_vector(query_tokens, self.tf_scheme)
        if not query_vec:
            return []

        scores: dict[int, float] = {}

        # Only score docs that share at least one term with query
        candidate_docs: set[int] = set()
        for term in query_vec:
            if term in self.index.index:
                candidate_docs.update(self.index.index[term].keys())

        for doc_id in candidate_docs:
            doc_vec = self._doc_vectors.get(doc_id, {})
            sim = cosine_similarity(query_vec, doc_vec)
            if sim > 0:
                scores[doc_id] = sim

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]

    def retrieve_all(
        self,
        query_tokens: list[str],
    ) -> list[tuple[int, float]]:
        """Retrieve ALL scored documents (needed for MAP calculation)."""
        query_vec = self.index.get_query_vector(query_tokens, self.tf_scheme)
        if not query_vec:
            return []

        candidate_docs: set[int] = set()
        for term in query_vec:
            if term in self.index.index:
                candidate_docs.update(self.index.index[term].keys())

        scores: dict[int, float] = {}
        for doc_id in candidate_docs:
            doc_vec = self._doc_vectors.get(doc_id, {})
            sim = cosine_similarity(query_vec, doc_vec)
            if sim > 0:
                scores[doc_id] = sim

        return sorted(scores.items(), key=lambda x: -x[1])


def display_results(
    results: list[tuple[int, float]],
    documents: dict,
    top_k: int = 10,
):
    """Pretty-print retrieval results."""
    print(f"\n{'Rank':<6} {'DocID':<8} {'Score':<10} {'Title'}")
    print("-" * 70)
    for rank, (doc_id, score) in enumerate(results[:top_k], 1):
        title = documents.get(doc_id, {}).get("title", "N/A")[:50]
        print(f"{rank:<6} {doc_id:<8} {score:<10.4f} {title}")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.parser.parser import parse_documents, parse_queries
    from src.preprocessing.preprocessing import preprocess_docs, preprocess_queries
    from src.indexing.inverted_index import InvertedIndex

    base = Path(__file__).parent.parent / "data"
    docs = parse_documents(str(base / "cisi.all"))
    queries = parse_queries(str(base / "query.text"))

    proc_docs = preprocess_docs(docs)
    proc_queries = preprocess_queries(queries)

    idx = InvertedIndex()
    idx.build(proc_docs)

    retriever = Retriever(idx, tf_scheme="log")
    retriever.build_doc_vectors()

    print("\n=== Query #1 ===")
    print(f"Query: {queries[1][:80]}")
    results = retriever.retrieve(proc_queries[1], top_k=10)
    display_results(results, docs)
