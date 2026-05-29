"""
inverted_index.py — Inverted index + TF/IDF weighting
Supports: raw TF, binary TF, log TF, augmented TF, IDF, TF-IDF
"""

import math
from collections import defaultdict


class InvertedIndex:
    def __init__(self):
        # {term: {doc_id: raw_tf}}
        self.index: dict[str, dict[int, int]] = defaultdict(dict)
        # {doc_id: total_token_count}
        self.doc_lengths: dict[int, int] = {}
        # {doc_id: max_tf_in_doc}
        self.doc_max_tf: dict[int, int] = {}
        # total docs
        self.N: int = 0
        # {term: df}
        self.df: dict[str, int] = {}
        # vocabulary list
        self.vocabulary: list[str] = []

    def build(self, processed_docs: dict[int, list[str]]):
        """Build the inverted index from preprocessed documents."""
        self.N = len(processed_docs)
        temp_index: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

        for doc_id, tokens in processed_docs.items():
            self.doc_lengths[doc_id] = len(tokens)
            for token in tokens:
                temp_index[token][doc_id] += 1

        # Compute max TF per doc
        for doc_id in processed_docs:
            self.doc_max_tf[doc_id] = 0

        for term, postings in temp_index.items():
            self.index[term] = dict(postings)
            self.df[term] = len(postings)
            for doc_id, tf in postings.items():
                if tf > self.doc_max_tf[doc_id]:
                    self.doc_max_tf[doc_id] = tf

        self.vocabulary = sorted(self.index.keys())
        print(f"✓ Index built: {len(self.vocabulary)} unique terms, {self.N} documents")

    # ─── TF weighting schemes ────────────────────────────────────────────────

    def tf_raw(self, term: str, doc_id: int) -> float:
        return float(self.index.get(term, {}).get(doc_id, 0))

    def tf_binary(self, term: str, doc_id: int) -> float:
        return 1.0 if self.index.get(term, {}).get(doc_id, 0) > 0 else 0.0

    def tf_log(self, term: str, doc_id: int) -> float:
        raw = self.index.get(term, {}).get(doc_id, 0)
        return 1.0 + math.log(raw) if raw > 0 else 0.0

    def tf_augmented(self, term: str, doc_id: int) -> float:
        raw = self.index.get(term, {}).get(doc_id, 0)
        max_tf = self.doc_max_tf.get(doc_id, 1) or 1
        return 0.5 + 0.5 * (raw / max_tf) if raw > 0 else 0.0

    def idf(self, term: str) -> float:
        df = self.df.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(self.N / df)

    def tfidf(self, term: str, doc_id: int, tf_scheme: str = "log") -> float:
        tf_fn = {
            "raw": self.tf_raw,
            "binary": self.tf_binary,
            "log": self.tf_log,
            "augmented": self.tf_augmented,
        }.get(tf_scheme, self.tf_log)
        return tf_fn(term, doc_id) * self.idf(term)

    # ─── Document vector builder ──────────────────────────────────────────────

    def get_doc_vector(self, doc_id: int, tf_scheme: str = "log") -> dict[str, float]:
        """Return TF-IDF vector for a document (only terms present in doc)."""
        vec = {}
        for term, postings in self.index.items():
            if doc_id in postings:
                vec[term] = self.tfidf(term, doc_id, tf_scheme)
        return vec

    def get_query_vector(self, tokens: list[str], tf_scheme: str = "log") -> dict[str, float]:
        """
        Build TF-IDF vector for a query.
        Uses same TF scheme as documents.
        """
        raw_tf: dict[str, int] = defaultdict(int)
        for t in tokens:
            raw_tf[t] += 1

        vec = {}
        for term, rf in raw_tf.items():
            idf_val = self.idf(term)
            if idf_val == 0:
                continue  # term not in corpus → skip
            if tf_scheme == "raw":
                tf_val = float(rf)
            elif tf_scheme == "binary":
                tf_val = 1.0
            elif tf_scheme == "log":
                tf_val = 1.0 + math.log(rf) if rf > 0 else 0.0
            elif tf_scheme == "augmented":
                max_tf = max(raw_tf.values()) or 1
                tf_val = 0.5 + 0.5 * (rf / max_tf)
            else:
                tf_val = 1.0 + math.log(rf) if rf > 0 else 0.0
            vec[term] = tf_val * idf_val
        return vec

    # ─── Inverted index viewer ────────────────────────────────────────────────

    def show_inverted_index(self, term: str, top_n: int = 20):
        """Display posting list for a term."""
        postings = self.index.get(term, {})
        if not postings:
            print(f"Term '{term}' not found in index.")
            return

        print(f"\nInverted Index for term: '{term}'")
        print(f"Document Frequency (DF): {self.df[term]}")
        print(f"IDF: {self.idf(term):.4f}")
        print(f"{'DocID':<10} {'Raw TF':<10} {'Log TF':<10} {'TF-IDF (log)':<15}")
        print("-" * 45)
        sorted_postings = sorted(postings.items(), key=lambda x: -x[1])
        for doc_id, tf in sorted_postings[:top_n]:
            log_tf = self.tf_log(term, doc_id)
            tfidf = self.tfidf(term, doc_id)
            print(f"{doc_id:<10} {tf:<10} {log_tf:<10.4f} {tfidf:<15.4f}")
        if len(postings) > top_n:
            print(f"  ... and {len(postings) - top_n} more docs")


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.parser.parser import parse_documents
    from src.preprocessing.preprocessing import preprocess_docs

    base = Path(__file__).parent.parent / "data"
    docs = parse_documents(str(base / "cisi.all"))
    proc_docs = preprocess_docs(docs)

    idx = InvertedIndex()
    idx.build(proc_docs)
    idx.show_inverted_index("retriev")
    idx.show_inverted_index("inform")
