"""
preprocessing.py — Text preprocessing
Options: stemming ON/OFF, stopword removal ON/OFF
"""

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

FALLBACK_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "did", "do",
    "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "it", "its", "itself", "just", "me", "more", "most", "my",
    "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s",
    "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there",
    "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves",
}


def _load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return FALLBACK_STOPWORDS

_stemmer = PorterStemmer()
_STOPWORDS = _load_stopwords()


def tokenize(text: str) -> list[str]:
    """Lowercase + remove punctuation + tokenize."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


def preprocess(
    text: str,
    use_stemming: bool = True,
    use_stopword: bool = True,
) -> list[str]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    text         : raw text string
    use_stemming : apply Porter stemmer
    use_stopword : remove English stopwords

    Returns
    -------
    list of processed tokens
    """
    tokens = tokenize(text)

    if use_stopword:
        tokens = [t for t in tokens if t not in _STOPWORDS]

    # Remove very short tokens (length <= 1)
    tokens = [t for t in tokens if len(t) > 1]

    if use_stemming:
        tokens = [_stemmer.stem(t) for t in tokens]

    return tokens


def preprocess_docs(
    documents: dict,
    use_stemming: bool = True,
    use_stopword: bool = True,
) -> dict[int, list[str]]:
    """
    Preprocess all documents.
    Combines title + abstract for richer representation.

    Returns dict {doc_id: [tokens]}
    """
    processed = {}
    for doc_id, doc in documents.items():
        combined = f"{doc.get('title', '')} {doc.get('abstract', '')}"
        processed[doc_id] = preprocess(combined, use_stemming, use_stopword)
    return processed


def preprocess_queries(
    queries: dict,
    use_stemming: bool = True,
    use_stopword: bool = True,
) -> dict[int, list[str]]:
    """
    Preprocess all queries.

    Returns dict {query_id: [tokens]}
    """
    processed = {}
    for q_id, q_text in queries.items():
        processed[q_id] = preprocess(q_text, use_stemming, use_stopword)
    return processed


if __name__ == "__main__":
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.parser.parser import parse_documents, parse_queries

    base = Path(__file__).parent.parent / "data"
    docs = parse_documents(str(base / "cisi.all"))
    queries = parse_queries(str(base / "query.text"))

    proc_docs = preprocess_docs(docs)
    proc_q = preprocess_queries(queries)

    print(f"Processed {len(proc_docs)} docs")
    print(f"Doc #1 tokens (first 15): {proc_docs[1][:15]}")
    print(f"Query #1 tokens: {proc_q[1][:15]}")
