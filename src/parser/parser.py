"""
parser.py — Parsing dataset CISI
Handles: cisi.all, query.text, qrels.text
"""

import re
from pathlib import Path


def parse_documents(filepath: str) -> dict[int, dict]:
    """
    Parse cisi.all → dict {doc_id: {title, author, abstract}}
    Format: .I <id> .T <title> .A <author> .W <abstract> .X <refs>
    """
    documents = {}
    current_id = None
    current_field = None
    buffer = []

    doc = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I "):
                # Save previous doc
                if current_id is not None:
                    if current_field and buffer:
                        doc[current_field] = " ".join(buffer).strip()
                    documents[current_id] = doc

                current_id = int(line[3:].strip())
                doc = {"id": current_id, "title": "", "author": "", "abstract": ""}
                current_field = None
                buffer = []

            elif line.startswith(".T"):
                if current_field and buffer:
                    doc[current_field] = " ".join(buffer).strip()
                current_field = "title"
                buffer = []

            elif line.startswith(".A"):
                if current_field and buffer:
                    doc[current_field] = " ".join(buffer).strip()
                current_field = "author"
                buffer = []

            elif line.startswith(".W"):
                if current_field and buffer:
                    doc[current_field] = " ".join(buffer).strip()
                current_field = "abstract"
                buffer = []

            elif line.startswith(".X"):
                if current_field and buffer:
                    doc[current_field] = " ".join(buffer).strip()
                current_field = None
                buffer = []

            elif line.startswith(".B"):
                if current_field and buffer:
                    doc[current_field] = " ".join(buffer).strip()
                current_field = None
                buffer = []

            else:
                if current_field is not None and line.strip():
                    buffer.append(line.strip())

    # Save last doc
    if current_id is not None:
        if current_field and buffer:
            doc[current_field] = " ".join(buffer).strip()
        documents[current_id] = doc

    return documents


def parse_queries(filepath: str) -> dict[int, str]:
    """
    Parse query.text → dict {query_id: query_text}
    Format: .I <id> .W <text>
    """
    queries = {}
    current_id = None
    current_field = None
    buffer = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith(".I "):
                if current_id is not None and buffer:
                    queries[current_id] = " ".join(buffer).strip()
                current_id = int(line[3:].strip())
                current_field = None
                buffer = []

            elif line.startswith(".W"):
                current_field = "text"
                buffer = []

            else:
                if current_field == "text" and line.strip():
                    buffer.append(line.strip())

    # Save last query
    if current_id is not None and buffer:
        queries[current_id] = " ".join(buffer).strip()

    return queries


def parse_query_file(filepath: str) -> dict[int, str]:
    """
    Parse an external batch query file.

    Supports CISI-style .I/.W files. If no CISI markers are found, falls back
    to one query per non-empty line with auto-generated IDs starting from 1.
    """
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()

    if ".I " in text and ".W" in text:
        return parse_queries(filepath)

    queries = {}
    for i, line in enumerate(text.splitlines(), 1):
        query = line.strip()
        if query:
            queries[len(queries) + 1] = query
    return queries


def parse_qrels(filepath: str) -> dict[int, set[int]]:
    """
    Parse qrels.text → dict {query_id: set of relevant doc_ids}
    Format: query_id doc_id col3 col4
    """
    qrels = {}

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                q_id = int(parts[0])
                doc_id = int(parts[1])
                if q_id not in qrels:
                    qrels[q_id] = set()
                qrels[q_id].add(doc_id)

    return qrels


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "data"

    docs = parse_documents(str(base / "cisi.all"))
    queries = parse_queries(str(base / "query.text"))
    qrels = parse_qrels(str(base / "qrels.text"))

    print(f"✓ Documents  : {len(docs)}")
    print(f"✓ Queries    : {len(queries)}")
    print(f"✓ Qrels      : {len(qrels)} queries with relevance judgements")
    print(f"\nContoh dokumen #1:")
    d = docs[1]
    print(f"  Title   : {d['title']}")
    print(f"  Author  : {d['author']}")
    print(f"  Abstract: {d['abstract'][:120]}...")
    print(f"\nContoh query #1: {queries[1][:100]}...")
    print(f"Relevant docs for query 1: {sorted(list(qrels[1]))[:5]}...")
