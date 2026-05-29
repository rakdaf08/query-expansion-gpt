"""
main.py — IR System with Query Expansion
CISI Collection | TF-IDF + Cosine Similarity | HuggingFace/Anthropic QE

Usage:
  python main.py                    # interactive menu
  python main.py --batch            # run all queries, save to output/
  python main.py --query "your text here"   # single query
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

from src.parser.parser import parse_documents, parse_queries, parse_qrels
from src.preprocessing.preprocessing import preprocess_docs, preprocess_queries, preprocess
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.retrieval import Retriever, display_results
from src.evaluation.evaluation import (
    evaluate_batch,
    evaluate_single,
    print_comparison,
    average_precision,
)


# ── Global state (loaded once) ────────────────────────────────────────────────
_STATE = {}


def load_system(
    use_stemming: bool = True,
    use_stopword: bool = True,
    tf_scheme: str = "log",
) -> dict:
    """Load and build the full IR system. Returns state dict."""
    print("\n" + "═" * 55)
    print("  Initializing IR System")
    print("═" * 55)

    print("  [1/5] Parsing dataset...")
    docs = parse_documents(str(DATA / "cisi.all"))
    queries = parse_queries(str(DATA / "query.text"))
    qrels = parse_qrels(str(DATA / "qrels.text"))
    print(f"        {len(docs)} docs | {len(queries)} queries | {len(qrels)} qrels")

    print("  [2/5] Preprocessing documents...")
    proc_docs = preprocess_docs(docs, use_stemming, use_stopword)
    proc_queries = preprocess_queries(queries, use_stemming, use_stopword)

    print("  [3/5] Building inverted index...")
    idx = InvertedIndex()
    idx.build(proc_docs)

    print("  [4/5] Building document vectors (TF scheme: {})...".format(tf_scheme))
    retriever = Retriever(idx, tf_scheme=tf_scheme)
    retriever.build_doc_vectors()

    print("  [5/5] System ready.\n")

    return {
        "docs": docs,
        "queries": queries,
        "qrels": qrels,
        "proc_queries": proc_queries,
        "idx": idx,
        "retriever": retriever,
        "use_stemming": use_stemming,
        "use_stopword": use_stopword,
        "tf_scheme": tf_scheme,
    }


def get_expander(model_name: str = None, hf_token: str = None) -> object:
    """Lazy-load query expander."""
    from src.qe.gpt_expansion import QueryExpander
    kwargs = {}
    if model_name:
        kwargs["model_name"] = model_name
    if hf_token:
        kwargs["hf_token"] = hf_token
    return QueryExpander(**kwargs)


# ── Single query flow ──────────────────────────────────────────────────────────

def run_single_query(
    query_text: str,
    state: dict,
    top_k: int = 10,
    expand: bool = False,
    n_terms: int = 5,
    expander=None,
    query_id: int = None,
) -> dict:
    """
    Run retrieval for a single query string.
    Returns result dict with rankings and optionally MAP.
    """
    retriever: Retriever = state["retriever"]
    docs = state["docs"]
    qrels = state["qrels"]
    use_stemming = state["use_stemming"]
    use_stopword = state["use_stopword"]

    # Preprocess
    tokens = preprocess(query_text, use_stemming, use_stopword)

    print(f"\n  Query: {query_text}")
    print(f"  Tokens: {tokens[:12]}{'...' if len(tokens) > 12 else ''}")

    # Retrieve
    all_results = retriever.retrieve_all(tokens)
    top_results = all_results[:top_k]

    print(f"\n  ── Retrieval Results (top {top_k}) ──")
    display_results(top_results, docs, top_k)

    # MAP for this query
    ap_before = None
    if query_id and query_id in qrels:
        ap_before = average_precision(all_results, qrels[query_id])
        print(f"\n  AP (before QE): {ap_before:.4f}")

    output = {
        "query": query_text,
        "tokens": tokens,
        "results_before": top_results,
        "ap_before": ap_before,
    }

    if expand and expander:
        try:
            expanded_query, weights = expander.expand(query_text, n_terms=n_terms)
            expander.display_expansion(query_text, expanded_query, weights)

            exp_tokens = preprocess(expanded_query, use_stemming, use_stopword)
            all_results_exp = retriever.retrieve_all(exp_tokens)
            top_results_exp = all_results_exp[:top_k]

            print(f"\n  ── Retrieval Results After Expansion (top {top_k}) ──")
            display_results(top_results_exp, docs, top_k)

            ap_after = None
            if query_id and query_id in qrels:
                ap_after = average_precision(all_results_exp, qrels[query_id])
                print(f"\n  AP (after  QE): {ap_after:.4f}")
                if ap_before is not None:
                    delta = ap_after - ap_before
                    sym = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
                    print(f"  Δ AP          : {sym} {abs(delta):.4f}")

            output["expanded_query"] = expanded_query
            output["expansion_weights"] = weights
            output["results_after"] = top_results_exp
            output["ap_after"] = ap_after
        except Exception as e:
            print(f"\n  ⚠️ Warning: Query expansion gagal ({e}). Melanjutkan tanpa ekspansi.")

    return output


# ── Batch query flow ───────────────────────────────────────────────────────────

def run_batch(state: dict, expand: bool = False, n_terms: int = 5, expander=None):
    """Run all queries and compute MAP. Save results to output/."""
    proc_queries = state["proc_queries"]
    qrels = state["qrels"]
    queries = state["queries"]

    print("\n  Running batch retrieval on all queries...")
    map_before, metrics_before = evaluate_batch(
        state["retriever"], proc_queries, qrels, label="Original"
    )
    print(f"  MAP (before QE): {map_before:.4f}")

    map_after = None
    metrics_after = None

    if expand and expander:
        print("\n  Running batch retrieval with query expansion...")
        use_stemming = state["use_stemming"]
        use_stopword = state["use_stopword"]

        expanded_proc = {}
        for q_id, q_text in queries.items():
            if q_id not in qrels:
                continue
            try:
                exp_text, _ = expander.expand(q_text, n_terms=n_terms)
                expanded_proc[q_id] = preprocess(exp_text, use_stemming, use_stopword)
                print(f"    Query {q_id} expanded.", end="\r")
            except Exception as e:
                print(f"    Warning: Query {q_id} expansion failed ({e}), using original.")
                expanded_proc[q_id] = proc_queries[q_id]

        map_after, metrics_after = evaluate_batch(
            state["retriever"], expanded_proc, qrels, label="Expanded"
        )
        print(f"\n  MAP (after  QE): {map_after:.4f}")

    if map_after is not None:
        print_comparison(map_before, map_after)

    # Save results
    out_path = OUTPUT / "batch_results.json"
    with open(out_path, "w") as f:
        json.dump(
            {
                "map_before": map_before,
                "map_after": map_after,
                "metrics_before": {str(k): v for k, v in metrics_before.items()},
                "metrics_after": (
                    {str(k): v for k, v in metrics_after.items()}
                    if metrics_after
                    else None
                ),
            },
            f,
            indent=2,
        )
    print(f"\n  ✓ Results saved to {out_path}")

    return map_before, map_after


# ── Interactive menu ───────────────────────────────────────────────────────────

def show_menu():
    print("\n" + "═" * 55)
    print("  IR System — CISI Collection")
    print("═" * 55)
    print("  [1] Single query retrieval")
    print("  [2] Single query + Query Expansion")
    print("  [3] Batch retrieval (all queries) + MAP")
    print("  [4] Batch retrieval + QE + MAP comparison")
    print("  [5] View inverted index for a term")
    print("  [6] View query from dataset")
    print("  [0] Exit")
    print("─" * 55)


def interactive_mode(state: dict):
    expander = None

    while True:
        show_menu()
        choice = input("  Choice: ").strip()

        if choice == "0":
            print("  Bye!")
            break

        elif choice in ("1", "2"):
            query_text = input("  Enter query: ").strip()
            if not query_text:
                print("  No query entered.")
                continue

            q_id_str = input("  Query ID for AP calculation (press Enter to skip): ").strip()
            q_id = int(q_id_str) if q_id_str.isdigit() else None

            top_k_str = input("  Top-K results [default 10]: ").strip()
            top_k = int(top_k_str) if top_k_str.isdigit() else 10

            expand = choice == "2"
            if expand:
                if expander is None:
                    print("  Expander: HuggingFace Inference API (Cloud)")
                    token_input = input("  HF Token (Enter = pakai env HF_TOKEN): ").strip()
                    hf_token = token_input or None
                    expander = get_expander(hf_token=hf_token)
                n_terms_str = input("  Number of expansion terms [default 5]: ").strip()
                n_terms = int(n_terms_str) if n_terms_str.isdigit() else 5
                select_all = input("  Use all generated terms? [y/N]: ").strip().lower() == "y"
                if select_all:
                    n_terms = 20
            else:
                n_terms = 5

            run_single_query(
                query_text, state, top_k=top_k,
                expand=expand, n_terms=n_terms,
                expander=expander if expand else None,
                query_id=q_id,
            )

        elif choice == "3":
            run_batch(state, expand=False)

        elif choice == "4":
            if expander is None:
                print("  Expander: HuggingFace Inference API (Cloud)")
                token_input = input("  HF Token (Enter = pakai env HF_TOKEN): ").strip()
                hf_token = token_input or None
                expander = get_expander(hf_token=hf_token)
            n_terms_str = input("  Number of expansion terms [default 5]: ").strip()
            n_terms = int(n_terms_str) if n_terms_str.isdigit() else 5
            run_batch(state, expand=True, n_terms=n_terms, expander=expander)

        elif choice == "5":
            term = input("  Enter term to inspect: ").strip().lower()
            state["idx"].show_inverted_index(term)

        elif choice == "6":
            q_id_str = input("  Query ID (1–112): ").strip()
            if q_id_str.isdigit():
                q_id = int(q_id_str)
                q_text = state["queries"].get(q_id)
                if q_text:
                    print(f"\n  Query #{q_id}:\n  {q_text}")
                    rel = state["qrels"].get(q_id, set())
                    print(f"  Relevant docs: {sorted(rel)[:10]}{'...' if len(rel) > 10 else ''} ({len(rel)} total)")
                else:
                    print(f"  Query {q_id} not found.")
            else:
                print("  Invalid ID.")

        else:
            print("  Unknown option.")


# ── CLI args mode ──────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="IR System with Query Expansion")
    parser.add_argument("--query", type=str, help="Single query text")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation")
    parser.add_argument("--expand", action="store_true", help="Enable query expansion")
    parser.add_argument("--n-terms", type=int, default=5, help="Expansion term count")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results")
    parser.add_argument("--tf", choices=["raw", "binary", "log", "augmented"], default="log")
    parser.add_argument("--no-stemming", action="store_true")
    parser.add_argument("--no-stopword", action="store_true")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    return parser.parse_args()


def main():
    args = parse_args()

    state = load_system(
        use_stemming=not args.no_stemming,
        use_stopword=not args.no_stopword,
        tf_scheme=args.tf,
    )

    expander = None
    if args.expand:
        expander = get_expander(hf_token=getattr(args, "hf_token", None))

    if args.query:
        run_single_query(
            args.query, state,
            top_k=args.top_k,
            expand=args.expand,
            n_terms=args.n_terms,
            expander=expander,
        )
    elif args.batch:
        run_batch(state, expand=args.expand, n_terms=args.n_terms, expander=expander)
    else:
        # Interactive mode
        interactive_mode(state)


if __name__ == "__main__":
    main()
