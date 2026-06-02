"""
main.py - IR System with Query Expansion using OpenAI GPT.

Usage:
  python main.py
  python main.py --query "information retrieval"
  python main.py --batch
  python main.py --batch --expand --query-file queries.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))

from src.evaluation.evaluation import average_precision, evaluate_single, print_comparison
from src.indexing.inverted_index import InvertedIndex
from src.parser.parser import parse_documents, parse_queries, parse_query_file, parse_qrels
from src.preprocessing.preprocessing import preprocess, preprocess_docs, preprocess_queries
from src.retrieval.retrieval import Retriever, display_results


TF_SCHEMES = ("raw", "binary", "log", "augmented")
WEIGHTING_MODES = ("tf", "idf", "tfidf", "tfidf_cosine")


def load_system(
    use_stemming: bool = True,
    use_stopword: bool = True,
    tf_scheme: str = "log",
    weighting_mode: str = "tfidf_cosine",
    query_file: str | None = None,
) -> dict:
    """Load dataset, preprocess text, build index, and return runtime state."""
    print("\n" + "=" * 55)
    print("  Initializing IR System")
    print("=" * 55)

    print("  [1/5] Parsing dataset...")
    docs = parse_documents(str(DATA / "cisi.all"))
    queries = parse_query_file(query_file) if query_file else parse_queries(str(DATA / "query.text"))
    qrels = parse_qrels(str(DATA / "qrels.text"))
    print(f"        {len(docs)} docs | {len(queries)} queries | {len(qrels)} qrels")

    print("  [2/5] Preprocessing documents and queries...")
    proc_docs = preprocess_docs(docs, use_stemming, use_stopword)
    proc_queries = preprocess_queries(queries, use_stemming, use_stopword)

    print("  [3/5] Building inverted index...")
    idx = InvertedIndex()
    idx.build(proc_docs)

    print(
        "  [4/5] Building document vectors "
        f"(TF: {tf_scheme}, weighting: {weighting_mode})..."
    )
    retriever = Retriever(idx, tf_scheme=tf_scheme, weighting_mode=weighting_mode)
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
        "weighting_mode": weighting_mode,
        "query_file": query_file,
    }


def get_expander(
    provider: str = "openai",
    model_name: str | None = None,
    openai_api_key: str | None = None,
    hf_token: str | None = None,
):
    """Lazy-load query expander."""
    from src.qe.gpt_expansion import QueryExpander

    kwargs = {"provider": provider}
    if model_name:
        kwargs["model_name"] = model_name
    if openai_api_key:
        kwargs["openai_api_key"] = openai_api_key
    if hf_token:
        kwargs["hf_token"] = hf_token
    return QueryExpander(**kwargs)


def build_weighted_expansion_tokens(
    original_query: str,
    expansion_weights: dict[str, float],
    use_stemming: bool,
    use_stopword: bool,
) -> tuple[list[str], dict[str, float]]:
    """Preprocess original + expansion terms and map GPT weights to query tokens."""
    original_tokens = preprocess(original_query, use_stemming, use_stopword)
    original_token_set = set(original_tokens)
    tokens = list(original_tokens)
    token_weights = {token: 1.0 for token in original_tokens}

    for term, weight in expansion_weights.items():
        expansion_tokens = preprocess(term, use_stemming, use_stopword)
        for token in expansion_tokens:
            if token in original_token_set:
                token_weights[token] = 1.0
            else:
                token_weights[token] = max(token_weights.get(token, 0.0), float(weight))
                tokens.append(token)

    return tokens, token_weights


def serialize_results(results: list[tuple[int, float]], documents: dict) -> list[dict]:
    """Convert ranked tuples to JSON/CSV-friendly dictionaries."""
    rows = []
    for rank, (doc_id, score) in enumerate(results, 1):
        doc = documents.get(doc_id, {})
        rows.append(
            {
                "rank": rank,
                "doc_id": doc_id,
                "score": round(float(score), 8),
                "title": doc.get("title", "N/A"),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    """Write rows to CSV even when the row list is empty."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def run_single_query(
    query_text: str,
    state: dict,
    top_k: int = 10,
    expand: bool = False,
    n_terms: int = 5,
    all_terms: bool = False,
    expander=None,
    query_id: int | None = None,
) -> dict:
    """Run retrieval for a single query string."""
    retriever: Retriever = state["retriever"]
    docs = state["docs"]
    qrels = state["qrels"]
    use_stemming = state["use_stemming"]
    use_stopword = state["use_stopword"]

    tokens = preprocess(query_text, use_stemming, use_stopword)

    print(f"\n  Query: {query_text}")
    print(f"  Tokens: {tokens[:12]}{'...' if len(tokens) > 12 else ''}")

    all_results = retriever.retrieve_all(tokens)
    top_results = all_results[:top_k]

    print(f"\n  -- Retrieval Results (top {top_k}) --")
    display_results(top_results, docs, top_k)

    ap_before = None
    if query_id and query_id in qrels:
        ap_before = average_precision(all_results, qrels[query_id])
        print(f"\n  AP (before QE): {ap_before:.4f}")

    output = {
        "query": query_text,
        "tokens": tokens,
        "results_before": serialize_results(top_results, docs),
        "ap_before": ap_before,
    }

    if expand and expander:
        try:
            expanded_query, weights = expander.expand(
                query_text,
                n_terms=n_terms,
                select_all=all_terms,
            )
            expander.display_expansion(query_text, expanded_query, weights)

            exp_tokens, token_weights = build_weighted_expansion_tokens(
                query_text,
                weights,
                use_stemming,
                use_stopword,
            )
            all_results_exp = retriever.retrieve_all(
                exp_tokens,
                query_term_weights=token_weights,
            )
            top_results_exp = all_results_exp[:top_k]

            print(f"\n  -- Retrieval Results After Expansion (top {top_k}) --")
            display_results(top_results_exp, docs, top_k)

            ap_after = None
            if query_id and query_id in qrels:
                ap_after = average_precision(all_results_exp, qrels[query_id])
                print(f"\n  AP (after  QE): {ap_after:.4f}")
                if ap_before is not None:
                    delta = ap_after - ap_before
                    symbol = "+" if delta > 0 else ("-" if delta < 0 else "=")
                    print(f"  Delta AP      : {symbol} {abs(delta):.4f}")

            output.update(
                {
                    "expanded_query": expanded_query,
                    "expansion_weights": weights,
                    "expansion_token_weights": token_weights,
                    "results_after": serialize_results(top_results_exp, docs),
                    "ap_after": ap_after,
                }
            )
        except Exception as exc:
            print(f"\n  Warning: Query expansion failed ({exc}). Continuing without expansion.")

    return output


def _map_from_reports(reports: list[dict], key: str) -> float | None:
    aps = [report[key] for report in reports if report.get(key) is not None]
    return sum(aps) / len(aps) if aps else None


def run_batch(
    state: dict,
    expand: bool = False,
    n_terms: int = 5,
    all_terms: bool = False,
    expander=None,
    top_k: int = 10,
):
    """Run batch retrieval, compute MAP, and save JSON + CSV outputs."""
    proc_queries = state["proc_queries"]
    qrels = state["qrels"]
    queries = state["queries"]
    docs = state["docs"]
    retriever: Retriever = state["retriever"]

    print("\n  Running batch retrieval...")
    query_reports = []
    original_ranking_rows = []
    expanded_ranking_rows = []

    for q_id, q_text in queries.items():
        tokens = proc_queries[q_id]
        results_before = retriever.retrieve_all(tokens)
        ap_before = average_precision(results_before, qrels[q_id]) if q_id in qrels else None
        metrics_before = evaluate_single(results_before, qrels[q_id]) if q_id in qrels else None

        report = {
            "query_id": q_id,
            "original_query": q_text,
            "original_tokens": tokens,
            "ap_before": ap_before,
            "metrics_before": metrics_before,
            "ranking_before": serialize_results(results_before, docs),
            "expanded_query": None,
            "expansion_weights": {},
            "expansion_token_weights": {},
            "ap_after": None,
            "metrics_after": None,
            "ranking_after": None,
        }

        for row in report["ranking_before"]:
            original_ranking_rows.append(
                {
                    "query_id": q_id,
                    "rank": row["rank"],
                    "doc_id": row["doc_id"],
                    "score": row["score"],
                    "title": row["title"],
                }
            )

        if expand and expander:
            try:
                expanded_query, weights = expander.expand(
                    q_text,
                    n_terms=n_terms,
                    select_all=all_terms,
                )
                exp_tokens, token_weights = build_weighted_expansion_tokens(
                    q_text,
                    weights,
                    state["use_stemming"],
                    state["use_stopword"],
                )
                results_after = retriever.retrieve_all(
                    exp_tokens,
                    query_term_weights=token_weights,
                )
                ap_after = average_precision(results_after, qrels[q_id]) if q_id in qrels else None
                metrics_after = evaluate_single(results_after, qrels[q_id]) if q_id in qrels else None

                report.update(
                    {
                        "expanded_query": expanded_query,
                        "expanded_tokens": exp_tokens,
                        "expansion_weights": weights,
                        "expansion_token_weights": token_weights,
                        "ap_after": ap_after,
                        "metrics_after": metrics_after,
                        "ranking_after": serialize_results(results_after, docs),
                    }
                )

                for row in report["ranking_after"]:
                    expanded_ranking_rows.append(
                        {
                            "query_id": q_id,
                            "rank": row["rank"],
                            "doc_id": row["doc_id"],
                            "score": row["score"],
                            "title": row["title"],
                        }
                    )
                print(f"    Query {q_id} expanded.", end="\r")
            except Exception as exc:
                print(f"    Warning: Query {q_id} expansion failed ({exc}), using original only.")

        query_reports.append(report)

    map_before = _map_from_reports(query_reports, "ap_before")
    map_after = _map_from_reports(query_reports, "ap_after") if expand else None

    print(f"\n  MAP (before QE): {map_before:.4f}" if map_before is not None else "\n  MAP before: N/A")
    if map_after is not None:
        print(f"  MAP (after  QE): {map_after:.4f}")
        print_comparison(map_before or 0.0, map_after)

    config = {
        "use_stemming": state["use_stemming"],
        "use_stopword_elimination": state["use_stopword"],
        "tf_scheme": state["tf_scheme"],
        "weighting_mode": state["weighting_mode"],
        "query_file": state["query_file"] or str(DATA / "query.text"),
        "expand": expand,
        "n_terms": n_terms,
        "all_terms": all_terms,
        "top_k_display": top_k,
    }
    if expander:
        config["qe_provider"] = getattr(expander, "provider", "")
        config["qe_model"] = getattr(expander, "model_name", "")

    json_path = OUTPUT / "batch_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "config": config,
                "map_before": map_before,
                "map_after": map_after,
                "queries": query_reports,
            },
            f,
            indent=2,
        )

    query_rows = []
    for report in query_reports:
        weights = report.get("expansion_weights") or {}
        query_rows.append(
            {
                "query_id": report["query_id"],
                "original_query": report["original_query"],
                "expanded_query": report.get("expanded_query") or "",
                "expansion_terms": "; ".join(weights.keys()),
                "expansion_weights": "; ".join(f"{term}:{weight:.4f}" for term, weight in weights.items()),
                "ap_before": "" if report.get("ap_before") is None else f"{report['ap_before']:.6f}",
                "ap_after": "" if report.get("ap_after") is None else f"{report['ap_after']:.6f}",
            }
        )

    write_csv(
        OUTPUT / "batch_queries.csv",
        query_rows,
        [
            "query_id",
            "original_query",
            "expanded_query",
            "expansion_terms",
            "expansion_weights",
            "ap_before",
            "ap_after",
        ],
    )
    write_csv(
        OUTPUT / "batch_rankings_original.csv",
        original_ranking_rows,
        ["query_id", "rank", "doc_id", "score", "title"],
    )
    write_csv(
        OUTPUT / "batch_rankings_expanded.csv",
        expanded_ranking_rows,
        ["query_id", "rank", "doc_id", "score", "title"],
    )
    write_csv(
        OUTPUT / "batch_summary.csv",
        [
            {
                **config,
                "map_before": "" if map_before is None else f"{map_before:.6f}",
                "map_after": "" if map_after is None else f"{map_after:.6f}",
            }
        ],
        [
            "use_stemming",
            "use_stopword_elimination",
            "tf_scheme",
            "weighting_mode",
            "query_file",
            "expand",
            "n_terms",
            "all_terms",
            "qe_provider",
            "qe_model",
            "top_k_display",
            "map_before",
            "map_after",
        ],
    )

    print(f"\n  Results saved to {OUTPUT}")
    print("   - batch_results.json")
    print("   - batch_queries.csv")
    print("   - batch_rankings_original.csv")
    print("   - batch_rankings_expanded.csv")
    print("   - batch_summary.csv")
    return map_before, map_after


def prompt_bool(label: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    answer = input(f"  {label} {suffix}: ").strip().lower()
    if not answer:
        return default
    return answer in ("y", "yes", "1", "true")


def prompt_choice(label: str, choices: tuple[str, ...], default: str) -> str:
    choices_text = "/".join(choices)
    answer = input(f"  {label} ({choices_text}) [default {default}]: ").strip().lower()
    return answer if answer in choices else default


def configure_interactive() -> dict:
    print("\n" + "=" * 55)
    print("  Initial Configuration")
    print("=" * 55)
    return {
        "use_stemming": prompt_bool("Use stemming?", True),
        "use_stopword": prompt_bool("Eliminate stopwords?", True),
        "tf_scheme": prompt_choice("TF scheme", TF_SCHEMES, "log"),
        "weighting_mode": prompt_choice("Weighting mode", WEIGHTING_MODES, "tfidf_cosine"),
        "query_file": None,
    }


def show_menu(state: dict):
    print("\n" + "=" * 55)
    print("  IR System - CISI Collection")
    print("=" * 55)
    print(
        f"  Config: stemming={state['use_stemming']} | "
        f"stopword_elim={state['use_stopword']} | "
        f"tf={state['tf_scheme']} | weighting={state['weighting_mode']}"
    )
    print("  [1] Single query retrieval")
    print("  [2] Single query + Query Expansion")
    print("  [3] Batch retrieval + MAP")
    print("  [4] Batch retrieval + QE + MAP comparison")
    print("  [5] View inverted file for a document")
    print("  [6] View inverted index for a term")
    print("  [7] View query from dataset")
    print("  [0] Exit")
    print("-" * 55)


def interactive_mode(state: dict):
    expander = None

    while True:
        show_menu(state)
        choice = input("  Choice: ").strip()

        if choice == "0":
            print("  Bye!")
            break

        if choice in ("1", "2"):
            query_text = input("  Enter query: ").strip()
            if not query_text:
                print("  No query entered.")
                continue

            q_id_str = input("  Query ID for AP calculation (press Enter to skip): ").strip()
            q_id = int(q_id_str) if q_id_str.isdigit() else None

            top_k_str = input("  Top-K results [default 10]: ").strip()
            top_k = int(top_k_str) if top_k_str.isdigit() else 10

            expand = choice == "2"
            n_terms = 5
            all_terms = False
            if expand:
                if expander is None:
                    provider_input = prompt_choice(
                        "QE provider",
                        ("openai", "huggingface"),
                        "huggingface",
                    )
                    if provider_input == "huggingface":
                        token_input = input("  HF_TOKEN (Enter = env/.env): ").strip()
                        model_input = input("  HF model [default Qwen/Qwen2.5-7B-Instruct]: ").strip()
                        expander = get_expander(
                            provider=provider_input,
                            model_name=model_input or None,
                            hf_token=token_input or None,
                        )
                    else:
                        token_input = input("  OPENAI_API_KEY (Enter = env/.env): ").strip()
                        model_input = input("  OpenAI model [default gpt-5-mini]: ").strip()
                        expander = get_expander(
                            provider=provider_input,
                            model_name=model_input or None,
                            openai_api_key=token_input or None,
                        )
                n_terms_str = input("  Number of expansion terms [default 5]: ").strip()
                n_terms = int(n_terms_str) if n_terms_str.isdigit() else 5
                all_terms = prompt_bool("Use all generated terms?", False)

            run_single_query(
                query_text,
                state,
                top_k=top_k,
                expand=expand,
                n_terms=n_terms,
                all_terms=all_terms,
                expander=expander if expand else None,
                query_id=q_id,
            )

        elif choice == "3":
            run_batch(state, expand=False)

        elif choice == "4":
            if expander is None:
                provider_input = prompt_choice(
                    "QE provider",
                    ("openai", "huggingface"),
                    "huggingface",
                )
                if provider_input == "huggingface":
                    token_input = input("  HF_TOKEN (Enter = env/.env): ").strip()
                    model_input = input("  HF model [default Qwen/Qwen2.5-7B-Instruct]: ").strip()
                    expander = get_expander(
                        provider=provider_input,
                        model_name=model_input or None,
                        hf_token=token_input or None,
                    )
                else:
                    token_input = input("  OPENAI_API_KEY (Enter = env/.env): ").strip()
                    model_input = input("  OpenAI model [default gpt-5-mini]: ").strip()
                    expander = get_expander(
                        provider=provider_input,
                        model_name=model_input or None,
                        openai_api_key=token_input or None,
                    )
            n_terms_str = input("  Number of expansion terms [default 5]: ").strip()
            n_terms = int(n_terms_str) if n_terms_str.isdigit() else 5
            all_terms = prompt_bool("Use all generated terms?", False)
            run_batch(state, expand=True, n_terms=n_terms, all_terms=all_terms, expander=expander)

        elif choice == "5":
            doc_id_str = input("  Document ID: ").strip()
            if doc_id_str.isdigit():
                state["idx"].show_doc_inverted_file(
                    int(doc_id_str),
                    tf_scheme=state["tf_scheme"],
                )
            else:
                print("  Invalid document ID.")

        elif choice == "6":
            term = input("  Enter term to inspect: ").strip().lower()
            state["idx"].show_inverted_index(term)

        elif choice == "7":
            q_id_str = input("  Query ID: ").strip()
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


def parse_args():
    parser = argparse.ArgumentParser(description="IR System with GPT Query Expansion")
    parser.add_argument("--query", type=str, help="Single query text")
    parser.add_argument("--query-id", type=int, default=None, help="Query ID for AP calculation")
    parser.add_argument("--query-file", type=str, default=None, help="Batch query file path")
    parser.add_argument("--batch", action="store_true", help="Run batch evaluation")
    parser.add_argument("--expand", action="store_true", help="Enable query expansion")
    parser.add_argument("--n-terms", type=int, default=5, help="Expansion term count")
    parser.add_argument("--all-terms", action="store_true", help="Use all generated expansion terms")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K results to display")
    parser.add_argument("--tf", choices=TF_SCHEMES, default="log")
    parser.add_argument("--weighting", choices=WEIGHTING_MODES, default="tfidf_cosine")
    parser.add_argument("--no-stemming", action="store_true")
    parser.add_argument("--no-stopword", action="store_true")
    parser.add_argument("--openai-api-key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument(
        "--provider",
        choices=("openai", "huggingface"),
        default="openai",
        help="Query expansion provider",
    )
    parser.add_argument("--model", type=str, default=None, help="QE model name")
    return parser.parse_args()


def main():
    args = parse_args()

    is_interactive = not args.query and not args.batch
    if is_interactive:
        config = configure_interactive()
    else:
        config = {
            "use_stemming": not args.no_stemming,
            "use_stopword": not args.no_stopword,
            "tf_scheme": args.tf,
            "weighting_mode": args.weighting,
            "query_file": args.query_file,
        }

    state = load_system(**config)

    expander = None
    if args.expand:
        expander = get_expander(
            provider=args.provider,
            model_name=args.model,
            openai_api_key=args.openai_api_key,
            hf_token=args.hf_token,
        )

    if args.query:
        run_single_query(
            args.query,
            state,
            top_k=args.top_k,
            expand=args.expand,
            n_terms=args.n_terms,
            all_terms=args.all_terms,
            expander=expander,
            query_id=args.query_id,
        )
    elif args.batch:
        run_batch(
            state,
            expand=args.expand,
            n_terms=args.n_terms,
            all_terms=args.all_terms,
            expander=expander,
            top_k=args.top_k,
        )
    else:
        interactive_mode(state)


if __name__ == "__main__":
    main()
