"""
evaluation.py — IR Evaluation Metrics
Computes: Precision, Recall, Average Precision, MAP
"""

from __future__ import annotations


def average_precision(
    ranked_results: list[tuple[int, float]],
    relevant_docs: set[int],
) -> float:
    """
    Compute Average Precision (AP) for a single query.

    Parameters
    ----------
    ranked_results : list of (doc_id, score) in ranked order
    relevant_docs  : set of relevant doc IDs for this query

    Returns
    -------
    AP score (float 0–1)
    """
    if not relevant_docs:
        return 0.0

    n_relevant_found = 0
    precision_sum = 0.0

    for rank, (doc_id, _score) in enumerate(ranked_results, 1):
        if doc_id in relevant_docs:
            n_relevant_found += 1
            precision_at_rank = n_relevant_found / rank
            precision_sum += precision_at_rank

    if n_relevant_found == 0:
        return 0.0

    return precision_sum / len(relevant_docs)


def mean_average_precision(
    all_results: dict[int, list[tuple[int, float]]],
    qrels: dict[int, set[int]],
) -> float:
    """
    Compute MAP across all queries.

    Parameters
    ----------
    all_results : {query_id: [(doc_id, score), ...]}
    qrels       : {query_id: set of relevant doc_ids}

    Returns
    -------
    MAP score (float 0–1)
    """
    aps = []
    for q_id, results in all_results.items():
        relevant = qrels.get(q_id, set())
        ap = average_precision(results, relevant)
        aps.append(ap)

    return sum(aps) / len(aps) if aps else 0.0


def precision_at_k(
    ranked_results: list[tuple[int, float]],
    relevant_docs: set[int],
    k: int = 10,
) -> float:
    """Precision@K — fraction of top-K results that are relevant."""
    if not ranked_results:
        return 0.0
    top_k = [doc_id for doc_id, _ in ranked_results[:k]]
    relevant_in_top_k = sum(1 for d in top_k if d in relevant_docs)
    return relevant_in_top_k / k


def recall_at_k(
    ranked_results: list[tuple[int, float]],
    relevant_docs: set[int],
    k: int = 10,
) -> float:
    """Recall@K — fraction of relevant docs found in top-K."""
    if not relevant_docs:
        return 0.0
    top_k = [doc_id for doc_id, _ in ranked_results[:k]]
    relevant_in_top_k = sum(1 for d in top_k if d in relevant_docs)
    return relevant_in_top_k / len(relevant_docs)


def f1_at_k(
    ranked_results: list[tuple[int, float]],
    relevant_docs: set[int],
    k: int = 10,
) -> float:
    """F1@K — harmonic mean of Precision@K and Recall@K."""
    p = precision_at_k(ranked_results, relevant_docs, k)
    r = recall_at_k(ranked_results, relevant_docs, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_single(
    ranked_results: list[tuple[int, float]],
    relevant_docs: set[int],
    k: int = 10,
    label: str = "",
) -> dict:
    """Return all metrics for a single query."""
    metrics = {
        "ap": average_precision(ranked_results, relevant_docs),
        f"p@{k}": precision_at_k(ranked_results, relevant_docs, k),
        f"r@{k}": recall_at_k(ranked_results, relevant_docs, k),
        f"f1@{k}": f1_at_k(ranked_results, relevant_docs, k),
        "n_relevant": len(relevant_docs),
        "n_retrieved": len(ranked_results),
    }
    return metrics


def evaluate_batch(
    retriever,
    processed_queries: dict[int, list[str]],
    qrels: dict[int, set[int]],
    label: str = "Original",
    k: int = 10,
) -> tuple[float, dict[int, dict]]:
    """
    Run retrieval on all queries and compute MAP.

    Returns
    -------
    (MAP, {query_id: metrics_dict})
    """
    all_results = {}
    per_query_metrics = {}

    for q_id, tokens in processed_queries.items():
        if q_id not in qrels:
            continue
        results = retriever.retrieve_all(tokens)
        all_results[q_id] = results
        per_query_metrics[q_id] = evaluate_single(results, qrels[q_id], k=k)

    map_score = mean_average_precision(all_results, qrels)
    return map_score, per_query_metrics


def print_comparison(
    map_before: float,
    map_after: float,
    metrics_before: dict[int, dict] = None,
    metrics_after: dict[int, dict] = None,
    k: int = 10,
):
    """Print MAP comparison table."""
    print(f"\n{'═'*55}")
    print(f"  Evaluation Results (K={k})")
    print(f"{'═'*55}")
    print(f"  {'Metric':<30} {'Before QE':>10} {'After QE':>10}")
    print(f"  {'─'*48}")
    print(f"  {'MAP':<30} {map_before:>10.4f} {map_after:>10.4f}")

    delta = map_after - map_before
    change = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
    print(f"  {'MAP Δ (change)':<30} {'':<10} {change} {abs(delta):.4f}")
    print(f"{'═'*55}")

    if map_after > map_before:
        pct = (delta / map_before * 100) if map_before > 0 else float("inf")
        print(f"  ✓ Query Expansion IMPROVED MAP by {pct:.1f}%")
    elif map_after < map_before:
        pct = (abs(delta) / map_before * 100) if map_before > 0 else float("inf")
        print(f"  ✗ Query Expansion DECREASED MAP by {pct:.1f}%")
    else:
        print(f"  → No change in MAP")


if __name__ == "__main__":
    # Quick smoke test
    results = [(1, 0.9), (2, 0.8), (3, 0.7), (4, 0.6), (5, 0.5)]
    relevant = {1, 3, 5}
    ap = average_precision(results, relevant)
    print(f"AP (expected ~0.756): {ap:.4f}")

    p5 = precision_at_k(results, relevant, k=5)
    r5 = recall_at_k(results, relevant, k=5)
    print(f"P@5: {p5:.4f}, R@5: {r5:.4f}")
