from pathlib import Path

import pytest

from src.evaluation.evaluation import average_precision, mean_average_precision
from src.indexing.inverted_index import InvertedIndex
from src.parser.parser import parse_documents, parse_queries, parse_query_file, parse_qrels
from src.preprocessing.preprocessing import preprocess
from src.qe.gpt_expansion import QueryExpander
from src.retrieval.retrieval import Retriever


def test_parser_reads_cisi_files(tmp_path: Path):
    docs_path = tmp_path / "docs.all"
    docs_path.write_text(
        ".I 1\n.T\nFirst Title\n.A\nAn Author\n.W\nDocument body.\n.X\nignored\n",
        encoding="utf-8",
    )
    queries_path = tmp_path / "query.text"
    queries_path.write_text(".I 1\n.W\nfirst query\n.I 2\n.W\nsecond query\n", encoding="utf-8")
    qrels_path = tmp_path / "qrels.text"
    qrels_path.write_text("1 7 0 0\n1 9 0 0\n", encoding="utf-8")

    assert parse_documents(str(docs_path))[1]["title"] == "First Title"
    assert parse_queries(str(queries_path))[2] == "second query"
    assert parse_qrels(str(qrels_path))[1] == {7, 9}


def test_parse_query_file_line_fallback(tmp_path: Path):
    query_path = tmp_path / "queries.txt"
    query_path.write_text("alpha search\n\nbeta retrieval\n", encoding="utf-8")

    assert parse_query_file(str(query_path)) == {1: "alpha search", 2: "beta retrieval"}


@pytest.mark.parametrize(
    "stemming,stopword,expected",
    [
        (True, True, ["run", "cat"]),
        (False, True, ["running", "cats"]),
        (True, False, ["run", "the", "cat"]),
        (False, False, ["running", "the", "cats"]),
    ],
)
def test_preprocessing_options(stemming, stopword, expected):
    assert preprocess("Running the cats!", use_stemming=stemming, use_stopword=stopword) == expected


def build_small_index():
    idx = InvertedIndex()
    idx.build(
        {
            1: ["apple", "apple", "banana"],
            2: ["banana", "carrot"],
        }
    )
    return idx


def test_tf_variants_and_idf():
    idx = build_small_index()

    assert idx.tf_raw("apple", 1) == 2.0
    assert idx.tf_binary("apple", 1) == 1.0
    assert idx.tf_log("apple", 1) > 1.0
    assert idx.tf_augmented("apple", 1) == 1.0
    assert idx.idf("apple") > 0.0
    assert idx.tfidf("apple", 1, "raw") == idx.tf_raw("apple", 1) * idx.idf("apple")


@pytest.mark.parametrize("weighting_mode", ["tf", "idf", "tfidf", "tfidf_cosine"])
def test_retriever_weighting_modes(weighting_mode):
    idx = build_small_index()
    retriever = Retriever(idx, tf_scheme="raw", weighting_mode=weighting_mode)
    retriever.build_doc_vectors()

    results = retriever.retrieve_all(["apple"])
    assert results
    assert results[0][0] == 1
    assert results[0][1] > 0.0


def test_query_expansion_weights_affect_query_vector():
    idx = build_small_index()
    vec_plain = idx.get_query_vector(["banana", "carrot"], "raw", "tfidf")
    vec_weighted = idx.get_query_vector(
        ["banana", "carrot"],
        "raw",
        "tfidf",
        term_weights={"carrot": 0.25},
    )

    assert vec_weighted["carrot"] == pytest.approx(vec_plain["carrot"] * 0.25)


def test_average_precision_and_map():
    results = [(1, 0.9), (2, 0.8), (3, 0.7)]

    assert average_precision(results, {1, 3}) == pytest.approx((1 / 1 + 2 / 3) / 2)
    assert mean_average_precision({1: results}, {1: {1, 3}}) == pytest.approx(
        average_precision(results, {1, 3})
    )


def test_gpt_expansion_parser_without_api_call():
    expander = QueryExpander.__new__(QueryExpander)
    raw = """
    {
      "expanded_terms": [
        {"term": "search", "weight": 0.91},
        {"term": "ranking", "weight": 0.82},
        {"term": "information", "weight": 0.4}
      ]
    }
    """

    expanded, weights = expander._parse_expansion_output("information retrieval", raw, 2)

    assert expanded == "information retrieval search ranking"
    assert weights == {"search": 0.91, "ranking": 0.82}
