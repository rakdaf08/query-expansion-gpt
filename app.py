import time
from pathlib import Path
from flask import Flask, request, render_template

# Import your STBI modules
from src.parser.parser import parse_documents
from src.preprocessing.preprocessing import preprocess_docs, preprocess
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.retrieval import Retriever
from src.qe.gpt_expansion import QueryExpander

app = Flask(__name__)

# Define paths relative to app.py
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

# Load documents once globally
print("Loading raw documents...")
raw_docs = parse_documents(str(DATA_DIR / "cisi.all"))

# Global cache for the InvertedIndex to avoid rebuilding it on every request.
# Key: (use_stemming, use_stopwords) -> Value: InvertedIndex instance
index_cache = {}

def get_or_build_index(use_stemming: bool, use_stopwords: bool) -> InvertedIndex:
    cache_key = (use_stemming, use_stopwords)
    if cache_key not in index_cache:
        print(f"Building index for Stemming={use_stemming}, Stopwords={use_stopwords}...")
        proc_docs = preprocess_docs(raw_docs, use_stemming=use_stemming, use_stopword=use_stopwords)
        idx = InvertedIndex()
        idx.build(proc_docs)
        index_cache[cache_key] = idx
    return index_cache[cache_key]


@app.route("/", methods=["GET", "POST"])
def home():
    hasil_pencarian = None
    query_text = ""
    error_msg = None

    # Default configuration state for the UI
    config = {
        "use_stemming": True,
        "use_stopwords": True,
        "tf_scheme": "log",
        "weighting_mode": "tfidf_cosine",
        "use_gpt": False,
        "gpt_provider": "openai",
        "term_limit": 5,
        "use_all_terms": False
    }

    if request.method == "POST":
        start_time = time.time()
        
        # 1. Capture Form Inputs
        query_text = request.form.get("query_text", "").strip()
        config["use_stemming"] = request.form.get("use_stemming") == "on"
        config["use_stopwords"] = request.form.get("use_stopwords") == "on"
        config["tf_scheme"] = request.form.get("tf_scheme", "log")
        config["weighting_mode"] = request.form.get("weighting_mode", "tfidf_cosine")
        
        config["use_gpt"] = request.form.get("use_gpt") == "on"
        config["gpt_provider"] = request.form.get("gpt_provider", "openai")
        config["term_limit"] = int(request.form.get("term_limit", 5))
        config["use_all_terms"] = request.form.get("use_all_terms") == "on"

        if query_text:
            try:
                # 2. Handle AI Query Expansion
                expanded_query = query_text
                expansion_weights = {}
                
                if config["use_gpt"]:
                    expander = QueryExpander(provider=config["gpt_provider"])
                    expanded_query, expansion_weights = expander.expand(
                        query=query_text,
                        n_terms=config["term_limit"],
                        select_all=config["use_all_terms"]
                    )

                # 3. Text Preprocessing for the final query
                query_tokens = preprocess(
                    expanded_query, 
                    use_stemming=config["use_stemming"], 
                    use_stopword=config["use_stopwords"]
                )

                # 4. Retrieval & Ranking
                # Retrieve the cached index based on the chosen preprocessing parameters
                idx = get_or_build_index(config["use_stemming"], config["use_stopwords"])
                
                retriever = Retriever(
                    inverted_index=idx, 
                    tf_scheme=config["tf_scheme"], 
                    weighting_mode=config["weighting_mode"]
                )
                
                # Pre-build vectors (in a real production app, this would also be cached)
                retriever.build_doc_vectors()
                
                # Fetch top 10 results
                ranked_results = retriever.retrieve(query_tokens, top_k=10, query_term_weights=expansion_weights)

                # 5. Format Output Data
                formatted_results = []
                for doc_id, score in ranked_results:
                    doc_data = raw_docs.get(doc_id, {})
                    formatted_results.append({
                        "id": doc_id,
                        "score": score,
                        "title": doc_data.get("title", "No Title"),
                        "abstract": doc_data.get("abstract", "No Abstract available.")
                    })

                hasil_pencarian = {
                    "is_expanded": config["use_gpt"] and len(expansion_weights) > 0,
                    "expanded_query": expanded_query,
                    "expansion_weights": expansion_weights,
                    "results": formatted_results,
                    "time_taken": time.time() - start_time
                }

            except Exception as e:
                error_msg = f"Terjadi kesalahan pada sistem: {str(e)}"

    return render_template(
        "index.html", 
        hasil=hasil_pencarian, 
        query_text=query_text, 
        config=config,
        error=error_msg
    )

if __name__ == "__main__":
    # Ensure data models compile on launch
    print("Initialize System STBI...")
    get_or_build_index(True, True) 
    app.run(debug=True, port=5000)