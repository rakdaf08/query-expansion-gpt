import time
import os
from pathlib import Path
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

# Import modul STBI
from src.parser.parser import parse_documents, parse_query_file, parse_qrels
from src.preprocessing.preprocessing import preprocess_docs, preprocess, preprocess_queries
from src.indexing.inverted_index import InvertedIndex
from src.retrieval.retrieval import Retriever
from src.qe.gpt_expansion import QueryExpander
from src.evaluation.evaluation import evaluate_batch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'output'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"

print("Loading raw documents & qrels...")
raw_docs = parse_documents(str(DATA_DIR / "cisi.all"))
qrels = parse_qrels(str(DATA_DIR / "qrels.text"))

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
    batch_hasil = None
    query_text = ""
    error_msg = None
    active_tab = "interactive"

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
        active_tab = request.form.get("mode", "interactive")
        
        config["use_stemming"] = request.form.get("use_stemming") == "on"
        config["use_stopwords"] = request.form.get("use_stopwords") == "on"
        config["tf_scheme"] = request.form.get("tf_scheme", "log")
        config["weighting_mode"] = request.form.get("weighting_mode", "tfidf_cosine")
        config["use_gpt"] = request.form.get("use_gpt") == "on"
        config["gpt_provider"] = request.form.get("gpt_provider", "openai")
        config["term_limit"] = int(request.form.get("term_limit") or 5)
        config["use_all_terms"] = request.form.get("use_all_terms") == "on"

        idx = get_or_build_index(config["use_stemming"], config["use_stopwords"])
        retriever = Retriever(inverted_index=idx, tf_scheme=config["tf_scheme"], weighting_mode=config["weighting_mode"])
        retriever.build_doc_vectors()

        try:
            if active_tab == "interactive":
                query_text = request.form.get("query_text", "").strip()
                if query_text:
                    expanded_query = query_text
                    expansion_weights = {}
                    
                    if config["use_gpt"]:
                        expander = QueryExpander(provider=config["gpt_provider"])
                        expanded_query, expansion_weights = expander.expand(
                            query=query_text, n_terms=config["term_limit"], select_all=config["use_all_terms"]
                        )

                    query_tokens = preprocess(expanded_query, use_stemming=config["use_stemming"], use_stopword=config["use_stopwords"])
                    ranked_results = retriever.retrieve(query_tokens, top_k=10, query_term_weights=expansion_weights)

                    formatted_results = []
                    for doc_id, score in ranked_results:
                        doc_data = raw_docs.get(doc_id, {})
                        formatted_results.append({
                            "id": doc_id, "score": score, 
                            "title": doc_data.get("title", "No Title"), 
                            "abstract": doc_data.get("abstract", "No Abstract")
                        })

                    hasil_pencarian = {
                        "is_expanded": config["use_gpt"] and len(expansion_weights) > 0,
                        "expanded_query": expanded_query,
                        "expansion_weights": expansion_weights,
                        "results": formatted_results,
                        "time_taken": time.time() - start_time
                    }

            elif active_tab == "batch":
                if 'batch_file' not in request.files:
                    raise ValueError("File batch tidak ditemukan.")
                
                file = request.files['batch_file']
                if file.filename == '':
                    raise ValueError("Tidak ada file yang dipilih.")

                filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
                file.save(filepath)

                # Proses File Batch
                raw_queries = parse_query_file(filepath)
                
                # 1. Evaluasi Baseline (Tanpa Ekspansi)
                proc_queries_base = preprocess_queries(raw_queries, config["use_stemming"], config["use_stopwords"])
                map_before, metrics_before = evaluate_batch(retriever, proc_queries_base, qrels, k=10)

                map_after = map_before
                metrics_after = metrics_before
                
                # 2. Evaluasi dengan Ekspansi (Jika diaktifkan)
                if config["use_gpt"]:
                    expander = QueryExpander(provider=config["gpt_provider"])
                    proc_queries_expanded = {}
                    
                    for q_id, q_text in raw_queries.items():
                        # Hati-hati: Loop API call ini bisa memakan waktu!
                        expanded_q, weights = expander.expand(q_text, config["term_limit"], config["use_all_terms"])
                        proc_queries_expanded[q_id] = preprocess(expanded_q, config["use_stemming"], config["use_stopwords"])
                    
                    map_after, metrics_after = evaluate_batch(retriever, proc_queries_expanded, qrels, k=10)

                batch_hasil = {
                    "total_queries": len(raw_queries),
                    "map_before": map_before,
                    "map_after": map_after,
                    "delta": map_after - map_before,
                    "time_taken": time.time() - start_time
                }
                
                # Sesuai spesifikasi, hasil disimpan ke file output
                output_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"output_MAP_{int(time.time())}.txt")
                with open(output_filepath, "w") as f:
                    f.write(f"Evaluasi Batch Selesai.\nMAP Asli: {map_before:.4f}\nMAP Ekspansi: {map_after:.4f}\n")

        except Exception as e:
            error_msg = f"Terjadi kesalahan pada sistem: {str(e)}"

    return render_template(
        "index.html", 
        hasil=hasil_pencarian, 
        batch_hasil=batch_hasil,
        query_text=query_text, 
        config=config,
        error=error_msg,
        active_tab=active_tab
    )

if __name__ == "__main__":
    get_or_build_index(True, True) 
    app.run(debug=True, port=5000)