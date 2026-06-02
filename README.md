# Query Expansion GPT - STBI

Program Information Retrieval untuk koleksi CISI dengan Query Expansion menggunakan OpenAI GPT. Program mendukung mode interaktif dan batch, pilihan preprocessing, pilihan pembobotan, ranking dokumen, similarity score, AP/MAP, serta output eksperimen dalam JSON dan CSV.

## Instalasi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Untuk fitur Query Expansion, buat file `.env`:

```bash
cp .env.example .env
```

Isi:

```text
OPENAI_API_KEY=sk-your-openai-api-key-here
```

## Cara Menjalankan

Mode interaktif:

```bash
python main.py
```

Saat start, user memilih:

- stemming: on/off
- stopword elimination: on/off
- TF scheme: `raw`, `binary`, `log`, `augmented`
- weighting mode: `tf`, `idf`, `tfidf`, `tfidf_cosine`

Single query:

```bash
python main.py --query "information retrieval" --weighting tfidf_cosine --top-k 5
```

Single query dengan GPT expansion:

```bash
python main.py --query "information retrieval" --expand --n-terms 5 --query-id 1
```

Batch retrieval default dari `data/query.text`:

```bash
python main.py --batch --weighting tfidf_cosine --top-k 10
```

Batch retrieval dengan GPT expansion:

```bash
python main.py --batch --expand --n-terms 5 --weighting tfidf_cosine
```

Batch memakai file query eksternal:

```bash
python main.py --batch --query-file queries.txt
```

Format `queries.txt` boleh CISI:

```text
.I 1
.W
information retrieval
.I 2
.W
library automation
```

Atau satu query per baris:

```text
information retrieval
library automation
```

## Output Batch

Semua output batch disimpan di folder `output/`:

- `batch_results.json`: konfigurasi eksperimen, MAP before/after, query, expanded query, weights, AP, metrics, ranking lengkap.
- `batch_queries.csv`: query asal, query expanded, expansion terms, expansion weights, AP before/after.
- `batch_rankings_original.csv`: ranking dokumen query asal beserta score.
- `batch_rankings_expanded.csv`: ranking dokumen query expanded beserta score.
- `batch_summary.csv`: ringkasan konfigurasi dan MAP.

## Fitur Interaktif

Menu utama:

1. Single query retrieval.
2. Single query + Query Expansion.
3. Batch retrieval + MAP.
4. Batch retrieval + QE + MAP comparison.
5. View inverted file for a document.
6. View inverted index for a term.
7. View query from dataset.

Fitur inverted file per dokumen menampilkan term pada dokumen tertentu beserta raw TF, TF sesuai skema terpilih, DF, IDF, dan TF-IDF.

## Pembobotan

TF scheme:

- `raw`: frekuensi asli term.
- `binary`: 1 jika term muncul, 0 jika tidak.
- `log`: `1 + log(tf)`.
- `augmented`: `0.5 + 0.5 * tf / max_tf`.

Weighting mode:

- `tf`: skor memakai bobot TF saja dan dot product.
- `idf`: skor memakai bobot IDF saja dan dot product.
- `tfidf`: skor memakai TF x IDF dan dot product.
- `tfidf_cosine`: skor memakai TF x IDF dengan cosine similarity.

Untuk query expanded, term asli diberi bobot `1.0`, sedangkan term tambahan memakai weight dari GPT.

## Struktur Program

- `main.py`: CLI, interactive menu, single query flow, batch flow, output JSON/CSV.
- `src/parser/parser.py`: parser dokumen CISI, query, qrels, dan query file eksternal.
- `src/preprocessing/preprocessing.py`: tokenisasi, stopword elimination, stemming.
- `src/indexing/inverted_index.py`: inverted index, TF/IDF/TF-IDF, inverted file per dokumen.
- `src/retrieval/retrieval.py`: retrieval, dot product, cosine similarity.
- `src/evaluation/evaluation.py`: AP, MAP, precision, recall, F1.
- `src/qe/gpt_expansion.py`: Query Expansion dengan OpenAI GPT.
- `tests/test_core.py`: unit test utama.

## Library

- `nltk`: stopword dan Porter stemming.
- `openai`: akses OpenAI GPT Responses API.
- `requests`, `numpy`, `pandas`, `scikit-learn`: dependensi pendukung/eksperimen.
- `pytest`: test runner.

## Testing

```bash
pytest
```

Smoke test tanpa GPT:

```bash
python main.py --query "information retrieval" --weighting tfidf_cosine --top-k 5
python main.py --batch --weighting tfidf_cosine --top-k 10
```

Smoke test dengan GPT membutuhkan `OPENAI_API_KEY`:

```bash
python main.py --query "information retrieval" --expand --n-terms 5
python main.py --batch --expand --n-terms 5
```
