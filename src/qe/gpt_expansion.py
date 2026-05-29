"""
gpt_expansion.py — Query Expansion using cloud HuggingFace Inference API

Setup HuggingFace API:
  1. Daftar di https://huggingface.co
  2. Ambil token di https://huggingface.co/settings/tokens
  3. Set environment variable: export HF_TOKEN="hf_xxxx..."
     atau taruh di file .env
"""

from __future__ import annotations
import os
import re
import json
import sys
import requests
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dotenv():
    """Locate and load .env file from the script's directory up to the workspace root."""
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current / ".env"
        if env_path.is_file():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip().strip("'\"")
            except Exception:
                pass
            break
        if current.parent == current:
            break
        current = current.parent


# Load env variables automatically
load_dotenv()


# Model recommendations untuk HF Inference API
HF_MODELS = {
    "llama-3.2-1b":   "meta-llama/Llama-3.2-1B-Instruct",
    "llama-3.1-8b":   "meta-llama/Llama-3.1-8B-Instruct",
    "qwen-2.5-7b":    "Qwen/Qwen2.5-7B-Instruct",
}

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


class QueryExpander:
    """
    Expands queries using the cloud HuggingFace Inference API.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        hf_token: str = None,
    ):
        self.model_name = model_name
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")

        if not self.hf_token:
            raise ValueError(
                "HF_TOKEN tidak ditemukan!\n"
                "  1. Daftar di https://huggingface.co\n"
                "  2. Buat token di https://huggingface.co/settings/tokens\n"
                "  3. Set: export HF_TOKEN='hf_xxxx...'\n"
                "     atau pass langsung: QueryExpander(hf_token='hf_xxxx...')"
            )
        
        print(f"✓ Query Expander: HuggingFace Inference API (Cloud)")
        print(f"  Model : {self.model_name}")

    def _expand_hf_api(self, query: str, n_terms: int) -> tuple[str, dict[str, float]]:
        """Call HuggingFace Inference API (cloud)."""
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }

        # Prompt format standard untuk chat completions
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query expansion assistant. "
                    "Your job is to expand the user's search query by adding related terms. "
                    "You must return ONLY a raw JSON object with no markdown formatting and no extra text. "
                    "The JSON object must have exactly this format: "
                    '{"expanded_terms": ["term1", "term2", ...], "weights": [0.9, 0.8, ...]}'
                )
            },
            {
                "role": "user",
                "content": f"Expand this query with exactly {n_terms} related terms:\nQuery: {query}"
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 256,
            "temperature": 0.3
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)

            if response.status_code == 503:
                # Model sedang loading (cold start), tunggu
                import time
                print("  Model loading on HF servers, waiting 20s...")
                time.sleep(20)
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 401:
                raise ValueError("HF_TOKEN tidak valid atau expired.")

            if response.status_code != 200:
                raise RuntimeError(
                    f"HF API error {response.status_code}: {response.text[:200]}"
                )

            result = response.json()
            raw_text = result["choices"][0]["message"]["content"]

        except requests.exceptions.Timeout:
            raise RuntimeError("HF API timeout. Coba lagi atau gunakan model lebih kecil.")
        except requests.exceptions.ConnectionError:
            raise RuntimeError("Tidak bisa konek ke HuggingFace API. Cek koneksi internet.")
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"Gagal memproses respons dari HuggingFace API: {e}")

        return self._parse_expansion_output(query, raw_text, n_terms)

    def _parse_expansion_output(
        self, query: str, raw: str, n_terms: int
    ) -> tuple[str, dict[str, float]]:
        """
        Parse model output → (expanded_query, weights).
        Tries JSON first, falls back to plain text parsing.
        """
        original_tokens = set(query.lower().split())
        terms = []
        weights_list = []

        # Try JSON parsing
        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            # Find first {...} block
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                data = json.loads(match.group())
                terms = [t.lower() for t in data.get("expanded_terms", [])][:n_terms]
                weights_list = data.get("weights", [])[:n_terms]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: extract plain words
        if not terms:
            words = re.findall(r"\b[a-zA-Z]{3,}\b", raw)
            terms = [
                w.lower() for w in words
                if w.lower() not in original_tokens
            ][:n_terms]

        # Pad or trim weights
        while len(weights_list) < len(terms):
            weights_list.append(round(0.9 - len(weights_list) * 0.05, 2))
        weights_list = weights_list[:len(terms)]

        weights = {t: w for t, w in zip(terms, weights_list)}
        expanded_query = query + " " + " ".join(terms)
        return expanded_query.strip(), weights

    def expand(
        self,
        query: str,
        n_terms: int = 5,
        select_all: bool = False,
    ) -> tuple[str, dict[str, float]]:
        """
        Expand a query.

        Parameters
        ----------
        query      : original query string
        n_terms    : number of expansion terms to add
        select_all : if True, generate many terms (up to 20)

        Returns
        -------
        (expanded_query_string, {term: weight})
        """
        if select_all:
            n_terms = 20

        return self._expand_hf_api(query, n_terms)

    def display_expansion(
        self,
        original_query: str,
        expanded_query: str,
        weights: dict[str, float],
    ):
        """Print expansion details."""
        print(f"\n{'─'*55}")
        print(f"  Query Expansion Report")
        print(f"{'─'*55}")
        print(f"  Original Query  : {original_query}")
        print(f"  Expanded Query  : {expanded_query}")
        print(f"  Model           : {self.model_name}")
        print(f"\n  Expansion Terms & Weights:")
        original_tokens = set(original_query.lower().split())
        for term, weight in sorted(weights.items(), key=lambda x: -x[1]):
            marker = "  +" if term not in original_tokens else "   "
            print(f"    {marker} {term:<25} weight: {weight:.2f}")
        print(f"{'─'*55}")


if __name__ == "__main__":
    # Usage: python gpt_expansion.py [hf_token]
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("HF_TOKEN", "")

    if not token:
        print("Error: HF_TOKEN not found. Set it in environment or pass as command argument.")
        sys.exit(1)

    expander = QueryExpander(hf_token=token)
    q = "information retrieval"
    expanded, weights = expander.expand(q, n_terms=5)
    expander.display_expansion(q, expanded, weights)
