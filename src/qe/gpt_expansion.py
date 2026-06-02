"""
gpt_expansion.py - Query Expansion using OpenAI GPT or HuggingFace LLMs.

Setup:
  OpenAI:
    export OPENAI_API_KEY="sk-..."

  HuggingFace:
    export HF_TOKEN="hf_..."
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_dotenv():
    """Locate and load .env file from this module up to the workspace root."""
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
                            key, value = line.split("=", 1)
                            os.environ[key.strip()] = value.strip().strip("'\"")
            except Exception:
                pass
            break
        if current.parent == current:
            break
        current = current.parent


load_dotenv()


DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"
PROVIDERS = ("openai", "huggingface")
MAX_ALL_TERMS = 20

EXPANSION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "expanded_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "term": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["term", "weight"],
            },
        }
    },
    "required": ["expanded_terms"],
}


class QueryExpander:
    """Expands queries using OpenAI GPT or HuggingFace LLMs."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: str | None = None,
        openai_api_key: str | None = None,
        hf_token: str | None = None,
    ):
        provider = provider.lower().strip()
        if provider not in PROVIDERS:
            raise ValueError(f"Provider tidak valid: {provider}. Pilih: {', '.join(PROVIDERS)}")

        self.provider = provider
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.hf_token = hf_token or os.environ.get("HF_TOKEN", "")
        self.client = None

        if self.provider == "openai":
            self.model_name = self.model_name or DEFAULT_OPENAI_MODEL
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY tidak ditemukan!\n"
                    "  1. Buat API key di dashboard OpenAI\n"
                    "  2. Set: export OPENAI_API_KEY='sk-...'\n"
                    "     atau pass langsung: QueryExpander(openai_api_key='sk-...')"
                )

            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ImportError(
                    "Package 'openai' belum terinstall. Jalankan: pip install -r requirements.txt"
                ) from exc

            self.client = OpenAI(api_key=self.openai_api_key)
            print("✓ Query Expander: OpenAI GPT")
        else:
            self.model_name = self.model_name or DEFAULT_HF_MODEL
            if not self.hf_token:
                raise ValueError(
                    "HF_TOKEN tidak ditemukan!\n"
                    "  1. Buat token di https://huggingface.co/settings/tokens\n"
                    "  2. Set: export HF_TOKEN='hf_...'\n"
                    "     atau pass langsung: QueryExpander(provider='huggingface', hf_token='hf_...')"
                )
            print("✓ Query Expander: HuggingFace Inference API")

        print(f"  Provider: {self.provider}")
        print(f"  Model : {self.model_name}")

    def _call_openai(self, query: str, n_terms: int) -> str:
        """Call OpenAI Responses API and return model text output."""
        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You are a search query expansion assistant for an "
                        "Information Retrieval experiment. Return concise related "
                        "single-word or short-phrase English terms. Each term weight "
                        "must be between 0.0 and 1.0, where higher means more relevant. "
                        "Avoid duplicating the user's original terms."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Expand this query with up to {n_terms} related terms.\n"
                        f"Query: {query}"
                    ),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "query_expansion",
                    "schema": EXPANSION_SCHEMA,
                    "strict": True,
                }
            },
            max_output_tokens=600,
        )

        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        # Fallback for SDK response shapes that expose output content directly.
        try:
            chunks = []
            for item in response.output:
                for content in getattr(item, "content", []) or []:
                    text = getattr(content, "text", None)
                    if text:
                        chunks.append(text)
            if chunks:
                return "\n".join(chunks)
        except Exception:
            pass

        raise RuntimeError("Respons OpenAI tidak berisi output teks yang bisa diproses.")

    def _call_huggingface(self, query: str, n_terms: int) -> str:
        """Call HuggingFace router chat completions API and return model text output."""
        api_url = "https://router.huggingface.co/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json",
        }
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a search query expansion assistant for an Information "
                    "Retrieval experiment. Return ONLY a raw JSON object with this "
                    'schema: {"expanded_terms": [{"term": "term", "weight": 0.9}]}. '
                    "Use concise English single-word or short-phrase terms. Weights "
                    "must be between 0.0 and 1.0. Do not include markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Expand this query with up to {n_terms} related terms.\n"
                    f"Query: {query}"
                ),
            },
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 600,
            "temperature": 0.3,
        }

        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 503:
                import time

                print("  Model loading on HuggingFace servers, waiting 20s...")
                time.sleep(20)
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)

            if response.status_code == 401:
                raise ValueError("HF_TOKEN tidak valid atau expired.")
            if response.status_code != 200:
                raise RuntimeError(f"HF API error {response.status_code}: {response.text[:300]}")

            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout as exc:
            raise RuntimeError("HF API timeout. Coba lagi atau gunakan model lebih kecil.") from exc
        except requests.exceptions.ConnectionError as exc:
            raise RuntimeError("Tidak bisa konek ke HuggingFace API. Cek koneksi internet.") from exc
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Gagal memproses respons HuggingFace API: {exc}") from exc

    def _normalize_terms(
        self,
        query: str,
        terms_with_weights: list[dict],
        n_terms: int,
    ) -> dict[str, float]:
        """Clean, clamp, deduplicate, and limit expansion terms."""
        original_tokens = set(re.findall(r"\b[a-z0-9]+\b", query.lower()))
        weights: dict[str, float] = {}

        for item in terms_with_weights:
            if not isinstance(item, dict):
                continue
            raw_term = str(item.get("term", "")).strip().lower()
            raw_term = re.sub(r"[^a-z0-9\s-]", " ", raw_term)
            raw_term = re.sub(r"\s+", " ", raw_term).strip()
            if not raw_term:
                continue

            term_tokens = re.findall(r"\b[a-z0-9]+\b", raw_term)
            if not term_tokens or all(token in original_tokens for token in term_tokens):
                continue

            try:
                weight = float(item.get("weight", 0.0))
            except (TypeError, ValueError):
                weight = 0.5
            weight = max(0.0, min(1.0, weight))

            if raw_term not in weights or weight > weights[raw_term]:
                weights[raw_term] = weight
            if len(weights) >= n_terms:
                break

        return dict(sorted(weights.items(), key=lambda item: -item[1]))

    def _parse_expansion_output(
        self,
        query: str,
        raw: str,
        n_terms: int,
    ) -> tuple[str, dict[str, float]]:
        """Parse model output into (expanded query, term weights)."""
        terms_with_weights = []

        try:
            cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            data = json.loads(match.group() if match else cleaned)
            terms_with_weights = data.get("expanded_terms", [])
            if terms_with_weights and not isinstance(terms_with_weights[0], dict):
                raw_terms = terms_with_weights
                raw_weights = data.get("weights", [])
                terms_with_weights = [
                    {
                        "term": term,
                        "weight": (
                            raw_weights[i]
                            if i < len(raw_weights)
                            else round(0.9 - i * 0.05, 2)
                        ),
                    }
                    for i, term in enumerate(raw_terms)
                ]
        except (json.JSONDecodeError, AttributeError, TypeError):
            words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{2,}\b", raw)
            terms_with_weights = [
                {"term": word.lower(), "weight": round(0.9 - i * 0.05, 2)}
                for i, word in enumerate(words)
            ]

        weights = self._normalize_terms(query, terms_with_weights, n_terms)
        expanded_query = " ".join([query, *weights.keys()]).strip()
        return expanded_query, weights

    def expand(
        self,
        query: str,
        n_terms: int = 5,
        select_all: bool = False,
    ) -> tuple[str, dict[str, float]]:
        """Expand a query and return (expanded_query_string, {term: weight})."""
        requested_terms = MAX_ALL_TERMS if select_all else max(0, n_terms)
        if requested_terms == 0:
            return query, {}

        if self.provider == "openai":
            raw_text = self._call_openai(query, requested_terms)
        else:
            raw_text = self._call_huggingface(query, requested_terms)
        return self._parse_expansion_output(query, raw_text, requested_terms)

    def display_expansion(
        self,
        original_query: str,
        expanded_query: str,
        weights: dict[str, float],
    ):
        """Print expansion details."""
        print(f"\n{'-'*55}")
        print("  Query Expansion Report")
        print(f"{'-'*55}")
        print(f"  Original Query  : {original_query}")
        print(f"  Expanded Query  : {expanded_query}")
        print(f"  Provider        : {self.provider}")
        print(f"  Model           : {self.model_name}")
        print("\n  Expansion Terms & Weights:")
        if not weights:
            print("    (no expansion terms)")
        for term, weight in sorted(weights.items(), key=lambda item: -item[1]):
            print(f"    + {term:<25} weight: {weight:.2f}")
        print(f"{'-'*55}")


if __name__ == "__main__":
    provider = sys.argv[1] if len(sys.argv) > 1 else "openai"
    token = sys.argv[2] if len(sys.argv) > 2 else ""
    kwargs = {"provider": provider}
    if provider == "huggingface":
        kwargs["hf_token"] = token or os.environ.get("HF_TOKEN", "")
    else:
        kwargs["openai_api_key"] = token or os.environ.get("OPENAI_API_KEY", "")
    if not token:
        env_name = "HF_TOKEN" if provider == "huggingface" else "OPENAI_API_KEY"
        if not os.environ.get(env_name, ""):
            print(f"Error: {env_name} not found. Set it in environment or pass as command argument.")
            sys.exit(1)

    expander = QueryExpander(**kwargs)
    original = "information retrieval"
    expanded, expansion_weights = expander.expand(original, n_terms=5)
    expander.display_expansion(original, expanded, expansion_weights)
