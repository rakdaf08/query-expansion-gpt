"""
gpt_expansion.py - Query Expansion using OpenAI GPT.

Setup:
  1. Create an API key from the OpenAI dashboard.
  2. Set environment variable: export OPENAI_API_KEY="sk-..."
     or place it in a local .env file.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

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
    """Expands queries using OpenAI GPT and returns weighted expansion terms."""

    def __init__(
        self,
        model_name: str = DEFAULT_OPENAI_MODEL,
        openai_api_key: str | None = None,
    ):
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")

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

        raw_text = self._call_openai(query, requested_terms)
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
        print(f"  Model           : {self.model_name}")
        print("\n  Expansion Terms & Weights:")
        if not weights:
            print("    (no expansion terms)")
        for term, weight in sorted(weights.items(), key=lambda item: -item[1]):
            print(f"    + {term:<25} weight: {weight:.2f}")
        print(f"{'-'*55}")


if __name__ == "__main__":
    token = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OPENAI_API_KEY", "")
    if not token:
        print("Error: OPENAI_API_KEY not found. Set it in environment or pass as command argument.")
        sys.exit(1)

    expander = QueryExpander(openai_api_key=token)
    original = "information retrieval"
    expanded, expansion_weights = expander.expand(original, n_terms=5)
    expander.display_expansion(original, expanded, expansion_weights)
