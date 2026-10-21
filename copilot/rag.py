"""Policy retrieval with ChromaDB.

Policies in docs/policies are split into sections by heading and stored in an
in-memory Chroma collection. Embeddings come from a stemmed TF-IDF model fitted on
the policy corpus, so the project runs offline and in CI with no model download.
Lexical retrieval misses paraphrases ("how long" vs "within 60 days"), so in
production `PolicyIndex.embed` should be swapped for a dense embedding model.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import chromadb
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

POLICY_DIR = Path(__file__).resolve().parent.parent / "docs" / "policies"

_SUFFIXES = ("ations", "ation", "ing", "ed", "es", "s", "e", "ly")


def _stem(word: str) -> str:
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in ENGLISH_STOP_WORDS]


def _chunks(path: Path) -> list[tuple[str, str, str]]:
    """Split a markdown file into (source, text_to_embed, body) on ## headings."""
    doc_title = path.stem.replace("-", " ").title()
    parts = re.split(r"^## ", path.read_text(encoding="utf-8"), flags=re.MULTILINE)
    out = []
    for part in parts[1:]:
        heading, _, body = part.partition("\n")
        body = " ".join(body.split())
        if body:
            out.append((f"{doc_title} > {heading.strip()}", f"{doc_title}. {heading.strip()}. {body}", body))
    return out


class PolicyIndex:
    def __init__(self, policy_dir: Path = POLICY_DIR):
        self._client = chromadb.EphemeralClient()
        self._col = self._client.get_or_create_collection("policies", metadata={"hnsw:space": "cosine"})
        self._vec = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, sublinear_tf=True, ngram_range=(1, 2))
        ids, docs, metas = [], [], []
        for path in sorted(policy_dir.glob("*.md")):
            for i, (source, text, body) in enumerate(_chunks(path)):
                ids.append(f"{path.stem}-{i}")
                docs.append(text)
                metas.append({"source": source, "body": body})
        self._vec.fit(docs)
        if ids:
            self._col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=self.embed(docs))

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._vec.transform(texts).toarray().tolist()

    def search(self, query: str, k: int = 2) -> list[dict]:
        res = self._col.query(query_embeddings=self.embed([query]), n_results=k)
        return [
            {"source": m["source"], "text": m["body"], "score": round(1 - dist, 3)}
            for m, dist in zip(res["metadatas"][0], res["distances"][0])
        ]


@lru_cache(maxsize=1)
def get_index() -> PolicyIndex:
    return PolicyIndex()
