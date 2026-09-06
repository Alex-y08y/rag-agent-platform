"""In-memory keyword (BM25-like) store for LITE_MODE (no Elasticsearch required).

Implements a simplified BM25 scoring over in-memory documents.
"""
from __future__ import annotations

import math
import re
import threading
from collections import Counter
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer: split on non-alphanumeric, keep Chinese chars individually."""
    # Extract words (sequences of letters/digits) and individual Chinese characters
    tokens = []
    for token in re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower()):
        tokens.append(token)
    return tokens


class LiteKeywordStore:
    """In-memory BM25-like keyword store.

    Drop-in replacement for ElasticsearchStore when LITE_MODE=true or ES unavailable.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._lock = threading.Lock()
        self._indices: dict[str, dict[str, dict[str, Any]]] = {}
        self.k1 = k1
        self.b = b
        logger.info("LiteKeywordStore initialized (in-memory BM25-like)")

    def index(
        self,
        kb_id: str,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict[str, Any]] | None = None,
    ) -> None:
        metadatas = metadatas or [{} for _ in chunk_ids]
        with self._lock:
            idx = self._indices.setdefault(kb_id, {})
            for cid, doc, meta in zip(chunk_ids, documents, metadatas):
                tokens = _tokenize(doc)
                idx[cid] = {
                    "content": doc,
                    "tokens": tokens,
                    "token_count": len(tokens),
                    "term_freq": Counter(tokens),
                    "metadata": meta,
                }
        logger.info("Lite keyword index: %d chunks into %s", len(chunk_ids), kb_id)

    def search(
        self,
        kb_id: str,
        query: str,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        with self._lock:
            idx = self._indices.get(kb_id, {})
            if not idx:
                return []

            query_tokens = _tokenize(query)
            if not query_tokens:
                return []

            # Compute average document length
            total_docs = len(idx)
            avgdl = sum(d["token_count"] for d in idx.values()) / max(total_docs, 1)

            # Document frequency for each query term
            doc_freq: dict[str, int] = {}
            for token in set(query_tokens):
                doc_freq[token] = sum(
                    1 for d in idx.values() if token in d["term_freq"]
                )

            results = []
            for cid, doc_data in idx.items():
                # Metadata filter
                if filters:
                    skip = False
                    for k, v in filters.items():
                        if doc_data["metadata"].get(k) != v:
                            skip = True
                            break
                    if skip:
                        continue

                score = 0.0
                dl = doc_data["token_count"]
                tf = doc_data["term_freq"]

                for token in query_tokens:
                    if token not in tf:
                        continue
                    # IDF
                    df = doc_freq.get(token, 0)
                    idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1)
                    # BM25 term score
                    term_freq = tf[token]
                    denom = term_freq + self.k1 * (1 - self.b + self.b * dl / max(avgdl, 1))
                    score += idf * (term_freq * (self.k1 + 1)) / max(denom, 1e-9)

                if score > 0:
                    results.append({
                        "chunk_id": cid,
                        "content": doc_data["content"],
                        "score": score,
                        "metadata": doc_data["metadata"],
                        "source": doc_data["metadata"].get("source", ""),
                        "page": doc_data["metadata"].get("page"),
                        "section": doc_data["metadata"].get("section"),
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def delete(self, kb_id: str, chunk_ids: list[str] | None = None) -> None:
        with self._lock:
            if chunk_ids is None:
                self._indices.pop(kb_id, None)
            else:
                idx = self._indices.get(kb_id, {})
                for cid in chunk_ids:
                    idx.pop(cid, None)

    def count(self, kb_id: str) -> int:
        with self._lock:
            return len(self._indices.get(kb_id, {}))


# Singleton
_lite_keyword_store: LiteKeywordStore | None = None


def get_lite_keyword_store() -> LiteKeywordStore:
    global _lite_keyword_store
    if _lite_keyword_store is None:
        _lite_keyword_store = LiteKeywordStore()
    return _lite_keyword_store
