"""Context Compression: filter and compress retrieved chunks to relevant info."""
from __future__ import annotations

import re
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class ContextCompressor:
    """Compress retrieved context to reduce noise and token usage.

    Strategies:
    1. Score threshold filtering
    2. Redundancy removal (deduplicate similar chunks)
    3. Sentence-level relevance filtering
    4. Truncation to max context length
    """

    def __init__(
        self,
        min_score: float = 0.01,
        max_context_chars: int = 8000,
        similarity_threshold: float = 0.85,
    ) -> None:
        self.min_score = min_score
        self.max_context_chars = max_context_chars
        self.similarity_threshold = similarity_threshold

    def compress(
        self,
        query: str,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Compress retrieved documents.

        Args:
            query: Original query for relevance filtering.
            documents: Retrieved documents with scores.

        Returns:
            Compressed document list.
        """
        if not documents:
            return []

        original_count = len(documents)
        start_len = sum(len(d.get("content", "")) for d in documents)

        # Step 1: Filter by minimum score
        filtered = [
            d for d in documents
            if d.get("score", 0) >= self.min_score
        ]

        # Step 2: Remove near-duplicate content
        filtered = self._remove_duplicates(filtered)

        # Step 3: Sentence-level relevance filtering
        query_terms = self._extract_terms(query)
        for doc in filtered:
            doc["content"] = self._filter_sentences(doc["content"], query_terms)

        # Step 4: Truncate to max context length
        compressed = self._truncate_context(filtered)

        end_len = sum(len(d.get("content", "")) for d in compressed)
        logger.info(
            "Context compression: %d->%d docs, %d->%d chars (%.0f%% reduction)",
            original_count, len(compressed),
            start_len, end_len,
            (1 - end_len / start_len) * 100 if start_len > 0 else 0,
        )

        return compressed

    def _remove_duplicates(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Remove chunks with highly overlapping content."""
        unique: list[dict[str, Any]] = []
        seen_contents: list[str] = []

        for doc in documents:
            content = doc.get("content", "")
            is_dup = False
            for seen in seen_contents:
                if self._text_similarity(content, seen) > self.similarity_threshold:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(doc)
                seen_contents.append(content)

        return unique

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """Simple character n-gram Jaccard similarity."""
        if not a or not b:
            return 0.0
        # Use 3-char shingles
        def shingles(text: str, n: int = 3) -> set[str]:
            text = re.sub(r"\s+", "", text)
            return {text[i:i+n] for i in range(len(text) - n + 1)}

        sa, sb = shingles(a), shingles(b)
        if not sa or not sb:
            return 0.0
        return len(sa & sb) / len(sa | sb)

    @staticmethod
    def _extract_terms(query: str) -> set[str]:
        """Extract meaningful terms from query."""
        # Simple: split by punctuation and whitespace, keep terms > 1 char
        terms = re.findall(r"[\w\u4e00-\u9fff]{2,}", query.lower())
        return set(terms)

    def _filter_sentences(self, content: str, query_terms: set[str]) -> str:
        """Keep sentences that contain at least one query term."""
        if not query_terms:
            return content

        sentences = re.split(r"(?<=[。！？.!?])\s*", content)
        relevant = []
        for sent in sentences:
            sent_lower = sent.lower()
            if any(term in sent_lower for term in query_terms):
                relevant.append(sent)
            elif len(relevant) < 2:  # Keep first couple sentences for context
                relevant.append(sent)

        result = "".join(relevant).strip()
        return result if result else content

    def _truncate_context(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Truncate total context to max characters, preserving highest-scored docs."""
        result = []
        total = 0
        for doc in documents:
            content = doc.get("content", "")
            if total + len(content) > self.max_context_chars:
                remaining = self.max_context_chars - total
                if remaining > 100:
                    doc = doc.copy()
                    doc["content"] = content[:remaining] + "..."
                    result.append(doc)
                break
            result.append(doc)
            total += len(content)
        return result


# Singleton
_compressor: ContextCompressor | None = None


def get_context_compressor() -> ContextCompressor:
    global _compressor
    if _compressor is None:
        _compressor = ContextCompressor()
    return _compressor
