"""Citation generation: attach source references to LLM answers."""
from __future__ import annotations

import re
from typing import Any

from app.core.logging import get_logger
from app.schemas.schemas import Citation

logger = get_logger(__name__)


class CitationManager:
    """Manage citations from retrieved documents.

    Generates citation markers in answers and provides source metadata.
    """

    def build_citations(
        self, documents: list[dict[str, Any]]
    ) -> list[Citation]:
        """Build Citation objects from retrieved documents."""
        citations = []
        seen = set()
        for i, doc in enumerate(documents):
            cid = doc.get("chunk_id", "")
            if cid in seen:
                continue
            seen.add(cid)
            citations.append(Citation(
                document_id=doc.get("document_id", ""),
                filename=doc.get("source", "未知文档"),
                page=doc.get("page"),
                section=doc.get("section"),
                title=doc.get("title"),
                chunk_id=cid,
                content=doc.get("content", ""),
                score=doc.get("score", 0.0),
            ))
        return citations

    def format_context(
        self, documents: list[dict[str, Any]]
    ) -> str:
        """Format retrieved documents into a context string for LLM prompt.

        Each chunk gets a numbered reference like [1], [2], etc.
        """
        if not documents:
            return ""

        parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.get("source", "未知文档")
            page = doc.get("page")
            section = doc.get("section", "")
            page_str = f" 第{page}页" if page else ""
            section_str = f" [{section}]" if section else ""

            parts.append(
                f"[{i}] 《{source}》{page_str}{section_str}\n{doc.get('content', '')}"
            )

        return "\n\n".join(parts)

    def extract_cited_ids(self, answer: str) -> list[int]:
        """Extract citation reference numbers [n] from answer text."""
        return [int(m) for m in re.findall(r"\[(\d+)\]", answer)]

    def build_source_list(
        self, documents: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Build a deduplicated source list for display."""
        sources = []
        seen_files = set()
        for doc in documents:
            filename = doc.get("source", "")
            if filename and filename not in seen_files:
                seen_files.add(filename)
                sources.append({
                    "filename": filename,
                    "page": doc.get("page"),
                    "section": doc.get("section"),
                    "title": doc.get("title"),
                    "chunk_id": doc.get("chunk_id"),
                    "score": doc.get("score", 0),
                })
        return sources


# Singleton
_citation_manager: CitationManager | None = None


def get_citation_manager() -> CitationManager:
    global _citation_manager
    if _citation_manager is None:
        _citation_manager = CitationManager()
    return _citation_manager
