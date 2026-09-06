"""Text chunking strategies: recursive, markdown-header, structure-aware."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.core.config import settings
from app.core.logging import get_logger
from app.services.parser import ParsedPage

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A text chunk with metadata."""
    content: str
    chunk_index: int
    page: int | None = None
    section: str | None = None
    title: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RecursiveCharacterChunker:
    """Split text by recursively trying different separators."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]

    def split_text(self, text: str) -> list[str]:
        """Split text into chunks using recursive separator approach."""
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []

        # Find the best separator
        separator = self.separators[-1]
        for sep in self.separators:
            if sep and sep in text:
                separator = sep
                break

        # Handle empty separator (fallback: split by character)
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)
        splits = [s for s in splits if s.strip()]

        chunks: list[str] = []
        current = ""

        for piece in splits:
            candidate = (current + separator + piece).strip() if current else piece
            if len(candidate) > self.chunk_size and current:
                chunks.append(current.strip())
                # Carry overlap
                overlap_text = current[-self.chunk_overlap:] if self.chunk_overlap > 0 else ""
                current = (overlap_text + separator + piece).strip()
            else:
                current = candidate

        if current.strip():
            chunks.append(current.strip())

        return chunks


class MarkdownHeaderChunker:
    """Split markdown by headers, preserving header hierarchy."""

    HEADER_PATTERN = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def split_text(self, text: str, chunk_size: int | None = None) -> list[tuple[str, str]]:
        """Split markdown into (header_path, content) tuples."""
        size = chunk_size or settings.CHUNK_SIZE
        matches = list(self.HEADER_PATTERN.finditer(text))

        if not matches:
            return [("", text)] if text.strip() else []

        results: list[tuple[str, str]] = []
        header_stack: list[tuple[int, str]] = []

        for i, match in enumerate(matches):
            level = len(match.group(1))
            header_text = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()

            # Update header stack
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, header_text))
            header_path = " > ".join(h[1] for h in header_stack)

            if content:
                results.append((header_path, content))

        return results


class StructureAwareChunker:
    """Chunk documents respecting their structural boundaries (pages, sections)."""

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
        self.recursive = RecursiveCharacterChunker(self.chunk_size, self.chunk_overlap)

    def chunk_pages(
        self,
        pages: list[ParsedPage],
        source: str | None = None,
    ) -> list[Chunk]:
        """Convert parsed pages into chunks, respecting structure."""
        chunks: list[Chunk] = []
        chunk_index = 0

        for page in pages:
            content = page.content.strip()
            if not content:
                continue

            # If content fits in one chunk, keep it as-is (preserve structure)
            if len(content) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=content,
                        chunk_index=chunk_index,
                        page=page.page,
                        section=page.section,
                        title=page.title,
                        source=source,
                        metadata={"page": page.page, "section": page.section},
                    )
                )
                chunk_index += 1
            else:
                # Split large pages with recursive chunker
                sub_chunks = self.recursive.split_text(content)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            content=sub,
                            chunk_index=chunk_index,
                            page=page.page,
                            section=page.section,
                            title=page.title,
                            source=source,
                            metadata={"page": page.page, "section": page.section},
                        )
                    )
                    chunk_index += 1

        logger.info(
            "Structure-aware chunking produced %d chunks from %d pages",
            len(chunks), len(pages),
        )
        return chunks

    def chunk_markdown(
        self,
        text: str,
        source: str | None = None,
    ) -> list[Chunk]:
        """Chunk markdown by headers, then recursively split large sections."""
        md_chunker = MarkdownHeaderChunker()
        sections = md_chunker.split_text(text)

        chunks: list[Chunk] = []
        chunk_index = 0

        for header_path, content in sections:
            content = content.strip()
            if not content:
                continue

            if len(content) <= self.chunk_size:
                chunks.append(
                    Chunk(
                        content=f"{header_path}\n\n{content}" if header_path else content,
                        chunk_index=chunk_index,
                        section=header_path or None,
                        source=source,
                        metadata={"section": header_path},
                    )
                )
                chunk_index += 1
            else:
                sub_chunks = self.recursive.split_text(content)
                for sub in sub_chunks:
                    chunks.append(
                        Chunk(
                            content=f"{header_path}\n\n{sub}" if header_path else sub,
                            chunk_index=chunk_index,
                            section=header_path or None,
                            source=source,
                            metadata={"section": header_path},
                        )
                    )
                    chunk_index += 1

        logger.info("Markdown chunking produced %d chunks", len(chunks))
        return chunks


def get_chunker() -> StructureAwareChunker:
    return StructureAwareChunker()
