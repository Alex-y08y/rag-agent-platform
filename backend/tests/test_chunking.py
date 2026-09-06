"""Tests for text chunking strategies."""
from __future__ import annotations

import pytest

from app.services.chunking import (
    RecursiveCharacterChunker,
    MarkdownHeaderChunker,
    StructureAwareChunker,
)


class TestRecursiveCharacterChunker:
    def test_split_short_text_returns_one_chunk(self):
        chunker = RecursiveCharacterChunker(chunk_size=1000, chunk_overlap=0)
        result = chunker.split_text("短文本")
        assert len(result) == 1
        assert result[0] == "短文本"

    def test_split_long_text_returns_multiple_chunks(self):
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        long_text = "句子一。" * 100
        result = chunker.split_text(long_text)
        assert len(result) > 1
        assert all(len(c) <= 100 for c in result)  # some overhead allowed

    def test_empty_text_returns_empty(self):
        chunker = RecursiveCharacterChunker()
        assert chunker.split_text("") == []
        assert chunker.split_text("   ") == []

    def test_chunk_overlap_preserved(self):
        chunker = RecursiveCharacterChunker(chunk_size=20, chunk_overlap=5)
        text = "a" * 60
        result = chunker.split_text(text)
        assert len(result) >= 2


class TestMarkdownHeaderChunker:
    def test_split_by_headers(self):
        chunker = MarkdownHeaderChunker()
        md = "# 标题一\n\n内容一\n\n## 子标题\n\n内容二\n\n# 标题二\n\n内容三"
        result = chunker.split_text(md)
        assert len(result) >= 2
        assert any("标题一" in h for h, _ in result)

    def test_no_headers_returns_single(self):
        chunker = MarkdownHeaderChunker()
        result = chunker.split_text("纯文本没有标题")
        assert len(result) == 1


class TestStructureAwareChunker:
    def test_chunk_pages_respects_page_boundaries(self, sample_pages):
        chunker = StructureAwareChunker(chunk_size=1000)
        chunks = chunker.chunk_pages(sample_pages, source="test.md")
        assert len(chunks) == 2
        assert chunks[0].page == 1
        assert chunks[1].page == 2

    def test_chunk_pages_splits_large_pages(self):
        from app.services.parser import ParsedPage
        chunker = StructureAwareChunker(chunk_size=50, chunk_overlap=10)
        pages = [ParsedPage(content="大" * 200, page=1)]
        chunks = chunker.chunk_pages(pages)
        assert len(chunks) > 1

    def test_chunk_markdown(self):
        chunker = StructureAwareChunker(chunk_size=1000)
        md = "# 第一章\n\n内容一\n\n# 第二章\n\n内容二"
        chunks = chunker.chunk_markdown(md, source="test.md")
        assert len(chunks) >= 2
        assert all(c.source == "test.md" for c in chunks)

    def test_chunk_index_sequential(self, sample_pages):
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_pages(sample_pages)
        indices = [c.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))
