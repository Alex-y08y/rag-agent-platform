"""Document parsing service supporting PDF, DOCX, TXT, MD, CSV, XLSX."""
from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from app.core.exceptions import FileUploadError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ParsedPage:
    """A single page/section of parsed document."""

    def __init__(
        self,
        content: str,
        page: int | None = None,
        section: str | None = None,
        title: str | None = None,
    ) -> None:
        self.content = content
        self.page = page
        self.section = section
        self.title = title


class DocumentParser:
    """Parse various document formats into structured text pages."""

    SUPPORTED_TYPES = {"pdf", "docx", "txt", "md", "csv", "xlsx"}

    def parse(self, file_path: str, file_type: str) -> list[ParsedPage]:
        """Parse a document file into a list of ParsedPage."""
        file_type = file_type.lower().lstrip(".")
        if file_type not in self.SUPPORTED_TYPES:
            raise FileUploadError(f"Unsupported file type: {file_type}")

        path = Path(file_path)
        if not path.exists():
            raise FileUploadError(f"File not found: {file_path}")

        parser_map = {
            "pdf": self._parse_pdf,
            "docx": self._parse_docx,
            "txt": self._parse_txt,
            "md": self._parse_md,
            "csv": self._parse_csv,
            "xlsx": self._parse_xlsx,
        }
        logger.info("Parsing %s file: %s", file_type, path.name)
        return parser_map[file_type](path)

    # ── PDF ────────────────────────────────────────────
    def _parse_pdf(self, path: Path) -> list[ParsedPage]:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[ParsedPage] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(ParsedPage(content=text.strip(), page=i + 1))
        logger.info("PDF parsed: %d pages", len(pages))
        return pages

    # ── DOCX ───────────────────────────────────────────
    def _parse_docx(self, path: Path) -> list[ParsedPage]:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        pages: list[ParsedPage] = []
        current_section = ""
        current_text: list[str] = []
        page_num = 1

        for para in doc.paragraphs:
            style = para.style.name if para.style else ""
            text = para.text.strip()
            if not text:
                continue

            if style.startswith("Heading"):
                # Save previous section
                if current_text:
                    pages.append(
                        ParsedPage(
                            content="\n".join(current_text),
                            page=page_num,
                            section=current_section,
                        )
                    )
                    current_text = []
                current_section = text
            else:
                current_text.append(text)

        if current_text:
            pages.append(
                ParsedPage(
                    content="\n".join(current_text),
                    page=page_num, section=current_section
                )
            )

        # If no headings found, treat whole doc as one page
        if not pages:
            full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            if full_text:
                pages.append(ParsedPage(content=full_text, page=1))

        logger.info("DOCX parsed: %d sections", len(pages))
        return pages

    # ── TXT ────────────────────────────────────────────
    def _parse_txt(self, path: Path) -> list[ParsedPage]:
        text = path.read_text(encoding="utf-8", errors="replace")
        return [ParsedPage(content=text.strip(), page=1)] if text.strip() else []

    # ── Markdown ───────────────────────────────────────
    def _parse_md(self, path: Path) -> list[ParsedPage]:
        text = path.read_text(encoding="utf-8", errors="replace")
        pages: list[ParsedPage] = []
        current_title = ""
        current_section = ""
        current_lines: list[str] = []
        page_num = 1

        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("# "):
                if current_lines:
                    pages.append(
                        ParsedPage(
                            content="\n".join(current_lines).strip(),
                            page=page_num,
                            section=current_section,
                            title=current_title,
                        )
                    )
                    current_lines = []
                current_title = stripped.lstrip("# ").strip()
                current_section = current_title
            elif stripped.startswith("## "):
                if current_lines:
                    pages.append(
                        ParsedPage(
                            content="\n".join(current_lines).strip(),
                            page=page_num,
                            section=current_section,
                            title=current_title,
                        )
                    )
                    current_lines = []
                current_section = stripped.lstrip("# ").strip()
            else:
                current_lines.append(line)

        if current_lines:
            pages.append(
                ParsedPage(
                    content="\n".join(current_lines).strip(),
                    page=page_num,
                    section=current_section,
                    title=current_title,
                )
            )

        if not pages and text.strip():
            pages.append(ParsedPage(content=text.strip(), page=1))

        logger.info("Markdown parsed: %d sections", len(pages))
        return pages

    # ── CSV ────────────────────────────────────────────
    def _parse_csv(self, path: Path) -> list[ParsedPage]:
        pages: list[ParsedPage] = []
        with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)

        if not rows:
            return []

        headers = rows[0]
        # Batch rows into pages of ~50 records
        batch_size = 50
        for batch_start in range(1, len(rows), batch_size):
            batch = rows[batch_start : batch_start + batch_size]
            lines = [", ".join(headers)]
            for row in batch:
                lines.append(", ".join(row))
            pages.append(
                ParsedPage(
                    content="\n".join(lines),
                    page=batch_start // batch_size + 1,
                    section=f"Rows {batch_start}-{batch_start + len(batch) - 1}",
                )
            )

        if not pages:
            pages.append(ParsedPage(content=", ".join(headers), page=1, section="Headers only"))

        logger.info("CSV parsed: %d pages, %d rows", len(pages), len(rows) - 1)
        return pages

    # ── XLSX ───────────────────────────────────────────
    def _parse_xlsx(self, path: Path) -> list[ParsedPage]:
        from openpyxl import load_workbook

        wb = load_workbook(str(path), read_only=True, data_only=True)
        pages: list[ParsedPage] = []

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            rows = list(ws.iter_rows(values_only=True))
            if not rows:
                continue

            headers = [str(c) if c is not None else "" for c in rows[0]]
            batch_size = 50
            for batch_start in range(1, len(rows), batch_size):
                batch = rows[batch_start : batch_start + batch_size]
                lines = [", ".join(headers)]
                for row in batch:
                    lines.append(", ".join(str(c) if c is not None else "" for c in row))
                pages.append(
                    ParsedPage(
                        content="\n".join(lines),
                        page=batch_start // batch_size + 1,
                        section=f"Sheet: {sheet_name}, Rows {batch_start}-{batch_start + len(batch) - 1}",
                        title=sheet_name,
                    )
                )

        wb.close()
        logger.info("XLSX parsed: %d pages", len(pages))
        return pages


# Singleton
_parser: DocumentParser | None = None


def get_parser() -> DocumentParser:
    global _parser
    if _parser is None:
        _parser = DocumentParser()
    return _parser
