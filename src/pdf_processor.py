from __future__ import annotations

import io

from pypdf import PdfReader

from src.utils import chunk_text


def extract_pdf_text(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def extract_chunks_from_pdf(file_bytes: bytes, chunk_size: int = 1800) -> list[str]:
    text = extract_pdf_text(file_bytes)
    return chunk_text(text, chunk_size=chunk_size)
