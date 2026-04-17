from __future__ import annotations

from io import BytesIO
from pathlib import Path

from pypdf import PdfReader


def load_document(file_name: str, file_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in {".txt", ".md"}:
        return file_bytes.decode("utf-8", errors="ignore").strip()
    if suffix == ".pdf":
        return _load_pdf(file_bytes)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(page.strip() for page in pages if page.strip())
