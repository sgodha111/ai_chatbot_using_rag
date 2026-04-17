from __future__ import annotations

from typing import List


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    clean_text = " ".join(text.split())
    if not clean_text:
        return []

    chunks: List[str] = []
    start = 0
    text_length = len(clean_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = clean_text[start:end]

        # Try to cut on a word boundary when possible for cleaner retrieval.
        if end < text_length:
            split_at = chunk.rfind(" ")
            if split_at > chunk_size // 2:
                end = start + split_at
                chunk = clean_text[start:end]

        chunks.append(chunk.strip())
        if end >= text_length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks
