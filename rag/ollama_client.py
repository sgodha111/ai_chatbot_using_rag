from __future__ import annotations

from typing import List

import requests

from rag.config import OLLAMA_BASE_URL


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_BASE_URL) -> None:
        self.base_url = base_url.rstrip("/")

    def embed(self, text: str, model: str) -> List[float]:
        response = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["embedding"]

    def generate(self, prompt: str, model: str) -> str:
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["response"].strip()
