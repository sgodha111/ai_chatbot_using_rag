from __future__ import annotations

from typing import List, Tuple

from rag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DEFAULT_EMBED_MODEL,
    DEFAULT_LLM_MODEL,
    TOP_K_RESULTS,
)
from rag.document_loader import load_document
from rag.ollama_client import OllamaClient
from rag.text_splitter import split_text
from rag.vector_store import VectorStore


SYSTEM_PROMPT = """You are a helpful AI assistant for question answering over uploaded documents.
Use the provided context to answer the user's question.
If the answer is not in the context, say you do not know based on the uploaded files.
Keep the answer clear and concise."""


class RAGPipeline:
    def __init__(self) -> None:
        self.ollama = OllamaClient()
        self.vector_store = VectorStore()

    def ingest_document(
        self,
        file_name: str,
        file_bytes: bytes,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> Tuple[int, int]:
        text = load_document(file_name, file_bytes)
        chunks = split_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        if not chunks:
            return 0, 0

        embeddings = [self.ollama.embed(chunk, embedding_model) for chunk in chunks]
        inserted = self.vector_store.add_chunks(
            file_name=file_name,
            chunks=chunks,
            embeddings=embeddings,
        )
        return len(text), inserted

    def ask(
        self,
        question: str,
        llm_model: str = DEFAULT_LLM_MODEL,
        embedding_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = TOP_K_RESULTS,
    ) -> Tuple[str, List[dict]]:
        if self.vector_store.count() == 0:
            return (
                "No documents are indexed yet. Upload files from the sidebar and click "
                "`Ingest documents` before asking a question.",
                [],
            )

        query_embedding = self.ollama.embed(question, embedding_model)
        matches = self.vector_store.search(query_embedding=query_embedding, top_k=top_k)
        if not matches:
            return (
                "I couldn't find relevant context in the indexed documents for that question.",
                [],
            )

        context = "\n\n".join(
            f"Source: {match['metadata']['source']}\nContent: {match['document']}"
            for match in matches
        )
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )
        answer = self.ollama.generate(prompt=prompt, model=llm_model)
        return answer, matches
