from __future__ import annotations

from typing import List
from uuid import uuid4

import chromadb
from chromadb.api.models.Collection import Collection

from rag.config import CHROMA_COLLECTION, VECTOR_DB_DIR


class VectorStore:
    def __init__(self, collection_name: str = CHROMA_COLLECTION) -> None:
        self.client = chromadb.PersistentClient(path=str(VECTOR_DB_DIR))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        file_name: str,
        chunks: List[str],
        embeddings: List[List[float]],
    ) -> int:
        ids = [str(uuid4()) for _ in chunks]
        metadatas = [
            {"source": file_name, "chunk_index": index}
            for index, _ in enumerate(chunks)
        ]
        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        return len(ids)

    def search(self, query_embedding: List[float], top_k: int) -> List[dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        matches = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            matches.append(
                {
                    "document": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )
        return matches

    def count(self) -> int:
        return self.collection.count()
