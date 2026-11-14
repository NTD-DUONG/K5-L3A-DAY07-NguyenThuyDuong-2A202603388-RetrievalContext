"""Persistent Chroma store with cosine search and stable-ID synchronization."""
from pathlib import Path
from typing import Any
import math

import chromadb
from chromadb.config import Settings


class ChromaStore:
    def __init__(self, path: Path, dimensions: int, model_signature: str):
        self.client = chromadb.PersistentClient(
            path=str(path), settings=Settings(anonymized_telemetry=False)
        )
        self.dimensions = dimensions
        self.model_signature = model_signature

    def collection(self, strategy: str):
        if strategy not in {"A", "B", "C"}:
            raise ValueError("Unknown strategy")
        collection = self.client.get_or_create_collection(
            name=f"day6_{strategy.lower()}", embedding_function=None,
            configuration={"hnsw": {"space": "cosine", "ef_search": 100}},
            metadata={"model_signature": self.model_signature, "dimensions": self.dimensions},
        )
        if collection.metadata.get("model_signature") != self.model_signature:
            raise ValueError("Stored model differs: use a new database directory")
        if collection.configuration["hnsw"]["space"] != "cosine":
            raise ValueError("Existing collection does not use cosine")
        return collection

    def sync(self, strategy: str, rows: list[dict], vectors: list[list[float]]) -> dict:
        if len(rows) != len(vectors) or not rows:
            raise ValueError("Require matching nonempty chunks and embeddings")
        ids = [row["chunk_id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate chunk IDs")
        if any(len(vector) != self.dimensions or not all(math.isfinite(v) for v in vector) for vector in vectors):
            raise ValueError("Invalid vector dimensions or values")
        collection = self.collection(strategy)
        before = collection.count()
        old_ids = set(collection.get(include=[])["ids"])
        for start in range(0, len(rows), 100):
            batch = rows[start:start + 100]
            collection.upsert(
                ids=ids[start:start + 100], embeddings=vectors[start:start + 100],
                documents=[row["text"] for row in batch],
                metadatas=[row["metadata"] for row in batch],
            )
        obsolete = sorted(old_ids - set(ids))
        if obsolete:
            collection.delete(ids=obsolete)
        after = collection.count()
        if after != len(rows):
            raise RuntimeError("Stored count does not match chunk snapshot")
        return {"before": before, "after": after, "obsolete_deleted": len(obsolete)}

    def search(self, strategy: str, vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict]:
        if len(vector) != self.dimensions or top_k <= 0:
            raise ValueError("Invalid query dimensions or top_k")
        if filters and (set(filters) != {"painting_tradition"} or
                        filters["painting_tradition"] not in {"dong_ho", "hang_trong", "kim_hoang", "lang_sinh", "general"}):
            raise ValueError("Unsupported metadata filter")
        collection = self.collection(strategy)
        if not collection.count():
            return []
        result = collection.query(
            query_embeddings=[vector], n_results=min(top_k, collection.count()),
            where=filters or None, include=["documents", "distances", "metadatas"],
        )
        return [{"chunk_id": cid, "text": text, "score": 1.0 - distance,
                 "cosine_distance": distance, "metadata": metadata}
                for cid, text, distance, metadata in zip(
                    result["ids"][0], result["documents"][0],
                    result["distances"][0], result["metadatas"][0])]
