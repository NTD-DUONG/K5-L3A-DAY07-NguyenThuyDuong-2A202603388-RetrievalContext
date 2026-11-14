import hashlib
import json
from pathlib import Path

from starter.baseline import TfidfIndex
from src.day6.embeddings import MyEmbedder
from src.day6.chroma_store import ChromaStore


ROOT = Path(__file__).resolve().parents[1]


class HybridRetriever:
    def __init__(self):
        chunks_path = ROOT / "data/day6/chunks.jsonl"
        self.chunks = []

        for line in chunks_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                self.chunks.append({
                    "id": row["chunk_id"],
                    "text": row["text"],
                    "source_id": row["document_id"],
                    "source": row["source"],
                    "page": row["page"],
                    "metadata": row["metadata"],
                })

        self.chunks_by_id = {
            chunk["id"]: chunk for chunk in self.chunks
        }
        self.lexical = TfidfIndex(self.chunks)

        config = json.loads(
            (ROOT / "data/day6/ingestion_summary.json").read_text(
                encoding="utf-8-sig"
            )
        )

        self.embedder = MyEmbedder(config["model"]["model_name"])
        signature = hashlib.sha256(
            json.dumps(
                self.embedder.configuration, sort_keys=True
            ).encode()
        ).hexdigest()

        if signature != config["model_signature"]:
            raise ValueError("Embedding configuration differs from Day 6")

        database_path = ROOT / ".runtime/chroma"
        if not (database_path / "chroma.sqlite3").is_file():
            raise FileNotFoundError(database_path)

        self.store = ChromaStore(
            database_path, self.embedder.dimensions, signature
        )

        stored_ids = set(
            self.store.collection("C").get(include=[])["ids"]
        )
        if stored_ids != set(self.chunks_by_id):
            raise ValueError("Chroma IDs differ from local chunk IDs")

    def search(self, query, *, filters=None, top_k=3):
        if not query.strip() or top_k < 1:
            raise ValueError("Invalid query or top_k")

        candidate_k = max(10, top_k)
        dense = self.store.search(
            "C",
            self.embedder.embed_query(query),
            top_k=candidate_k,
            filters=filters,
        )

        lexical = self.lexical
        if filters:
            filtered_chunks = [
                chunk for chunk in self.chunks
                if chunk["metadata"].get("painting_tradition")
                == filters["painting_tradition"]
            ]
            lexical = TfidfIndex(filtered_chunks)

        sparse = lexical.search(query, top_k=candidate_k)

        scores = {}
        rankings = [
            [result["chunk_id"] for result in dense],
            [result["id"] for result in sparse],
        ]
        for ranking in rankings:
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0) + 1 / (60 + rank)
                )

        ranked_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )[:top_k]

        return [
            {
                **self.chunks_by_id[chunk_id],
                "score": scores[chunk_id],
            }
            for chunk_id in ranked_ids
        ]


def main():
    retriever = HybridRetriever()
    question = input("Nhap cau hoi: ").strip()
    results = retriever.search(question, filters=None, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(f"\nRank {rank}: {result['id']}")
        print("Source:", result["source"])
        print("Location:", result["page"])
        print(result["text"])


if __name__ == "__main__":
    main()