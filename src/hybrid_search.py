import hashlib
import json
from pathlib import Path

from starter.baseline import TfidfIndex
from src.day6.embeddings import MyEmbedder
from src.day6.chroma_store import ChromaStore


ROOT = Path(__file__).resolve().parents[1]


def main():
    chunks_path = ROOT / "data/day6/chunks.jsonl"
    chunks = []

    for line in chunks_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            chunks.append({
                "id": row["chunk_id"],
                "text": row["text"],
                "source_id": row["document_id"],
                "source": row["source"],
                "page": row["page"],
                "metadata": row["metadata"],
            })

    chunks_by_id = {chunk["id"]: chunk for chunk in chunks}
    lexical = TfidfIndex(chunks)

    config_path = ROOT / "data/day6/ingestion_summary.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))

    embedder = MyEmbedder(config["model"]["model_name"])
    signature = hashlib.sha256(
        json.dumps(embedder.configuration, sort_keys=True).encode()
    ).hexdigest()

    if signature != config["model_signature"]:
        raise ValueError("Embedding configuration differs from Day 6")

    database_path = ROOT / ".runtime/chroma"
    if not (database_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Chroma database missing: {database_path}")

    store = ChromaStore(database_path, embedder.dimensions, signature)

    question = input("Nhap cau hoi: ").strip()
    if not question:
        raise ValueError("Question must contain text")

    dense_results = store.search(
        "C", embedder.embed_query(question), top_k=10
    )
    lexical_results = lexical.search(question, top_k=10)

    rankings = [
        [result["chunk_id"] for result in dense_results],
        [result["id"] for result in lexical_results],
    ]

    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (60 + rank)

    selected_ids = sorted(
        scores,
        key=lambda chunk_id: (-scores[chunk_id], chunk_id),
    )[:3]

    for rank, chunk_id in enumerate(selected_ids, start=1):
        chunk = chunks_by_id[chunk_id]
        print(f"\n--- Rank {rank} ---")
        print("Chunk ID:", chunk_id)
        print("RRF score:", round(scores[chunk_id], 6))
        print("Source:", chunk["source"])
        print("Location:", chunk["page"])
        print(chunk["text"])


if __name__ == "__main__":
    main()