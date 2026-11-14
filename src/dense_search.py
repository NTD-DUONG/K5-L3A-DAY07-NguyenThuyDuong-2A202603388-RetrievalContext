import hashlib
import json
from pathlib import Path

from src.day6.embeddings import MyEmbedder
from src.day6.chroma_store import ChromaStore


ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = ROOT / "data/day6/ingestion_summary.json"
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))

    embedder = MyEmbedder(saved_config["model"]["model_name"])
    signature = hashlib.sha256(
        json.dumps(embedder.configuration, sort_keys=True).encode()
    ).hexdigest()

    if signature != saved_config["model_signature"]:
        raise ValueError("Embedding configuration differs from Day 6")

    database_path = ROOT / ".runtime/chroma"
    if not (database_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Chroma database missing: {database_path}")

    store = ChromaStore(database_path, embedder.dimensions, signature)

    question = input("Nhap cau hoi: ").strip()
    if not question:
        raise ValueError("Question must contain text")

    vector = embedder.embed_query(question)
    results = store.search("C", vector, top_k=3)

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Rank {rank} ---")
        print("Chunk ID:", result["chunk_id"])
        print("Score:", round(result["score"], 4))
        print("Source:", result["metadata"].get("source"))
        print("Location:", result["metadata"].get("page"))
        print(result["text"])


if __name__ == "__main__":
    main()