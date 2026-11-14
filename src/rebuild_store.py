import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

from src.day6.embeddings import MyEmbedder
from src.day6.chroma_store import ChromaStore


ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads(
        (ROOT / "data/day6/ingestion_summary.json").read_text(
            encoding="utf-8-sig"
        )
    )
    model_config = config["model"]

    # Download the exact revision, or reuse the cached snapshot.
    snapshot_download(
        repo_id=model_config["model_name"],
        revision=model_config["revision"],
        allow_patterns=[
            "*.json",
            "*.safetensors",
            "sentencepiece.bpe.model",
        ],
    )

    embedder = MyEmbedder(model_config["model_name"])
    signature = hashlib.sha256(
        json.dumps(embedder.configuration, sort_keys=True).encode()
    ).hexdigest()

    if signature != config["model_signature"]:
        raise ValueError("Embedding configuration differs from Day 6")

    chunks = [
        json.loads(line)
        for line in (ROOT / "data/day6/chunks.jsonl")
        .read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    if not chunks:
        raise ValueError("No chunks")

    print("Embedding chunks:", len(chunks), flush=True)
    vectors = embedder.embed_documents(
        [chunk["text"] for chunk in chunks]
    )

    database_path = ROOT / ".runtime/chroma"
    store = ChromaStore(
        database_path,
        embedder.dimensions,
        signature,
    )
    result = store.sync("C", chunks, vectors)

    stored = store.collection("C").get(
        include=["documents", "metadatas"]
    )
    expected = {chunk["chunk_id"]: chunk for chunk in chunks}

    assert set(stored["ids"]) == set(expected)
    for chunk_id, text, metadata in zip(
        stored["ids"], stored["documents"], stored["metadatas"]
    ):
        assert text == expected[chunk_id]["text"]
        assert metadata == expected[chunk_id]["metadata"]

    print("Database:", database_path)
    print("Collection: day6_c")
    print("Stored chunks:", result["after"])
    print("Document and metadata check: OK")


if __name__ == "__main__":
    main()