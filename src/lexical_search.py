import json
from pathlib import Path

from starter.baseline import TfidfIndex


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

    index = TfidfIndex(chunks)
    print("So chunk:", len(chunks))

    question = input("Nhap cau hoi: ").strip()
    if not question:
        raise ValueError("Question must contain text")

    results = index.search(question, top_k=3)

    if not results:
        print("Khong tim thay ket qua co tu trung khop.")

    for rank, result in enumerate(results, start=1):
        print(f"\n--- Rank {rank} ---")
        print("Chunk ID:", result["id"])
        print("Score:", round(result["score"], 4))
        print("Source:", result["source"])
        print("Location:", result["page"])
        print(result["text"])


if __name__ == "__main__":
    main()