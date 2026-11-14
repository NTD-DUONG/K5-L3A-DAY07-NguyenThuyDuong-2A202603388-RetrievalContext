import hashlib
import json
import time
from pathlib import Path

from starter.baseline import TfidfIndex, hit
from src.day6.embeddings import MyEmbedder
from src.day6.chroma_store import ChromaStore


ROOT = Path(__file__).resolve().parents[1]


def main():
    chunks_path = ROOT / "data/day6/chunks.jsonl"
    chunks = []

    for line in chunks_path.read_text(encoding="utf-8-sig").splitlines():
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

    questions_path = ROOT / "data/questions.json"
    all_questions = json.loads(
        questions_path.read_text(encoding="utf-8-sig")
    )
    questions = [
        question for question in all_questions
        if question["split"] == "dev" and question["answerable"]
    ]

    if not questions:
        raise ValueError("No answerable dev questions")

    if len({question["id"] for question in questions}) != len(questions):
        raise ValueError("Duplicate question IDs")

    for question in questions:
        if not any(hit(chunk, question["gold"]) for chunk in chunks):
            raise ValueError(f"Evidence missing: {question['id']}")

    setup_start = time.perf_counter()
    lexical = TfidfIndex(chunks)

    config = json.loads(
        (ROOT / "data/day6/ingestion_summary.json").read_text(
            encoding="utf-8-sig"
        )
    )
    embedder = MyEmbedder(config["model"]["model_name"])
    signature = hashlib.sha256(
        json.dumps(embedder.configuration, sort_keys=True).encode()
    ).hexdigest()

    if signature != config["model_signature"]:
        raise ValueError("Embedding configuration differs from Day 6")

    database_path = ROOT / ".runtime/chroma"
    if not (database_path / "chroma.sqlite3").is_file():
        raise FileNotFoundError(database_path)

    store = ChromaStore(database_path, embedder.dimensions, signature)

    stored_ids = set(store.collection("C").get(include=[])["ids"])
    if stored_ids != set(chunks_by_id):
        raise ValueError("Chroma IDs differ from local chunk IDs")

    setup_seconds = time.perf_counter() - setup_start
    traces = []

    for question in questions:
        query = question["question"]

        started = time.perf_counter()
        dense_results = store.search(
            "C", embedder.embed_query(query), top_k=10
        )
        dense_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        lexical_results = lexical.search(query, top_k=10)
        lexical_ms = (time.perf_counter() - started) * 1000

        dense_ids = [result["chunk_id"] for result in dense_results]
        lexical_ids = [result["id"] for result in lexical_results]

        started = time.perf_counter()
        scores = {}
        for ranking in (dense_ids, lexical_ids):
            for rank, chunk_id in enumerate(ranking, start=1):
                scores[chunk_id] = (
                    scores.get(chunk_id, 0.0) + 1 / (60 + rank)
                )

        hybrid_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )
        fusion_ms = (time.perf_counter() - started) * 1000

        methods = [
            ("dense", dense_ids, dense_ms),
            ("tfidf", lexical_ids, lexical_ms),
            (
                "hybrid",
                hybrid_ids,
                dense_ms + lexical_ms + fusion_ms,
            ),
        ]

        for method, ranked_ids, elapsed_ms in methods:
            results = []
            for chunk_id in ranked_ids:
                chunk = chunks_by_id[chunk_id]
                results.append({
                    **chunk,
                    "hit": hit(chunk, question["gold"]),
                })

            traces.append({
                "query_id": question["id"],
                "question": query,
                "method": method,
                "results": results,
                "hit1": bool(results and results[0]["hit"]),
                "hit3": any(result["hit"] for result in results[:3]),
                "elapsed_ms": elapsed_ms,
            })

        print("Done:", question["id"])

    summary = {}
    for method in ("dense", "tfidf", "hybrid"):
        rows = [trace for trace in traces if trace["method"] == method]
        total = len(rows)
        hit1 = sum(trace["hit1"] for trace in rows)
        hit3 = sum(trace["hit3"] for trace in rows)

        summary[method] = {
            "n": total,
            "hit1_count": hit1,
            "hit1_rate": hit1 / total,
            "hit3_count": hit3,
            "hit3_rate": hit3 / total,
            "mean_query_ms": sum(
                trace["elapsed_ms"] for trace in rows
            ) / total,
        }

        print(
            f"{method}: "
            f"Hit@1 = {hit1}/{total} ({hit1 / total:.1%}), "
            f"Evidence Hit@3 = {hit3}/{total} ({hit3 / total:.1%})"
        )

    output = ROOT / "results/retrieval_dev"
    output.mkdir(parents=True, exist_ok=True)

    report = {
        "split": "dev",
        "candidate_k": 10,
        "evaluation_cutoffs": [1, 3],
        "rrf_constant": 60,
        "filter": None,
        "model_signature": signature,
        "setup_seconds": setup_seconds,
        "timing_note": (
            "Sequential queries after setup. Hybrid reuses measured "
            "dense and lexical results and sums their times plus fusion."
        ),
        "external_api_cost_usd": 0,
        "methods": summary,
    }

    for filename, content in (
        ("summary.json", report),
        ("traces.json", traces),
    ):
        (output / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()