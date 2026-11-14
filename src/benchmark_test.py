import json
import time
from pathlib import Path

from src.benchmark_dev_filter import infer_filter
from src.retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]


def evidence_hit(results, gold):
    return bool(gold) and all(
        any(
            chunk["source_id"] == evidence["source_id"]
            and chunk["page"] == evidence["page"]
            and evidence["evidence"] in chunk["text"]
            for chunk in results
        )
        for evidence in gold
    )


def main():
    questions = json.loads(
        (ROOT / "data/test_questions.json").read_text(
            encoding="utf-8-sig"
        )
    )
    questions = [
        question for question in questions
        if question["split"] == "test" and question["answerable"]
    ]

    if not questions:
        raise ValueError("No answerable test questions")

    retriever = HybridRetriever()

    for question in questions:
        if not evidence_hit(retriever.chunks, question["gold"]):
            raise ValueError(f"Evidence missing: {question['id']}")

    configurations = [
        ("baseline", 10),
        ("selected", 12),
    ]
    traces = []
    summary = {}

    for name, candidate_k in configurations:
        rows = []

        for question in questions:
            query = question["question"]
            filters = infer_filter(query)

            started = time.perf_counter()
            results = retriever.search(
                query,
                filters=filters,
                top_k=candidate_k,
            )
            elapsed_ms = (time.perf_counter() - started) * 1000

            trace = {
                "query_id": question["id"],
                "split": "test",
                "configuration": name,
                "question": query,
                "query_policy": "original question",
                "filters": filters,
                "method": "hybrid_rrf",
                "candidate_k": candidate_k,
                "results": results,
                "hit1": evidence_hit(results[:1], question["gold"]),
                "hit3": evidence_hit(results[:3], question["gold"]),
                "elapsed_ms": elapsed_ms,
            }
            rows.append(trace)
            traces.append(trace)

        count = len(rows)
        hit1 = sum(row["hit1"] for row in rows)
        hit3 = sum(row["hit3"] for row in rows)

        summary[name] = {
            "n": count,
            "hit1_count": hit1,
            "hit1_rate": hit1 / count,
            "hit3_count": hit3,
            "hit3_rate": hit3 / count,
            "mean_query_ms": sum(
                row["elapsed_ms"] for row in rows
            ) / count,
        }

        print(
            f"{name}: Hit@1 = {hit1}/{count} ({hit1/count:.1%}), "
            f"Evidence Hit@3 = {hit3}/{count} ({hit3/count:.1%})"
        )

    output = ROOT / "results/retrieval_test"
    output.mkdir(parents=True, exist_ok=True)

    for filename, content in (
        ("summary.json", summary),
        ("traces.json", traces),
    ):
        (output / filename).write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()