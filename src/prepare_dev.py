import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DAY6 = Path(
    r"E:\AI\Lab\Labday6"
    r"\K5-L3A-DAY06-NguyenThuyDuong-2A202603388-DataIngestion"
)


def main():
    source_path = DAY6 / "benchmark/questions.json"
    old_questions = json.loads(source_path.read_text(encoding="utf-8-sig"))

    chunks_path = ROOT / "data/day6/chunks.jsonl"
    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]

    questions = []

    for old in old_questions:
        if old["split"] != "dev":
            continue

        gold = {
            "source_id": old["gold_source_id"],
            "page": old["gold_page"],
            "evidence": old["evidence_quote"],
        }

        evidence_found = any(
            chunk["document_id"] == gold["source_id"]
            and chunk["page"] == gold["page"]
            and gold["evidence"] in chunk["text"]
            for chunk in chunks
        )

        if not evidence_found:
            raise ValueError(f"Evidence missing for {old['id']}")

        questions.append({
            "id": f"dev_{len(questions) + 1:02d}",
            "split": "dev",
            "answerable": True,
            "question": old["question"],
            "expected_answer": old["expected_answer"],
            "gold": [gold],
        })

    if len(questions) < 15:
        raise ValueError("Need at least 15 answerable dev questions")

    output_path = ROOT / "data/questions.json"
    output_path.write_text(
        json.dumps(questions, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Saved:", output_path)
    print("So cau dev co dap an:", len(questions))
    print("Evidence check: OK")


if __name__ == "__main__":
    main()