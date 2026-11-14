import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from starter.agentic import run_agent
from src.benchmark_dev_filter import infer_filter
from src.gemini_controller import GeminiController, INSTRUCTIONS
from src.retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]

FIRST_SEARCH_RULE = (
    "For the first search, copy the user's question exactly into query.\n"
    "Only rewrite the query after receiving the first search result.\n"
)


def context_has_evidence(question, context_ids, chunks_by_id):
    if not question["answerable"]:
        return None

    matches = [
        any(
            chunks_by_id[chunk_id]["source_id"] == gold["source_id"]
            and chunks_by_id[chunk_id]["page"] == gold["page"]
            and gold["evidence"] in chunks_by_id[chunk_id]["text"]
            for chunk_id in context_ids
        )
        for gold in question["gold"]
    ]

    if question.get("evidence_rule", "all") == "all":
        return bool(matches) and all(matches)

    return any(matches)


def main():
    questions = json.loads(
        (ROOT / "data/test_questions.json").read_text(
            encoding="utf-8-sig"
        )
    )
    selected = json.loads(
        (ROOT / "configs/selected_agent.json").read_text(
            encoding="utf-8-sig"
        )
    )

    assert len(questions) >= 7
    assert all(question["split"] == "test" for question in questions)
    assert len({q["id"] for q in questions}) == len(questions)
    assert sum(q["answerable"] for q in questions) >= 5
    assert sum(not q["answerable"] for q in questions) >= 2

    if FIRST_SEARCH_RULE not in INSTRUCTIONS:
        raise ValueError("Cannot reconstruct the baseline prompt")

    baseline_prompt = INSTRUCTIONS.replace(FIRST_SEARCH_RULE, "", 1)

    configurations = [
        {
            "name": "baseline",
            "top_k": 3,
            "budget": 6000,
            "candidate_k": 10,
            "max_searches": 3,
            "instructions": baseline_prompt,
        },
        {
            "name": "selected",
            "top_k": selected["top_k"],
            "budget": selected["context_budget_chars"],
            "candidate_k": selected["candidate_k"],
            "max_searches": selected["max_searches"],
            "instructions": INSTRUCTIONS,
        },
    ]

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid4().hex[:8]
    )
    output = ROOT / "results/agent_test" / run_id
    output.mkdir(parents=True, exist_ok=True)

    (output / "protocol.json").write_text(
        json.dumps(
            {
                "questions": questions,
                "configurations": configurations,
                "selected_configuration": selected,
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )

    retriever = HybridRetriever()
    summary = {}

    for config in configurations:
        case_output = output / config["name"]
        case_output.mkdir()
        traces = []

        controller = GeminiController(
            instructions=config["instructions"]
        )

        try:
            if controller.model != selected["model"]:
                raise ValueError("Model differs from selected configuration")

            for question in questions:
                controller.calls = []
                filters = infer_filter(question["question"])
                started = time.perf_counter()
                events = []

                try:
                    result = run_agent(
                        question["question"],
                        controller,
                        retriever,
                        events=events,
                        filters=filters,
                        max_searches=config["max_searches"],
                        top_k=config["top_k"],
                        budget=config["budget"],
                    )
                except Exception as error:
                    result = {
                        "status": "error",
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "events": events,
                    }

                elapsed = time.perf_counter() - started

                searches = [
                    event for event in result.get("events", [])
                    if event["event"] == "search"
                ]
                context = searches[-1]["context"] if searches else ""
                context_ids = (
                    searches[-1]["context_ids"] if searches else []
                )

                # Gold is used only after the agent has finished.
                context_hit = context_has_evidence(
                    question, context_ids, retriever.chunks_by_id
                )
                if question["answerable"] and result["status"] == "error":
                    context_hit = False

                trace = {
                    "query_id": question["id"],
                    "split": "test",
                    "question": question["question"],
                    "answerable": question["answerable"],
                    "configuration": config["name"],
                    "model": controller.model,
                    "filters": filters,
                    "result": result,
                    "final_context": context,
                    "context_ids": context_ids,
                    "context_evidence_hit": context_hit,
                    "model_calls": list(controller.calls),
                    "elapsed_seconds_excluding_setup": elapsed,
                }
                traces.append(trace)

                (case_output / f"{question['id']}.json").write_text(
                    json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )

                print(
                    config["name"],
                    question["id"],
                    result["status"],
                    f"{elapsed:.1f}s",
                    flush=True,
                )

        finally:
            controller.close()

        answerable = [t for t in traces if t["answerable"]]
        context_hits = sum(
            bool(t["context_evidence_hit"]) for t in answerable
        )
        calls = [
            call for trace in traces for call in trace["model_calls"]
        ]

        summary[config["name"]] = {
            "total": len(traces),
            "answered": sum(
                t["result"]["status"] == "answered" for t in traces
            ),
            "abstain": sum(
                t["result"]["status"] == "abstain" for t in traces
            ),
            "errors": sum(
                t["result"]["status"] == "error" for t in traces
            ),
            "context_evidence_count": context_hits,
            "answerable_count": len(answerable),
            "context_evidence_rate": context_hits / len(answerable),
            "model_calls": len(calls),
            "reported_total_tokens": sum(
                (call.get("usage") or {}).get("total_token_count", 0)
                or 0
                for call in calls
            ),
            "total_seconds_excluding_setup": sum(
                t["elapsed_seconds_excluding_setup"] for t in traces
            ),
        }

    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("Saved:", output)


if __name__ == "__main__":
    main()
