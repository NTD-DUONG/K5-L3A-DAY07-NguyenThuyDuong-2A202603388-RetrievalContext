import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from starter.agentic import run_agent
from starter.baseline import hit
from src.benchmark_dev_filter import infer_filter
from src.gemini_controller import GeminiController, INSTRUCTIONS
from src.evaluate_agent_test import FIRST_SEARCH_RULE
from src.retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fixed agent configuration on dev")
    parser.add_argument("--configuration", choices=("baseline", "selected"), default="selected")
    args = parser.parse_args()
    selected = json.loads(
        (ROOT / "configs/selected_agent.json").read_text(encoding="utf-8-sig")
    )
    instructions = INSTRUCTIONS
    settings = {
        "max_searches": selected["max_searches"],
        "top_k": selected["top_k"],
        "budget": selected["context_budget_chars"],
    }
    if args.configuration == "baseline":
        if FIRST_SEARCH_RULE not in instructions:
            raise ValueError("Cannot reconstruct the baseline prompt")
        instructions = instructions.replace(FIRST_SEARCH_RULE, "", 1)
        settings = {"max_searches": 3, "top_k": 3, "budget": 6000}

    questions = json.loads(
        (ROOT / "data/questions.json").read_text(encoding="utf-8-sig")
    )
    questions = [
        question for question in questions
        if question["split"] == "dev"
    ]

    if not questions:
        raise ValueError("No dev questions")

    ids = [question["id"] for question in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "_" + uuid4().hex[:8]
    )
    output = ROOT / "results/agent_dev" / run_id
    output.mkdir(parents=True, exist_ok=True)
    (output / "protocol.json").write_text(
        json.dumps({
            "configuration": args.configuration,
            "settings": settings,
            "instructions": instructions,
            "model": selected["model"],
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    retriever = HybridRetriever()
    controller = GeminiController(instructions=instructions)
    traces = []

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
                    filters=filters,
                    events=events,
                    **settings,
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
            final_context = searches[-1]["context"] if searches else ""
            context_ids = (
                searches[-1]["context_ids"] if searches else []
            )

            context_hit = None
            if question["answerable"]:
                context_hit = result["status"] != "error" and any(
                    hit(retriever.chunks_by_id[chunk_id], question["gold"])
                    for chunk_id in context_ids
                )

            trace = {
                "query_id": question["id"],
                "split": "dev",
                "question": question["question"],
                "answerable": question["answerable"],
                "model": controller.model,
                "configuration": {
                    "name": args.configuration,
                    "method": "hybrid_rrf",
                    "candidate_k": max(10, settings["top_k"]),
                    "rrf_constant": 60,
                    "filters": filters,
                    "max_searches": settings["max_searches"],
                    "top_k": settings["top_k"],
                    "context_budget_chars": settings["budget"],
                },
                "result": result,
                "final_context": final_context,
                "context_ids": context_ids,
                "context_evidence_hit": context_hit,
                "model_calls": list(controller.calls),
                "elapsed_seconds_excluding_setup": elapsed,
            }
            traces.append(trace)

            (output / f"{question['id']}.json").write_text(
                json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

            print(
                question["id"],
                result["status"],
                f"{elapsed:.1f}s",
                flush=True,
            )

        answerable = [
            trace for trace in traces if trace["answerable"]
        ]
        context_hits = sum(
            bool(trace["context_evidence_hit"]) for trace in answerable
        )

        summary = {
            "configuration": args.configuration,
            "total": len(traces),
            "answered": sum(
                trace["result"]["status"] == "answered" for trace in traces
            ),
            "abstain": sum(
                trace["result"]["status"] == "abstain" for trace in traces
            ),
            "errors": sum(
                trace["result"]["status"] == "error" for trace in traces
            ),
            "context_evidence_count": context_hits,
            "answerable_count": len(answerable),
            "context_evidence_rate": (
                context_hits / len(answerable) if answerable else None
            ),
            "evaluation_note": (
                "Context evidence uses declared gold quotes. "
                "Answer correctness, citation support and abstention "
                "require content review. Error cases count as context misses."
            ),
        }

        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print("Saved:", output)
    finally:
        controller.close()


if __name__ == "__main__":
    main()
