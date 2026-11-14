import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from starter.agentic import run_agent
from src.benchmark_dev_filter import infer_filter
from src.gemini_controller import GeminiController
from src.retriever import HybridRetriever


ROOT = Path(__file__).resolve().parents[1]


def main():
    question = input("Nhap cau hoi: ").strip()
    if not question:
        raise ValueError("Question must contain text")

    # Infer scope from the original user question.
    filters = infer_filter(question)

    retriever = HybridRetriever()
    controller = GeminiController()

    settings = {
        "filters": filters,
        "max_searches": 3,
        "top_k": 12,
        "budget": 16000,
    }

    started = time.perf_counter()
    result = {}

    try:
        result = run_agent(
            question,
            controller,
            retriever,
            **settings,
        )
    except Exception as error:
        result = {
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        raise
    finally:
        trace = {
            "question": question,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": controller.model,
            "retrieval_method": "hybrid_rrf",
            "candidate_k": 12,
            "rrf_constant": 60,
            "settings": settings,
            "context_budget_unit": "characters",
            "filter_origin": "explicit tradition name in original question",
            "elapsed_seconds_excluding_setup": (
                time.perf_counter() - started
            ),
            "result": result,
            "model_calls": controller.calls,
        }

        output = ROOT / "results/agent_smoke"
        output.mkdir(parents=True, exist_ok=True)
        trace_path = output / f"{uuid4().hex}.json"
        trace_path.write_text(
            json.dumps(trace, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        controller.close()
        print("\nTrace saved:", trace_path)

    print("\nStatus:", result["status"])
    print("Answer:", result["answer"])
    print("Citations:", result["citations"])

    searches = [
        event for event in result["events"]
        if event["event"] == "search"
    ]
    print("Search calls:", len(searches))


if __name__ == "__main__":
    main()