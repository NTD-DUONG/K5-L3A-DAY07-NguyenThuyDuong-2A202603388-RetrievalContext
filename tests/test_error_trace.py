import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from starter.agentic import run_agent
from src.gemini_controller import GeminiController


class ErrorTraceTests(unittest.TestCase):
    def test_model_failure_preserves_previous_search_and_context(self):
        chunk = {
            "id": "chunk-1", "text": "Evidence", "source": "source",
            "page": "page-1", "source_id": "document-1", "metadata": {},
        }
        controller = Mock()
        controller.decide.side_effect = [
            {"action": "search", "query": "query"}, RuntimeError("API unavailable"),
        ]
        retriever = Mock()
        retriever.search.return_value = [chunk]
        events = []

        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            run_agent("question", controller, retriever, events=events)

        self.assertEqual([event["event"] for event in events],
                         ["search", "controller_error"])
        self.assertEqual(events[0]["context_ids"], ["chunk-1"])
        self.assertIn("Evidence", events[0]["context"])

    def test_failed_search_records_query_and_parameters(self):
        controller = Mock()
        controller.decide.return_value = {"action": "search", "query": "query"}
        retriever = Mock()
        retriever.search.side_effect = RuntimeError("Store unavailable")
        events = []
        filters = {"painting_tradition": "dong_ho"}

        with self.assertRaisesRegex(RuntimeError, "Store unavailable"):
            run_agent("question", controller, retriever,
                      filters=filters, top_k=12, events=events)

        self.assertEqual(events[0]["event"], "search_error")
        self.assertEqual(events[0]["query"], "query")
        self.assertEqual(events[0]["filter"], filters)
        self.assertEqual(events[0]["top_k"], 12)

    def test_failed_gemini_call_is_recorded_without_fake_usage(self):
        controller = GeminiController.__new__(GeminiController)
        controller.model = "test-model"
        controller.instructions = "test instructions"
        controller.calls = []
        generate = Mock(side_effect=RuntimeError("API unavailable"))
        controller.client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate)
        )

        with self.assertRaisesRegex(RuntimeError, "API unavailable"):
            controller.decide("question", [])

        self.assertEqual(len(controller.calls), 1)
        call = controller.calls[0]
        self.assertEqual(call["status"], "error")
        self.assertEqual(call["request"]["question"], "question")
        self.assertIsNone(call["usage"])
        self.assertIsNone(call["raw_response"])
        self.assertGreaterEqual(call["elapsed_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
