import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag.answer import NO_DATA, rag_answer, rag_answer_with_router


def completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class AnswerTest(unittest.TestCase):
    def setUp(self):
        self.hits = [
            {
                "document": "Заявку закрито.",
                "source": "data/tickets.csv",
                "distance": 0.2,
            }
        ]

    def run_answer(self, outputs: list[str]) -> tuple[str, Mock]:
        client = Mock()
        client.chat.completions.create.side_effect = [
            completion(output) for output in outputs
        ]
        with (
            patch("rag.answer.retrieve", return_value=self.hits),
            patch("rag.answer.get_client", return_value=client),
        ):
            answer = rag_answer("Що зробили по заявці?")
        return answer, client

    def test_accepts_answer_with_allowed_source(self):
        answer, client = self.run_answer(
            ["Заявку закрито.\n\nДжерела:\n- [source=data/tickets.csv]"]
        )

        self.assertIn("[source=data/tickets.csv]", answer)
        self.assertEqual(client.chat.completions.create.call_count, 1)

        messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Спирайся ЛИШЕ", messages[0]["content"])
        self.assertNotIn("Що зробили по заявці?", messages[0]["content"])
        self.assertIn("Що зробили по заявці?", messages[1]["content"])

    def test_retries_answer_without_sources(self):
        answer, client = self.run_answer(
            [
                "Заявку закрито.",
                "Заявку закрито.\n\nДжерела:\n- [source=data/tickets.csv]",
            ]
        )

        self.assertIn("[source=data/tickets.csv]", answer)
        self.assertEqual(client.chat.completions.create.call_count, 2)
        retry_messages = client.chat.completions.create.call_args.kwargs["messages"]
        self.assertEqual(
            [message["role"] for message in retry_messages],
            ["system", "user", "assistant", "user"],
        )

    def test_refuses_when_retry_still_has_no_valid_source(self):
        answer, client = self.run_answer(
            [
                "Заявку закрито.\n\nДжерела:\n- [source=data/invented.csv]",
                "Заявку закрито без джерел.",
            ]
        )

        self.assertEqual(answer, NO_DATA)
        self.assertEqual(client.chat.completions.create.call_count, 2)

    def test_normalizes_refusal_without_retry(self):
        answer, client = self.run_answer(
            [f"«{NO_DATA}»\n\nДжерела:\n- [source=data/tickets.csv]"]
        )

        self.assertEqual(answer, NO_DATA)
        self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_routed_answer_passes_source_filter_to_retrieval_pipeline(self):
        with (
            patch(
                "rag.answer.choose_sources_for_query",
                return_value=["data/tickets.csv"],
            ),
            patch("rag.answer.rag_answer", return_value="Answer") as base_answer,
        ):
            answer = rag_answer_with_router("Question", k=3)

        self.assertEqual(answer, "Answer")
        base_answer.assert_called_once_with(
            "Question", k=3, where={"source": "data/tickets.csv"}
        )


if __name__ == "__main__":
    unittest.main()
