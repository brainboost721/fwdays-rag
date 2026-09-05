import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from rag.router import (
    _parse_router_sources,
    choose_sources_for_query,
    list_indexed_sources,
    where_from_sources,
)


def completion(content: str):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class RouterTest(unittest.TestCase):
    def test_lists_unique_indexed_sources_in_stable_order(self):
        collection = Mock()
        collection.get.return_value = {
            "metadatas": [
                {"source": "data/tickets.csv"},
                {"source": "data/policies/security_policy.md"},
                {"source": "data/tickets.csv"},
                {},
            ]
        }

        with patch("rag.router.get_collection", return_value=collection):
            sources = list_indexed_sources()

        self.assertEqual(
            sources,
            ["data/policies/security_policy.md", "data/tickets.csv"],
        )
        collection.get.assert_called_once_with(include=["metadatas"])

    def test_parses_fenced_numbered_sources_and_removes_duplicates(self):
        allowed = ["data/a.md", "data/b.pdf"]
        raw = "```text\n1. data/a.md\n2. data/b.pdf\n3. data/a.md\n```"

        self.assertEqual(
            _parse_router_sources(raw, allowed),
            ["data/a.md", "data/b.pdf"],
        )

    def test_unknown_source_falls_back_to_unfiltered_search(self):
        self.assertIsNone(
            _parse_router_sources(
                "data/tickets.csv\ndata/invented.csv",
                ["data/tickets.csv"],
            )
        )

    def test_null_means_unfiltered_search(self):
        self.assertIsNone(_parse_router_sources("null", ["data/tickets.csv"]))

    def test_builds_single_and_multiple_source_filters(self):
        self.assertIsNone(where_from_sources(None))
        self.assertEqual(
            where_from_sources(["data/tickets.csv"]),
            {"source": "data/tickets.csv"},
        )
        self.assertEqual(
            where_from_sources(["data/a.pdf", "data/b.pdf", "data/a.pdf"]),
            {"source": {"$in": ["data/a.pdf", "data/b.pdf"]}},
        )

    def test_chooses_only_allowed_sources_with_system_constraints(self):
        client = Mock()
        client.chat.completions.create.return_value = completion(
            "data/tickets.csv"
        )

        with patch("rag.router.get_client", return_value=client):
            sources = choose_sources_for_query(
                "Що зробили по заявці T-1042?",
                ["data/tickets.csv", "data/services.csv"],
            )

        self.assertEqual(sources, ["data/tickets.csv"])
        kwargs = client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["temperature"], 0)
        self.assertEqual(
            [message["role"] for message in kwargs["messages"]],
            ["system", "user"],
        )
        self.assertIn("data/services.csv", kwargs["messages"][1]["content"])
        self.assertIn("T-1042", kwargs["messages"][1]["content"])

    def test_skips_llm_when_index_has_no_sources(self):
        client = Mock()

        with patch("rag.router.get_client", return_value=client):
            sources = choose_sources_for_query("Question", [])

        self.assertIsNone(sources)
        client.chat.completions.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()
