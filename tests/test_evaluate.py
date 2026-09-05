import unittest
from unittest.mock import Mock

from rag.answer import NO_DATA
from rag.evaluate import (
    GoldenCase,
    evaluate_case,
    extract_sources,
    load_golden_set,
)


class EvaluateTest(unittest.TestCase):
    def test_loads_project_golden_set(self):
        cases = load_golden_set()

        self.assertEqual(len(cases), 3)
        self.assertEqual(len({case.id for case in cases}), 3)
        self.assertTrue(any(case.expect_refusal for case in cases))

    def test_extracts_unique_sources(self):
        answer = """Джерела:
- [source=data/tickets.csv]
- [source=data/tickets.csv]
- [source=data/policies/security_policy.md]"""

        self.assertEqual(
            extract_sources(answer),
            {"data/tickets.csv", "data/policies/security_policy.md"},
        )

    def test_positive_case_requires_answer_and_exact_sources(self):
        case = GoldenCase(
            id="E01",
            question="Question",
            expected_sources=("data/tickets.csv",),
            final_answer="Reference",
            expect_refusal=False,
        )
        rag_fn = Mock(
            return_value="Answer\n\nДжерела:\n- [source=data/tickets.csv]"
        )
        judge_fn = Mock(return_value=True)

        result = evaluate_case(case, rag_fn=rag_fn, judge_fn=judge_fn, k=4)

        self.assertTrue(result.passed)
        self.assertEqual(result.source_recall, 1.0)
        self.assertEqual(result.source_precision, 1.0)
        rag_fn.assert_called_once_with("Question", k=4)
        judge_fn.assert_called_once_with("Question", "Reference", result.actual)

    def test_unexpected_source_fails_positive_case(self):
        case = GoldenCase(
            id="E01",
            question="Question",
            expected_sources=("data/tickets.csv",),
            final_answer="Reference",
            expect_refusal=False,
        )
        rag_fn = Mock(
            return_value=(
                "Answer\n\nДжерела:\n"
                "- [source=data/tickets.csv]\n"
                "- [source=data/services.csv]"
            )
        )

        result = evaluate_case(
            case, rag_fn=rag_fn, judge_fn=Mock(return_value=True)
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.source_recall, 1.0)
        self.assertEqual(result.source_precision, 0.5)

    def test_negative_case_is_deterministic(self):
        case = GoldenCase(
            id="E03",
            question="Unknown question",
            expected_sources=(),
            final_answer=NO_DATA,
            expect_refusal=True,
        )
        judge_fn = Mock()

        result = evaluate_case(
            case, rag_fn=Mock(return_value=NO_DATA), judge_fn=judge_fn
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.source_recall, 1.0)
        self.assertEqual(result.source_precision, 1.0)
        judge_fn.assert_not_called()


if __name__ == "__main__":
    unittest.main()
