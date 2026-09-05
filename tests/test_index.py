import unittest
from unittest.mock import Mock, patch

from rag.config import DATA_DIR
from rag.index import PROCESSORS, build_index


class IndexTest(unittest.TestCase):
    def test_upserts_corpus_when_collection_is_not_empty(self):
        collection = Mock()
        collection.count.return_value = 7
        path = DATA_DIR / "policy.md"
        processor = Mock(
            return_value=(
                ["data__policy.md:part0000"],
                ["Policy text"],
                [{"source": "data/policy.md"}],
            )
        )

        with (
            patch("rag.index.get_collection", return_value=collection),
            patch("rag.index.iter_corpus_files", return_value=[path]),
            patch.dict(PROCESSORS, {".md": processor}),
            patch("rag.index.upsert_batches") as upsert_batches,
            patch("builtins.print"),
        ):
            result = build_index()

        self.assertIs(result, collection)
        processor.assert_called_once_with(path)
        upsert_batches.assert_called_once_with(
            collection,
            ["data__policy.md:part0000"],
            ["Policy text"],
            [{"source": "data/policy.md"}],
        )


if __name__ == "__main__":
    unittest.main()
