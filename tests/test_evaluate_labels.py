"""Golden-query labelling: positives, out-of-corpus negatives, mismatch probes.

``negative`` records must return zero results; ``mismatch`` records describe
queries whose only corpus neighbours are version/architecture mismatches, so a
returned neighbour is informative evidence rather than a leak.
"""
import json
import tempfile
import unittest
from pathlib import Path

from rag_service.evaluate import load_queries

GOLDEN_PATH = Path(__file__).resolve().parent / "data" / "retrieval_queries.jsonl"


class LoadQueriesTests(unittest.TestCase):
    def _load(self, records: list[dict]) -> list[dict]:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "queries.jsonl"
            path.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )
            return load_queries(path)

    def test_prefixes_imply_positive_label(self):
        queries = self._load(
            [{"query": "JNDI Log4Shell", "prefixes": [" 09_hacktricks/ "]}]
        )
        self.assertEqual(queries[0]["label"], "positive")
        self.assertEqual(queries[0]["prefixes"], ["09_hacktricks/"])

    def test_legacy_empty_flag_means_negative(self):
        queries = self._load([{"query": "best pizza topping", "empty": True}])
        self.assertEqual(queries[0]["label"], "negative")
        self.assertEqual(queries[0]["prefixes"], [])

    def test_explicit_mismatch_label_keeps_no_prefixes(self):
        queries = self._load(
            [{"query": "glibc 2.32 safe-linking exploit", "label": "mismatch"}]
        )
        self.assertEqual(queries[0]["label"], "mismatch")
        self.assertEqual(queries[0]["prefixes"], [])

    def test_positive_without_ground_truth_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "prefixes' or 'sources"):
            self._load([{"query": "x", "label": "positive"}])

    def test_positive_accepts_exact_sources(self):
        queries = self._load(
            [
                {
                    "query": "JNDI Log4Shell",
                    "sources": ["13_xianzhi/10102-Apache Log4j2 JNDI RCE.md"],
                }
            ]
        )
        self.assertEqual(queries[0]["label"], "positive")
        self.assertEqual(
            queries[0]["sources"], ["13_xianzhi/10102-Apache Log4j2 JNDI RCE.md"]
        )
        self.assertEqual(queries[0]["prefixes"], [])

    def test_negative_with_prefixes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not declare"):
            self._load(
                [{"query": "x", "label": "negative", "prefixes": ["14_ctf_wp/"]}]
            )

    def test_mismatch_with_sources_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "must not declare"):
            self._load(
                [{"query": "x", "label": "mismatch", "sources": ["14_ctf_wp/a.md"]}]
            )

    def test_unknown_label_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "label"):
            self._load([{"query": "x", "label": "maybe"}])

    def test_golden_set_labels(self):
        queries = load_queries(GOLDEN_PATH)
        labels = [item["label"] for item in queries]
        self.assertEqual(labels.count("positive"), 16)
        self.assertEqual(labels.count("negative"), 1)
        self.assertEqual(labels.count("mismatch"), 1)
        self.assertTrue(all(item["query"] for item in queries))
        self.assertTrue(
            all(item["prefixes"] or item["sources"] for item in queries if item["label"] == "positive")
        )

    def test_relevance_matches_prefixes_and_exact_sources(self):
        from rag_service.evaluate import is_relevant

        by_prefix = {"prefixes": ["15_butian/"], "sources": []}
        self.assertTrue(is_relevant("15_butian/1003-Grafana.md", by_prefix))
        self.assertFalse(is_relevant("13_xianzhi/other.md", by_prefix))

        by_source = {
            "prefixes": [],
            "sources": ["13_xianzhi/10102-Apache Log4j2 JNDI RCE.md"],
        }
        self.assertTrue(is_relevant("13_xianzhi/10102-Apache Log4j2 JNDI RCE.md", by_source))
        self.assertFalse(is_relevant("13_xianzhi/9999-other-log4j.md", by_source))
        self.assertFalse(is_relevant(None, by_source))

    def test_windows_separators_normalize_before_matching(self):
        from rag_service.evaluate import is_relevant

        item = {"prefixes": ["15_butian/"], "sources": []}
        self.assertTrue(is_relevant("15_butian\\1003-Grafana.md", item))


if __name__ == "__main__":
    unittest.main()
