"""The improvements added from review feedback.

Two of the reviewed suggestions rested on premises the measurements did not
support, so the tests here also pin what was *rejected*, to stop it being
re-introduced:

- a top1/top2 score margin as a confidence signal: measured on the real corpus a
  correct hit scored a 0.0011 margin (several distinct documents cover the
  topic) while off-corpus queries sat at 0.0039-0.0057, so the margin is
  anti-correlated with what it was meant to indicate. What does discriminate is
  lexical support: off-corpus probes all sat at <= 0.43 while relevant hits
  ranged 0.16-1.00.
- collapsing near-duplicate results: the top hits for a topic are usually
  *different* documents (verified: not near-duplicates), so collapsing them would
  remove genuine coverage rather than redundancy.

The extract mode and image references come from the same review.
"""
import unittest

from rag_service.models import RetrievalRequest, SearchResult
from rag_service.relevance import extract_segments, image_references
from rag_service.service import RagService


class FakeBackend:
    """Returns pre-built results, so these tests need no index or provider."""

    name = "fake"

    def __init__(self, results):
        self._results = results

    def search(self, request):
        return list(self._results)

    def close(self):
        pass


def _config(**overrides):
    from pathlib import Path

    from rag_service.config import RagConfig

    values = {
        "knowledge_base_root": Path("."),
        "allowed_knowledge_bases": frozenset({"cybersec"}),
        "embedding_model": "bge-m3",
    }
    values.update(overrides)
    return RagConfig(**values)


def _service(results, **overrides):
    return RagService(_config(**overrides), FakeBackend(results))


class ExtractModeTests(unittest.TestCase):
    def test_code_mode_returns_every_fence_with_its_language(self):
        text = "intro\n\n```bash\ncurl x\n```\n\nprose\n\n```\nplain block\n```\n"
        segments = extract_segments(text, "code")
        self.assertEqual([s["language"] for s in segments], ["bash", None])
        self.assertEqual(segments[0]["text"], "curl x")

    def test_payload_mode_drops_blocks_that_are_not_runnable(self):
        text = "```\njust prose in a fence\n```\n\n```python\nimport os\n```\n"
        self.assertEqual(len(extract_segments(text, "code")), 2)
        payload = extract_segments(text, "payload")
        self.assertEqual([s["language"] for s in payload], ["python"])

    def test_an_unlabelled_command_block_still_counts_as_payload(self):
        # Writeups routinely omit the language on the one block that matters.
        segments = extract_segments("```\n$ nmap -sV 10.0.0.1\n```", "payload")
        self.assertEqual(len(segments), 1)

    def test_a_document_without_fences_returns_nothing_rather_than_guessing(self):
        self.assertEqual(extract_segments("no code here at all", "code"), [])
        self.assertEqual(extract_segments("", "payload"), [])

    def test_segments_reach_the_tool_text_ahead_of_the_prose(self):
        result = SearchResult(
            content="prose\n\n```bash\nnc -vn 127.0.0.1 873\n```",
            score=0.7,
            source="a/b.md",
            chunk_id="cybersec:1",
            metadata={
                "lexical_score": 0.8,
                "segment_kind": "payload",
                "segments": [{"language": "bash", "text": "nc -vn 127.0.0.1 873"}],
            },
        )
        text = _service([result]).search(
            RetrievalRequest(query="rsync", knowledge_base="cybersec", extract="payload")
        ).as_tool_text()
        self.assertIn("EXTRACTED SEGMENTS (1)", text)
        self.assertIn("[bash]", text)
        self.assertIn("nc -vn 127.0.0.1 873", text)
        # The surrounding prose is still there, so the citation stays checkable.
        self.assertIn("surrounding content", text)

    def test_a_document_with_no_segment_says_so(self):
        result = SearchResult(
            content="prose only",
            score=0.7,
            source="a/b.md",
            chunk_id="cybersec:1",
            metadata={"lexical_score": 0.8, "segment_kind": "code", "segments": []},
        )
        text = _service([result]).search(
            RetrievalRequest(query="x", knowledge_base="cybersec", extract="code")
        ).as_tool_text()
        self.assertIn("no matching segment", text)

    def test_an_unknown_mode_is_rejected_rather_than_ignored(self):
        # Silently behaving like None would return windowed excerpts that the
        # caller reads as "this document contains no code".
        with self.assertRaises(ValueError) as ctx:
            RetrievalRequest(query="x", extract="exp")
        self.assertIn("extract", str(ctx.exception))

    def test_the_mode_is_normalised(self):
        self.assertEqual(RetrievalRequest(query="x", extract="  CODE ").extract, "code")
        self.assertIsNone(RetrievalRequest(query="x", extract="   ").extract)


class LineEndingTests(unittest.TestCase):
    """CRLF is a real trigger, not a hypothetical.

    This repository's own importer writes CRLF on Windows (measured: 3000 of 3001
    source documents), so a rebuild after a Windows import yields CRLF content.
    The fence pattern's `[ \\t]*$` never matched before a trailing `\\r`, so every
    document extracted zero segments and the caller read that as "no code here".
    """

    def test_a_crlf_document_still_yields_its_segments(self):
        text = "# head\r\n\r\n```bash\r\ncurl -s http://x/\r\n```\r\n"
        segments = extract_segments(text, "code")
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["language"], "bash")
        self.assertEqual(segments[0]["text"], "curl -s http://x/")
        self.assertNotIn("\r", segments[0]["text"])

    def test_crlf_and_lf_documents_extract_identically(self):
        body = "```python\nimport os\n```"
        lf = extract_segments(body + "\n", "code")
        crlf = extract_segments(body.replace("\n", "\r\n") + "\r\n", "code")
        self.assertEqual(lf, crlf)

    def test_payload_mode_works_on_crlf_too(self):
        text = "```\r\nprose in a fence\r\n```\r\n\r\n```bash\r\nnmap -sV x\r\n```\r\n"
        self.assertEqual([s["language"] for s in extract_segments(text, "payload")], ["bash"])


class ImageTargetSafetyTests(unittest.TestCase):
    """A corpus document is untrusted input, so its "image" address is too."""

    def test_script_and_data_urls_are_not_returned_as_images(self):
        for payload in ("![x](javascript:alert(1))", "![x](data:text/html,<b>x</b>)",
                        "<img src=\"javascript:alert(1)\">", "![x](vbscript:msgbox)"):
            self.assertEqual(
                image_references(payload), [],
                f"{payload!r} must not be handed back as an image address",
            )

    def test_fetchable_targets_survive(self):
        self.assertEqual(
            [r["src"] for r in image_references("![a](images/x.png) ![b](https://h.test/y.png)")],
            ["images/x.png", "https://h.test/y.png"],
        )

    def test_malformed_markdown_yields_nothing_rather_than_a_bogus_path(self):
        # The earlier parse produced values like "1" for unbalanced parentheses,
        # which is worse than nothing because it looks like a usable address.
        for broken in ("![unclosed](x", "<img>", "![]()", "![](   )"):
            self.assertEqual(image_references(broken), [], broken)


class ImageReferenceTests(unittest.TestCase):
    def test_markdown_and_html_images_both_yield_addresses(self):
        text = "![cracked key](images/a.png) and <img src='images/b.jpg'>"
        refs = image_references(text)
        self.assertEqual([r["src"] for r in refs], ["images/a.png", "images/b.jpg"])
        self.assertEqual(refs[0]["alt"], "cracked key")

    def test_a_markdown_title_is_not_taken_for_part_of_the_url(self):
        refs = image_references('![x](images/a.png "screenshot")')
        self.assertEqual(refs[0]["src"], "images/a.png")

    def test_duplicates_are_reported_once(self):
        refs = image_references("![](a.png) ![](a.png)")
        self.assertEqual(len(refs), 1)

    def test_addresses_reach_the_signal_line(self):
        result = SearchResult(
            content="step\n\n![key](images/k.png)",
            score=0.8,
            source="a/b.md",
            chunk_id="cybersec:2",
            metadata={
                "lexical_score": 0.9,
                "has_screenshots": 1,
                "image_refs": [{"src": "images/k.png", "alt": "key"}],
            },
        )
        text = _service([result]).search(
            RetrievalRequest(query="key", knowledge_base="cybersec")
        ).as_tool_text()
        self.assertIn("shots=1", text)
        self.assertIn("images=images/k.png", text)


class ConfidenceTests(unittest.TestCase):
    """Grade by the evidence behind the hit, not by the raw score."""

    def _top(self, content="doc text", score=0.7, lexical=0.8):
        metadata = {} if lexical is None else {"lexical_score": lexical}
        return [SearchResult(content=content, score=score, source="a.md",
                             chunk_id="cybersec:1", metadata=metadata)]

    def test_an_identifier_from_the_query_appearing_in_the_document_is_anchored(self):
        response = _service(self._top(content="affected CVE-2021-43798 builds")).search(
            RetrievalRequest(query="CVE-2021-43798 Grafana", knowledge_base="cybersec")
        )
        self.assertEqual(response.confidence, "anchored")
        self.assertEqual([w for w in response.warnings if "meaning alone" in w], [])

    def test_strong_word_support_is_graded_lexical(self):
        response = _service(self._top(content="unrelated wording", lexical=0.7)).search(
            RetrievalRequest(query="rsync daemon", knowledge_base="cybersec")
        )
        self.assertEqual(response.confidence, "lexical")

    def test_no_word_support_above_the_threshold_is_reported_as_semantic(self):
        response = _service(self._top(content="unrelated wording", score=0.62, lexical=0.2)).search(
            RetrievalRequest(query="rsync daemon", knowledge_base="cybersec")
        )
        self.assertEqual(response.confidence, "semantic")
        self.assertTrue(any("meaning alone" in w for w in response.warnings))

    def test_a_genuinely_low_score_reports_the_low_score_not_the_grading(self):
        # Both messages would be noise; the low score is the actionable one.
        response = _service(self._top(content="unrelated", score=0.12, lexical=0.05)).search(
            RetrievalRequest(query="rsync", knowledge_base="cybersec")
        )
        self.assertTrue(any("score is low" in w for w in response.warnings))
        self.assertEqual([w for w in response.warnings if "meaning alone" in w], [])

    def test_a_browse_listing_is_not_graded(self):
        # Navigation, not a match: there is no query to compare against.
        response = _service(self._top(content="listing", lexical=None)).search(
            RetrievalRequest(query="", knowledge_base="cybersec")
        )
        self.assertIsNone(response.confidence)
        self.assertEqual([w for w in response.warnings if "meaning alone" in w], [])

    def test_a_backend_without_a_lexical_component_is_not_graded(self):
        # Dense-only mode (or a third-party backend) leaves the field absent;
        # that is not evidence the match was semantic.
        response = _service(self._top(content="anything", lexical=None)).search(
            RetrievalRequest(query="rsync", knowledge_base="cybersec")
        )
        self.assertIsNone(response.confidence)
        self.assertEqual([w for w in response.warnings if "meaning alone" in w], [])

    def test_an_empty_result_set_is_not_graded(self):
        response = _service([]).search(
            RetrievalRequest(query="nothing matches", knowledge_base="cybersec")
        )
        self.assertIsNone(response.confidence)
        self.assertTrue(response.no_match)


    def test_the_grade_rides_on_the_first_hit_only(self):
        results = [
            SearchResult(content="affected CVE-2021-43798", score=0.9, source="a.md",
                         chunk_id="c:1", metadata={"lexical_score": 0.9}),
            SearchResult(content="second", score=0.5, source="b.md",
                         chunk_id="c:2", metadata={"lexical_score": 0.5}),
        ]
        text = _service(results).search(
            RetrievalRequest(query="CVE-2021-43798", knowledge_base="cybersec")
        ).as_tool_text()
        self.assertIn("conf=anchored", text)
        self.assertEqual(text.count("conf="), 1, "the grade describes the top hit only")


class RejectedSuggestionTests(unittest.TestCase):
    """Pin the measurements that argue against two of the reviewed ideas."""

    def test_a_tiny_margin_with_solid_evidence_outranks_a_huge_margin_without(self):
        # A margin-based grader would call the second case the confident one.
        # Measured reality is the opposite: near-equal scores here mean several
        # distinct documents cover the topic, and off-corpus hits are what share
        # no terms. This pins the ordering a margin rule would invert.
        close_pair = _service(
            [
                SearchResult(content="rsync daemon docs", score=0.70, source="a.md",
                             chunk_id="c:1", metadata={"lexical_score": 0.80}),
                SearchResult(content="rsync other docs", score=0.699, source="b.md",
                             chunk_id="c:2", metadata={"lexical_score": 0.79}),
            ]
        ).search(RetrievalRequest(query="rsync daemon", knowledge_base="cybersec"))
        wide_pair = _service(
            [
                SearchResult(content="unrelated", score=0.62, source="a.md",
                             chunk_id="c:1", metadata={"lexical_score": 0.10}),
                SearchResult(content="also unrelated", score=0.20, source="b.md",
                             chunk_id="c:2", metadata={"lexical_score": 0.02}),
            ]
        ).search(RetrievalRequest(query="rsync daemon", knowledge_base="cybersec"))

        self.assertEqual(close_pair.confidence, "lexical")
        self.assertEqual(wide_pair.confidence, "semantic")

    def test_confidence_uses_lexical_support_not_the_fused_score_alone(self):
        # Same fused score, different evidence -> different grade.
        weak = _service(
            [
                SearchResult(content="unrelated", score=0.60, source="a.md",
                             chunk_id="c:1", metadata={"lexical_score": 0.2})
            ]
        ).search(RetrievalRequest(query="rsync", knowledge_base="cybersec"))
        strong = _service(
            [
                SearchResult(content="unrelated", score=0.60, source="a.md",
                             chunk_id="c:1", metadata={"lexical_score": 0.8})
            ]
        ).search(RetrievalRequest(query="rsync", knowledge_base="cybersec"))
        self.assertEqual(weak.confidence, "semantic")
        self.assertEqual(strong.confidence, "lexical")


if __name__ == "__main__":
    unittest.main()
