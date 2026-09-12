import json
import sqlite3
import struct
import tempfile
import unittest
from pathlib import Path

from rag_service.relevance import (
    LexicalReranker,
    QueryTerms,
    extract_environment_facts,
    fusion_score,
    is_near_duplicate,
    low_information_reasons,
    screenshot_placeholders,
    signature_similarity,
    snippet_window,
    strip_markdown_images,
    strip_provenance_frontmatter,
    text_signature,
    tokenize_query,
)


class TokenizeTests(unittest.TestCase):
    def test_ascii_tokens_lowercased_and_kept_whole(self):
        terms = tokenize_query("Grafana CVE-2021-43798 任意文件读取")
        self.assertIn("grafana", terms.ascii_tokens)
        self.assertIn("cve-2021-43798", terms.ascii_tokens)
        self.assertEqual(terms.ascii_tokens, ["grafana", "cve-2021-43798"])

    def test_cjk_bigrams_cover_full_hanzi_run(self):
        terms = tokenize_query("任意文件读取")
        self.assertEqual(terms.cjk_bigrams, ["任意", "意文", "文件", "件读", "读取"])
        self.assertEqual(terms.total, 5)

    def test_space_separated_cjk_runs_do_not_fabricate_bigrams(self):
        # Joining the runs would yield 洞利 and 用提, which match no document
        # but still consume IDF weight and dilute real hits.
        terms = tokenize_query("漏洞 利用 提权")
        self.assertEqual(terms.cjk_bigrams, ["漏洞", "利用", "提权"])
        self.assertEqual(terms.total, 3)

    def test_single_hanzi_run_still_produces_overlapping_bigrams(self):
        self.assertEqual(tokenize_query("反序列化").cjk_bigrams, ["反序", "序列", "列化"])

    def test_blank_query_rejected(self):
        with self.assertRaises(ValueError):
            tokenize_query("   ")


class LexicalRerankerTests(unittest.TestCase):
    def _score(self, query: str, candidates: list[tuple[str, str]]) -> list[float]:
        return LexicalReranker(tokenize_query(query), candidates).scores

    def test_rare_identifier_outweighs_common_tokens(self):
        # The reported failure: three generic Chinese tokens used to outweigh
        # the one discriminating token, burying the document about rsync.
        # The pool must be corpus-scale: with two candidates every token looks
        # equally rare and IDF has nothing to measure.
        generic = [
            (f"通用提权与漏洞利用笔记 {index}", f"13_xianzhi/generic-{index}.md")
            for index in range(20)
        ]
        candidates = [
            *generic,
            ("The rsync daemon on port 873 lists modules.", "09_hacktricks/873-pentesting-rsync.md"),
        ]
        scores = self._score("rsync 漏洞 利用 提权", candidates)
        self.assertGreater(scores[-1], max(scores[:-1]))

    def test_source_path_match_outweighs_body_mention(self):
        candidates = [
            ("本文提到 rsync 一次。", "13_xianzhi/random.md"),
            ("Unrelated body text.", "09_hacktricks/873-pentesting-rsync.md"),
        ]
        scores = self._score("rsync", candidates)
        self.assertGreater(scores[1], scores[0])

    def test_ascii_token_does_not_match_inside_a_longer_word(self):
        # "rsync" is a substring of "UserSynchronization"; matching it there
        # rewarded an unrelated deserialization writeup.
        candidates = [
            ("用友 NC UserSynchronizationServlet 反序列化漏洞分析", "13_xianzhi/91316.md"),
            ("The rsync daemon lists modules.", "09_hacktricks/rsync.md"),
        ]
        scores = self._score("rsync", candidates)
        self.assertEqual(scores[0], 0.0)
        self.assertGreater(scores[1], 0.0)

    def test_token_adjacent_to_hanzi_still_matches(self):
        candidates = [("的rsync的端口", "a/b.md")]
        self.assertGreater(self._score("rsync", candidates)[0], 0.0)

    def test_no_matching_token_scores_zero(self):
        scores = self._score("sqlmap injection", [("纯中文文章内容无英文", "x/y.md")])
        self.assertEqual(scores[0], 0.0)

    def test_empty_terms_score_zero(self):
        candidates = [("a", "b.md")]
        self.assertEqual(LexicalReranker(QueryTerms(""), candidates).scores, [0.0])

    def test_scores_stay_within_unit_interval(self):
        scores = self._score(
            "rsync 漏洞", [("rsync 漏洞利用手册", "13_xianzhi/rsync.md")]
        )
        self.assertLessEqual(scores[0], 1.0)
        self.assertGreater(scores[0], 0.0)


class FusionTests(unittest.TestCase):
    def test_zero_weight_returns_dense(self):
        self.assertEqual(fusion_score(0.7, 1.0, 0.0), 0.7)

    def test_weight_blends_scores(self):
        value = fusion_score(0.6, 0.8, 0.25)
        self.assertAlmostEqual(value, 0.65, places=6)

    def test_negative_dense_clamped_to_zero(self):
        self.assertEqual(fusion_score(-0.2, 0.5, 0.5), 0.25)

    def test_default_weight_is_the_measured_optimum(self):
        # 0.35 holds golden recall at 1.00 while 0.5 degrades it; pin the knob
        # so a silent change cannot undo the tuning.
        from rag_service.config import RagConfig

        config = RagConfig(knowledge_base_root=Path("."))
        self.assertEqual(config.lexical_weight, 0.35)


    def test_tiny_size_never_yields_blank_evidence(self):
        # A collapsed window must not masquerade as an empty chunk: the caller
        # would read it as "no evidence" rather than "window too small".
        text = "\n\n\n" + "内容" * 200
        for size in (1, 2, 3, 8):
            window = snippet_window(text, "内容", size)
            self.assertTrue(window.strip("…").strip(), f"size={size} produced {window!r}")

    def test_blank_source_stays_blank(self):
        self.assertEqual(snippet_window("", "x", 10), "")
        self.assertEqual(snippet_window("   ", "x", 10), "   ")

    def test_size_zero_returns_full_text(self):
        self.assertEqual(snippet_window("abcdef", "a", 0), "abcdef")


class EnvironmentFactsTests(unittest.TestCase):
    def test_glibc_cve_and_arch_are_reported(self):
        facts = extract_environment_facts(
            "Target glibc 2.31 tcache double free on x86_64; CVE-2020-1983 applies."
        )
        self.assertEqual(facts["glibc"], ["2.31"])
        self.assertEqual(facts["cve"], ["CVE-2020-1983"])
        self.assertIn("amd64", facts["arch"])

    def test_multiple_versions_sorted_ascending(self):
        facts = extract_environment_facts("glibc 2.35 differs from glibc 2.31")
        self.assertEqual(facts["glibc"], ["2.31", "2.35"])

    def test_document_stating_no_version_yields_no_guess(self):
        # Silence must stay silence: inferring a version would fabricate exactly
        # the false confidence this fact block exists to prevent.
        self.assertEqual(extract_environment_facts("a generic exploitation note"), {})

    def test_arm64_is_not_reported_as_arm32(self):
        self.assertEqual(extract_environment_facts("build for arm64")["arch"], ["arm64"])
        self.assertEqual(extract_environment_facts("build for armv7")["arch"], ["arm"])

    def test_x86_64_not_reported_as_i386(self):
        self.assertEqual(extract_environment_facts("x86_64 binary")["arch"], ["amd64"])

    def test_fact_header_rendering(self):
        from rag_service.models import _format_facts

        rendered = _format_facts({"glibc": ["2.31"], "arch": ["amd64"], "cve": ["CVE-2020-1983"]})
        self.assertIn("glibc=2.31", rendered)
        self.assertIn("arch=amd64", rendered)
        self.assertIn("cve=CVE-2020-1983", rendered)
        self.assertEqual(_format_facts({}), "")


class ScreenshotTests(unittest.TestCase):
    def test_image_references_are_counted(self):
        text = "步骤一\n\n![Pasted image 20251104202924.png](x.png)\n\n![image.png](y.png)"
        self.assertEqual(screenshot_placeholders(text), 2)

    def test_text_without_images_counts_zero(self):
        self.assertEqual(screenshot_placeholders("纯文字步骤说明"), 0)

    def test_strip_keeps_alt_text_and_removes_the_reference(self):
        # Ordering contract: the backend counts image references before
        # stripping, because afterwards the syntax is gone and the placeholder
        # alone cannot be detected.
        stripped = strip_markdown_images("![Pasted image 20251104202924.png](x.png)")
        self.assertIn("Pasted image", stripped)
        self.assertEqual(screenshot_placeholders(stripped), 0)


class LowInformationTests(unittest.TestCase):
    def test_repetitive_test_run_flagged(self):
        reasons = low_information_reasons("test" * 130)
        self.assertIn("repetitive", reasons)

    def test_base64_blob_chunk_flagged(self):
        blob = "\n".join("QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9wcXJzdHV2d3h5ejAxMjM0NTY3ODk=" for _ in range(6))
        reasons = low_information_reasons(blob)
        self.assertIn("blob-heavy", reasons)

    def test_payload_list_with_short_lines_not_flagged(self):
        payloads = "\n".join(f"payload_{index}" for index in range(20))
        self.assertNotIn("blob-heavy", low_information_reasons(payloads))

    def test_whitespace_padded_chunk_flagged(self):
        # Seen in real CTF attachment chunks: "12321" plus ~480 spaces is five
        # characters of content, but its stripped length hid it from the
        # repetition rule.
        self.assertIn("sparse", low_information_reasons("12321" + " " * 480))

    def test_numeric_dump_flagged(self):
        dump = "{48, 83}, {26, 154}, {19, 159}, {389, 433}, {95, 440}" * 6
        self.assertIn("numeric", low_information_reasons(dump))

    def test_prose_with_numbers_not_flagged(self):
        prose = (
            "第一步把 0x48 大小的 chunk 释放两次，随后覆盖 tcache 的 next 指针。"
            "在 glibc 2.31 上 safe-linking 会校验对齐，因此要构造合法的伪造指针。"
        )
        self.assertEqual(low_information_reasons(prose), [])

    def test_short_numeric_value_not_flagged(self):
        # Two characters is not enough evidence of a dump; do not guess.
        self.assertEqual(low_information_reasons("65"), [])

    def test_code_table_with_identifiers_not_flagged(self):
        table = "offset  size  name\n0x0000  0x10  struct_id\n0x0010  0x08  next_ptr\n"
        self.assertEqual(low_information_reasons(table), [])

    def test_normal_article_not_flagged(self):
        article = "Grafana 任意文件读取漏洞分析。\n\n漏洞成因是路径拼接。参考官方通告 https://grafana.com/blog。"
        self.assertEqual(low_information_reasons(article), [])

    def test_link_heavy_section_flagged(self):
        links = "\n".join(f"- https://example.com/{index}" for index in range(12))
        reasons = low_information_reasons(links)
        self.assertIn("link-heavy", reasons)

    def test_symbol_only_chunk_flagged(self):
        reasons = low_information_reasons("----====----\n****\n")
        self.assertIn("no-text", reasons)

    def test_short_repetition_not_flagged(self):
        self.assertEqual(low_information_reasons("testtest"), [])

    def test_code_snippet_not_flagged(self):
        code = "def check(data):\n    return data.replace('a', 'b')  # 处理输入"
        self.assertEqual(low_information_reasons(code), [])


class DisplayHelperTests(unittest.TestCase):
    def test_provenance_fields_parsed_before_stripping(self):
        from rag_service.relevance import parse_provenance_frontmatter

        text = (
            "---\n"
            "source_repository: xianzhi\n"
            "source_url: local://MyDB/xianzhi\n"
            "source_commit: 4f1c2ab9d3e5\n"
            "retrieved_at: 2026-09-08\n"
            "usage: authorized-lab-ctf-defense-research-only\n"
            "---\n\n# 正文标题\n\n内容"
        )
        fields = parse_provenance_frontmatter(text)

        self.assertEqual(fields["source_repository"], "xianzhi")
        self.assertEqual(fields["source_commit"], "4f1c2ab9d3e5")
        self.assertEqual(fields["retrieved_at"], "2026-09-08")
        self.assertEqual(strip_provenance_frontmatter(text), "# 正文标题\n\n内容")

    def test_author_frontmatter_yields_no_provenance(self):
        from rag_service.relevance import parse_provenance_frontmatter

        self.assertEqual(
            parse_provenance_frontmatter("---\ntitle: 我的文章\ndate: 2024-01-01\n---\n\n正文"),
            {},
        )

    def test_body_mentioning_provenance_keys_is_not_frontmatter(self):
        from rag_service.relevance import parse_provenance_frontmatter

        text = "# 导入说明\n\nsource_repository: 只是正文里的示例\n"
        self.assertEqual(parse_provenance_frontmatter(text), {})

    def test_provenance_frontmatter_stripped_from_chunk(self):
        text = (
            "---\n"
            "source_repository: xianzhi\n"
            "source_url: local://MyDB/xianzhi\n"
            "retrieved_at: 2026-09-08\n"
            "usage: authorized-lab-ctf-defense-research-only\n"
            "notice: Imported text is for authorized systems.\n"
            "---\n\n"
            "# 正文标题\n\n内容"
        )
        cleaned = strip_provenance_frontmatter(text)
        self.assertNotIn("source_repository", cleaned)
        self.assertTrue(cleaned.startswith("# 正文标题"))

    def test_author_frontmatter_without_provenance_kept(self):
        text = "---\ntitle: 我的文章\ndate: 2024-01-01\n---\n\n正文内容"
        self.assertEqual(strip_provenance_frontmatter(text), text)

    def test_no_frontmatter_untouched(self):
        text = "# 普通文章\n\n正文"
        self.assertEqual(strip_provenance_frontmatter(text), text)

    def test_markdown_images_removed_alt_kept(self):
        text = "看这张图 ![POC 截图](https://cdn.example.com/a.png) 再读正文。"
        cleaned = strip_markdown_images(text)
        self.assertIn("POC 截图", cleaned)
        self.assertNotIn("https://cdn.example.com", cleaned)

    def test_empty_alt_image_removed(self):
        self.assertEqual(strip_markdown_images("![](https://cdn.example.com/x.png)"), "")
        self.assertEqual(strip_markdown_images("no image"), "no image")

    def test_snippet_window_contains_query_hit(self):
        text = ("开始 " * 40) + "JNDI 注入利用演示" + (" 结束" * 40)
        window = snippet_window(text, "JNDI", 300)
        self.assertIn("JNDI", window)
        self.assertLessEqual(len(window), 300 + 80 + 2)

    def test_snippet_window_without_hit_starts_at_top(self):
        text = "ABCDEFGH " * 60
        window = snippet_window(text, "不存在词xyz", 200)
        self.assertTrue(window.startswith("ABCDEFGH"))
        self.assertIn("…", window)

    def test_short_text_not_truncated(self):
        text = "short body"
        self.assertEqual(snippet_window(text, "short", 200), text)

    def test_truncation_marks_line_aligned(self):
        text = "\n".join(f"line-{index}-" + "填充" * 30 for index in range(40))
        window = snippet_window(text, "line-25", 300)
        self.assertIn("line-25", window)
        self.assertTrue(window.startswith("…"))


class SignatureTests(unittest.TestCase):
    # A realistic article body: varied sentences produce a wide gram set, as in
    # the indexed corpus. Highly repetitive fixtures (one sentence × 40) have
    # too few unique grams for any Jaccard estimate to call a header-prefixed
    # copy a near-duplicate, so they cannot express the mirror contract.
    ARTICLE = (
        "# Laravel 反序列化漏洞分析\n\n"
        "## 环境\nLaravel 8.83.27 / PHP 7.4.33，phar 只读。\n\n"
        "## 入口\n触发点是 PendingBroadcast 的 __destruct，它调用 event 属性上的 dispatch；"
        "把该属性设为 Faker\\Generator 后即可进入 __call。\n\n"
        "## 链\n1. Generator::__call -> call_user_func_array\n"
        "2. call_user_func 第一元素为对象时进入其 __call\n3. Validator::__call 调用 extend\n\n"
        "## 验证\n构造 phar:// 前缀后由 file_exists 触发反序列化，观察日志确认执行。\n"
    )

    def test_identical_text_similar(self):
        self.assertEqual(
            signature_similarity(text_signature(self.ARTICLE), text_signature(self.ARTICLE)),
            1.0,
        )

    def test_mirrored_copies_near_duplicate(self):
        mirror = "本文首发于安全社区\n" + self.ARTICLE + "\n转载请注明来源\n"
        self.assertTrue(is_near_duplicate(text_signature(self.ARTICLE), text_signature(mirror)))

    def test_distinct_articles_not_confused(self):
        left = "Java 反序列化 ysoserial gadget 分析 " * 30
        right = "SQL 注入绕过 WAF 技巧总结 " * 30
        self.assertFalse(is_near_duplicate(text_signature(left), text_signature(right)))

    def test_short_text_signature_all_zeros(self):
        self.assertEqual(text_signature("短"), [0] * 12)

    def test_mirror_filter_rejects_second_source(self):
        # emulate collection order: two copies of the same article
        kept = text_signature(self.ARTICLE)
        copy = text_signature("导读行\n" + self.ARTICLE)
        second = text_signature("SQL 注入完全不同的文章 " * 40)
        self.assertTrue(is_near_duplicate(kept, copy))
        self.assertFalse(is_near_duplicate(kept, second))


if __name__ == "__main__":
    unittest.main()
