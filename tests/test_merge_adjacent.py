import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rag_service.backends.faiss import FaissBackend, _join_merged_texts, _merge_adjacent_rows

_U64 = struct.Struct("<Q")


def _make_store(path: Path):
    payloads = []
    specs = [
        ("source-a", "a-0"),
        ("source-a", "a-1"),
        ("source-a", "a-2"),
        ("source-b", "b-0"),
        ("source-b", "b-1"),
        ("source-c", "c-0"),
    ]
    for source, text in specs:
        payload = json.dumps(
            {"text": text, "metadata": {"source": source}}, ensure_ascii=False
        ).encode("utf-8") + b"\n"
        payloads.append(payload)
    docs = path / "docs.cos.jsonl"
    docs.write_bytes(b"".join(payloads))
    offsets = []
    cursor = 0
    for payload in payloads:
        offsets.append(cursor)
        cursor += len(payload)
    return {
        "docs_path": docs,
        "offsets": np.asarray(offsets, dtype="<u8"),
        "documents": {},
    }, [text for _, text in specs]


class MergeAdjacentTests(unittest.TestCase):
    def test_merges_same_source_neighbors_within_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store, texts = _make_store(Path(temp_dir))
            merged, count, rows = _merge_adjacent_rows(store, 1, "source-a", 2, texts[1])
            self.assertEqual(count, 3)
            self.assertEqual(rows, [0, 1, 2])
            self.assertEqual(merged, "\n\n".join(texts[:3]))

    def test_stops_at_source_boundary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store, texts = _make_store(Path(temp_dir))
            merged, count, rows = _merge_adjacent_rows(store, 3, "source-b", 2, texts[3])
            self.assertEqual(count, 2)
            self.assertEqual(rows, [3, 4])
            self.assertEqual(merged, "\n\n".join(texts[3:5]))

    def test_single_chunk_when_no_neighbors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store, texts = _make_store(Path(temp_dir))
            merged, count, rows = _merge_adjacent_rows(store, 5, "source-c", 2, texts[5])
            self.assertEqual(count, 1)
            self.assertEqual(rows, [5])
            self.assertEqual(merged, texts[5])


class JoinMergedTests(unittest.TestCase):
    def test_exact_100_char_overlap_trimmed(self):
        tail = "结尾段落" * 20  # 100 chars
        left = "开头正文" + tail
        right = tail + "后续内容"
        joined = _join_merged_texts([left, right])
        self.assertEqual(joined, "开头正文" + tail + "\n\n" + "后续内容")

    def test_no_overlap_joins_with_separator(self):
        joined = _join_merged_texts(["first chunk", "second chunk"])
        self.assertEqual(joined, "first chunk\n\nsecond chunk")

    def test_short_overlap_kept_intact(self):
        left = "标题相同但无重叠的前文"
        right = "标题相同但无重叠的后文"
        joined = _join_merged_texts([left, right])
        self.assertEqual(joined, left + "\n\n" + right)

    def test_multi_chunk_chain_trimmed_once(self):
        tail = "A" * 100
        chunks = ["x" + tail, tail + "y" + tail, tail + "z"]
        joined = _join_merged_texts(chunks)
        # each shared overlap is kept once: tails of chunk0 and chunk1 remain
        self.assertEqual(joined.count("A"), 200)
        self.assertEqual(joined.count("\n\n"), 2)
        self.assertEqual(joined, "x" + tail + "\n\ny" + tail + "\n\nz")

    def test_identical_two_line_run_at_seam_trimmed(self):
        left = "上文内容\n我们可以使用soapui对这类api进行测试\n## WADL"
        right = "我们可以使用soapui对这类api进行测试\n## WADL\n文件里面有很明显的wadl标志\n再后文"
        joined = _join_merged_texts([left, right])
        self.assertEqual(joined.count("我们可以使用soapui对这类api进行测试"), 1)
        self.assertEqual(joined.count("## WADL"), 1)
        self.assertIn("再后文", joined)

    def test_bridge_chunk_keeps_only_new_lines(self):
        left = "正文A\n我们可以使用soapui对这类api进行测试\n## WADL"
        bridge = "我们可以使用soapui对这类api进行测试\n## WADL\n文件里面有很明显的wadl标志"
        right = "## WADL\n文件里面有很明显的wadl标志\n更多正文"
        joined = _join_merged_texts([left, bridge, right])
        self.assertEqual(joined.count("我们可以使用soapui对这类api进行测试"), 1)
        self.assertEqual(joined.count("## WADL"), 1)
        self.assertIn("文件里面有很明显的wadl标志", joined)
        self.assertIn("更多正文", joined)
        self.assertIn("正文A", joined)

    def test_single_short_code_fence_line_not_trimmed(self):
        left = "代码块开始\n```\nprint(1)"
        right = "```\nprint(2)\n代码块结束"
        joined = _join_merged_texts([left, right])
        self.assertEqual(joined, left + "\n\n" + right)


if __name__ == "__main__":
    unittest.main()
