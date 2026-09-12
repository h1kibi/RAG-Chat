import importlib.util
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "import_des_ctf_knowledge.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("import_noise_clean", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NoiseCleaningTests(unittest.TestCase):
    def test_provenance_header_removed_from_frontmatter(self):
        helper = load_helper()
        body = "source_repository: x\nsource_url: u\n---\n\n# Title\n\ntext\n"
        text = f"---\n{body}"
        cleaned = helper.strip_provenance_frontmatter(text)
        self.assertNotIn("source_repository", cleaned)
        self.assertTrue(cleaned.startswith("# Title"))

    def test_plain_file_without_frontmatter_untouched(self):
        helper = load_helper()
        text = "# Title\n\nbody\n"
        self.assertEqual(helper.strip_provenance_frontmatter(text), text)

    def test_mdbook_directives_and_ref_blocks_removed(self):
        helper = load_helper()
        text = (
            "## Page\n\n"
            "{{#include ../banners/training.md}}\n"
            "{{#ref}}\n../other/page.md\n{{#endref}}\n"
            "{{#tabs}}\n{{#tab name=\"Rust\"}}\n```rust\nfn main() {}\n```\n{{#endtab}}\n{{#endtabs}}\n"
            "Body text.\n"
        )
        cleaned = helper.strip_markdown_noise(text)
        self.assertNotIn("{{#", cleaned)
        self.assertIn("Body text.", cleaned)
        self.assertIn("fn main() {}", cleaned)

    def test_relative_md_links_keep_label_only(self):
        helper = load_helper()
        text = "See [File Inclusion](../pentesting-web/file-inclusion.md) and [this](https://example.com/x)."
        cleaned = helper.strip_markdown_noise(text)
        self.assertIn("See File Inclusion", cleaned)
        self.assertIn("https://example.com/x", cleaned)

    def test_import_without_header_and_with_noise_strip(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / "source"
            destination = root / "destination"
            metadata = root / "metadata"
            (source / "web").mkdir(parents=True)
            (source / "web" / "ssrf.md").write_text(
                "{{#include banners/x.md}}\n# SSRF\n\nsee [more](../web/other.md)\n",
                encoding="utf-8",
            )
            summary = helper.import_documents(
                source,
                destination,
                "abc123",
                "2026-09-10",
                metadata_root=metadata,
                provenance_header=False,
                strip_markdown_noise_enabled=True,
            )
            self.assertEqual(summary["imported_files"], 1)
            text = (destination / "web" / "ssrf.md").read_text(encoding="utf-8")
            self.assertNotIn("source_repository", text)
            self.assertNotIn("{{#", text)
            self.assertNotIn("../web/other.md", text)
            self.assertIn("# SSRF", text)

    def test_plain_flag_attachments_excluded_from_import(self):
        helper = load_helper()
        self.assertFalse(
            helper.should_import(pathlib.Path("challenge/flag.txt"), include_plain_text=True)
        )
        self.assertFalse(
            helper.should_import(
                pathlib.Path("crypto/flag.fernet.txt"), include_plain_text=True
            )
        )
        # ... and when plain text is disabled it never imports anyway
        self.assertFalse(
            helper.should_import(pathlib.Path("challenge/flag.txt"), include_plain_text=False)
        )
        # markdown write-ups naming flags stay importable
        self.assertTrue(
            helper.should_import(pathlib.Path("2025/flag-lottery.md"), include_plain_text=True)
        )
        self.assertTrue(
            helper.should_import(pathlib.Path("misc/flagcheck67.md"), include_plain_text=True)
        )

    def test_default_import_still_writes_provenance_header(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / "source"
            destination = root / "destination"
            source.mkdir()
            (source / "a.md").write_text("# A\n", encoding="utf-8")
            helper.import_documents(source, destination, "abc123", "2026-09-10")
            self.assertIn("source_repository", (destination / "a.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
