import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "import_des_ctf_knowledge.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("security_repo_import", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SecurityRepoImportTests(unittest.TestCase):
    def test_plain_text_is_opt_in_and_license_summary_are_skipped(self):
        helper = load_helper()
        self.assertFalse(helper.should_import(pathlib.Path("payloads/test.txt")))
        self.assertTrue(helper.should_import(pathlib.Path("payloads/test.txt"), include_plain_text=True))
        self.assertFalse(helper.should_import(pathlib.Path("src/LICENSE.md"), include_plain_text=True))
        self.assertFalse(helper.should_import(pathlib.Path("src/SUMMARY.md"), include_plain_text=True))

    def test_strip_prefix_does_not_drop_colliding_documents(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / "source"
            destination = root / "destination"
            metadata = root / "metadata"
            (source / "src").mkdir(parents=True)
            (source / "README.md").write_text("root", encoding="utf-8")
            (source / "src" / "README.md").write_text("src", encoding="utf-8")

            summary = helper.import_documents(
                source,
                destination,
                "abc123",
                "2026-08-23",
                metadata_root=metadata,
                strip_prefix="src",
            )

            self.assertEqual(summary["imported_files"], 2)
            self.assertEqual(len(list(destination.glob("*.md"))), 2)
            manifest = summary["manifest"]
            self.assertTrue(any("destination_collision_with" in doc for doc in manifest["documents"]))
    def test_custom_repository_and_strip_prefix_are_recorded(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / "source"
            destination = root / "destination"
            metadata = root / "metadata"
            (source / "src" / "web").mkdir(parents=True)
            (source / "src" / "web" / "ssrf.md").write_text("# SSRF\n", encoding="utf-8")
            (source / "src" / "payload.txt").write_text("token = 'example-secret-value'\n", encoding="utf-8")
            (source / "src" / "LICENSE.md").write_text("license", encoding="utf-8")

            summary = helper.import_documents(
                source,
                destination,
                "deadbeef",
                "2026-08-23",
                metadata_root=metadata,
                source_repository="ExampleRepo",
                source_url="https://example.invalid/repo",
                include_plain_text=True,
                strip_prefix="src",
                license_files=["src/LICENSE.md"],
            )

            self.assertEqual(summary["imported_files"], 2)
            imported = destination / "web" / "ssrf.md"
            self.assertTrue(imported.is_file())
            text = imported.read_text(encoding="utf-8")
            self.assertIn("source_repository: ExampleRepo", text)
            self.assertIn("source_url: https://example.invalid/repo", text)
            self.assertIn("source_path: src/web/ssrf.md", text)
            self.assertTrue((destination / "payload.txt").is_file())
            self.assertNotIn("example-secret-value", (destination / "payload.txt").read_text(encoding="utf-8"))
            self.assertTrue((metadata / "UPSTREAM-LICENSE.md").is_file())
            manifest = json.loads((metadata / "import-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_repository"], "ExampleRepo")
            self.assertEqual(manifest["filter"]["strip_prefix"], "src")
            self.assertEqual(len(manifest["documents"]), 2)


if __name__ == "__main__":
    unittest.main()


