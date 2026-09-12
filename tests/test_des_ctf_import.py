import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "import_des_ctf_knowledge.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("import_des_ctf_knowledge", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DesCtfImportTests(unittest.TestCase):
    def test_should_import_markdown_but_skip_indexes_scripts_and_binary_assets(self):
        helper = load_helper()
        self.assertTrue(helper.should_import(pathlib.Path("Web/ssrf.md")))
        self.assertTrue(helper.should_import(pathlib.Path("tools/README.MD")))
        self.assertFalse(helper.should_import(pathlib.Path("Web/ssrf.idx.md")))
        self.assertFalse(helper.should_import(pathlib.Path("tools/check.py")))
        self.assertFalse(helper.should_import(pathlib.Path("assets/logo.png")))
        self.assertFalse(helper.should_import(pathlib.Path("archive.zip")))

    def test_redacts_common_secret_material_without_removing_normal_code(self):
        helper = load_helper()
        text = "api_key = 'sk-live-example-value'\nAuthorization: Bearer abcdefghijklmnop\nprint('hello')"
        redacted = helper.redact_sensitive_text(text)
        self.assertNotIn("sk-live-example-value", redacted)
        self.assertNotIn("abcdefghijklmnop", redacted)
        self.assertIn("print('hello')", redacted)
        self.assertIn("<REDACTED>", redacted)

    def test_import_adds_provenance_and_writes_manifest_idempotently(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / "source"
            destination = root / "destination"
            (source / "Web").mkdir(parents=True)
            (source / "tools").mkdir()
            (source / "Web" / "ssrf.md").write_text("# SSRF\nUse a lab target.", encoding="utf-8")
            (source / "Web" / "ssrf.idx.md").write_text("duplicate", encoding="utf-8")
            (source / "tools" / "run.py").write_text("print('not indexed')", encoding="utf-8")
            (source / "LICENSE").write_text("MIT", encoding="utf-8")

            metadata = root / "metadata"
            first = helper.import_documents(
                source, destination, "abc123", "2026-08-23", metadata_root=metadata
            )
            second = helper.import_documents(
                source, destination, "abc123", "2026-08-23", metadata_root=metadata
            )

            imported = destination / "Web" / "ssrf.md"
            self.assertEqual(first["imported_files"], 1)
            self.assertEqual(second["imported_files"], 1)
            self.assertIn("source_repository", imported.read_text(encoding="utf-8"))
            self.assertIn("abc123", imported.read_text(encoding="utf-8"))
            manifest = json.loads((metadata / "import-manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_commit"], "abc123")
            self.assertEqual(len(manifest["documents"]), 1)
            self.assertFalse((destination / "tools" / "run.py").exists())
            self.assertTrue((metadata / "UPSTREAM-LICENSE").exists())
            self.assertFalse((destination / "import-manifest.json").exists())


if __name__ == "__main__":
    unittest.main()


