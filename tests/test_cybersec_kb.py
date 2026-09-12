import json
import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TEMPLATE_ROOT = REPO_ROOT / "knowledge-base" / "cybersec"


class CybersecTemplateTests(unittest.TestCase):
    def test_manifest_declares_authorized_cybersecurity_categories(self):
        manifest_path = TEMPLATE_ROOT / "manifest.json"
        self.assertTrue(manifest_path.exists(), manifest_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))

        self.assertEqual(manifest["name"], "cybersec")
        self.assertIn("授权", manifest["authorized_use"])
        self.assertGreaterEqual(len(manifest["categories"]), 6)
        self.assertGreaterEqual(len(manifest["seed_documents"]), 6)

    def test_required_categories_have_non_empty_markdown_documents(self):
        required = {
            "00_foundations",
            "01_web_security",
            "02_network_security",
            "04_vulnerability",
            "05_pentest_method",
            "06_ctf",
            "07_defense",
        }
        for category in required:
            category_dir = TEMPLATE_ROOT / "content" / category
            self.assertTrue(category_dir.is_dir(), category_dir)
            documents = list(category_dir.glob("*.md"))
            self.assertTrue(documents, category)
            self.assertTrue(
                all(document.read_text(encoding="utf-8-sig").strip() for document in documents),
                category,
            )


if __name__ == "__main__":
    unittest.main()
import importlib.util
import json
import pathlib
import tempfile
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HELPER_PATH = REPO_ROOT / "scripts" / "cybersec_kb.py"


def load_helper():
    spec = importlib.util.spec_from_file_location("cybersec_kb", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CybersecInitializerTests(unittest.TestCase):
    def test_initialize_copies_template_and_preserves_user_files(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = pathlib.Path(temp_dir) / "data"
            template_root = REPO_ROOT / "knowledge-base" / "cybersec"
            destination = data_root / "data" / "knowledge_base" / "cybersec" / "content"
            destination.mkdir(parents=True)
            user_file = destination / "my-authorized-notes.md"
            user_file.write_text("keep this", encoding="utf-8")

            summary = helper.initialize(template_root, data_root)

            self.assertGreater(summary["copied_files"], 0)
            self.assertTrue(user_file.exists())
            self.assertEqual(user_file.read_text(encoding="utf-8"), "keep this")
            self.assertTrue((destination / "00_foundations" / "security-foundations.md").exists())
            manifest = json.loads(
                (data_root / "data" / "knowledge_base" / "cybersec" / "manifest.json").read_text(
                    encoding="utf-8-sig"
                )
            )
            self.assertEqual(manifest["name"], "cybersec")

    def test_initialize_merges_managed_manifest_metadata_without_overwriting_user_fields(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            data_root = pathlib.Path(temp_dir) / "data"
            template_root = REPO_ROOT / "knowledge-base" / "cybersec"
            runtime_root = data_root / "data" / "knowledge_base" / "cybersec"
            runtime_root.mkdir(parents=True)
            (runtime_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "name": "cybersec",
                        "categories": ["custom_category"],
                        "custom_field": "keep me",
                    }
                ),
                encoding="utf-8",
            )

            helper.initialize(template_root, data_root)

            manifest = json.loads((runtime_root / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["custom_field"], "keep me")
            self.assertIn("custom_category", manifest["categories"])
            self.assertIn("ctf_des_knowledge", manifest["categories"])
            template_manifest = json.loads(
                (template_root / "manifest.json").read_text(encoding="utf-8-sig")
            )
            expected_sources = {
                source["name"] for source in template_manifest["external_sources"]
            }
            actual_sources = {
                source["name"] for source in manifest["external_sources"]
            }
            self.assertTrue(expected_sources.issubset(actual_sources))
    def test_unsafe_knowledge_base_name_is_rejected(self):
        helper = load_helper()
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                helper.validate_knowledge_base_name("..\\outside")
            with self.assertRaises(ValueError):
                helper.validate_knowledge_base_name("cybersec/name")


if __name__ == "__main__":
    unittest.main()


