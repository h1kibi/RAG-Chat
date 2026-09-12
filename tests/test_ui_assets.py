"""The chat UI renders Markdown, which makes its sanitizer load-bearing.

Model output is influenced by retrieved corpus text, and the corpus deliberately
contains prompt-injection payloads (see rag_service/README.md). Parsing that into
HTML without a sanitizer would let a poisoned document run script in the
operator's browser, so these checks guard the pieces that prevent it:

- both libraries are vendored locally (a CDN reference would also break the
  documented "loads offline" behaviour on an air-gapped host), and
- the renderer keeps calling the sanitizer with a restrictive allow-list.

They are text assertions, intended to fail loudly if someone simplifies the UI.
The behavioural proof lives in the browser: 15/15 cases (benign markdown
untouched, script/iframe/style/data:/javascript: payloads stripped, no execution).
"""
import unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent / "agent_service"
INDEX = PACKAGE / "static" / "index.html"
VENDOR = PACKAGE / "static" / "vendor"
PYPROJECT = PACKAGE.parent / "pyproject.toml"


class VendoredAssetTests(unittest.TestCase):
    def test_libraries_are_present_and_not_stubs(self):
        for name in ("marked.min.js", "purify.min.js"):
            path = VENDOR / name
            self.assertTrue(path.is_file(), f"missing vendored asset: {path}")
            self.assertGreater(path.stat().st_size, 10_000, f"{name} looks truncated")
            self.assertIn(b"license", path.read_bytes()[:400].lower(), f"{name} lost its header")

    def test_dependencies_are_documented_with_their_licences(self):
        # MIT and Apache-2.0/MPL-2.0 both require attribution to travel with the
        # distribution.
        text = (VENDOR / "README.md").read_text(encoding="utf-8")
        for token in ("marked", "DOMPurify", "MIT", "Apache", "MPL"):
            self.assertIn(token, text)

    def test_the_page_loads_them_locally_and_never_from_a_cdn(self):
        html = INDEX.read_text(encoding="utf-8")
        self.assertIn("/static/vendor/marked.min.js", html)
        self.assertIn("/static/vendor/purify.min.js", html)
        # Any absolute URL in a script/link tag would break offline use.
        self.assertNotRegex(
            html,
            r'(?:src|href)\s*=\s*["\']https?://',
            "the UI must not reference an external URL",
        )

    def test_the_wheel_ships_the_vendor_directory(self):
        # The UI loads these at runtime, so they must be package data; an
        # editable install hides a missing entry because it reads the source tree.
        text = PYPROJECT.read_text(encoding="utf-8")
        self.assertRegex(text, r'agent_service\s*=\s*\[[^\]]*static/vendor/\*\.js')


class SanitizerContractTests(unittest.TestCase):
    def setUp(self):
        self.html = INDEX.read_text(encoding="utf-8")

    def test_markdown_output_is_sanitized(self):
        self.assertRegex(
            self.html,
            r"DOMPurify\.sanitize\(\s*[A-Za-z]+,\s*MARKDOWN_ALLOWED\s*\)",
            "rendered markdown must pass through DOMPurify before reaching the DOM",
        )

    def test_script_and_embedding_surfaces_are_not_allowed(self):
        block = self.html.split("MARKDOWN_ALLOWED", 1)[1].split("};", 1)[0]
        for forbidden in ("script", "iframe", "style", "object", "embed", "form"):
            self.assertNotIn(
                f'"{forbidden}"', block, f"{forbidden} must not be in the allow-list"
            )

    def test_dangerous_url_schemes_are_rejected(self):
        # `javascript:` hrefs and `data:` sources must not survive; the regexp
        # only admits http(s), mailto, in-page anchors, and same-origin paths.
        block = self.html.split("ALLOWED_URI_REGEXP", 1)[1].split("\n", 1)[0]
        self.assertIn("https?", block)
        self.assertIn("mailto:", block)
        self.assertNotIn("data:", block)
        self.assertNotIn("javascript", block)

    def test_a_stripped_payload_is_reported_to_the_reader(self):
        # Silently mutilated output would be trusted as complete.
        self.assertIn("mdstripped", self.html)
        self.assertIn("安全过滤", self.html)

    def test_a_missing_library_degrades_to_plain_text(self):
        # A blank bubble would be worse than an unformatted answer, so the
        # renderer must check for the global instead of assuming it loaded.
        self.assertRegex(self.html, r'typeof\s+marked\s*===\s*"undefined"')
        self.assertRegex(self.html, r'typeof\s+DOMPurify\s*===\s*"undefined"')


class MarkdownConfigTests(unittest.TestCase):
    def test_gfm_and_single_newline_breaks_are_enabled(self):
        # Chat text relies on both: tables/strikethrough come from GFM, and a
        # single newline has to break a line rather than join a paragraph.
        html = INDEX.read_text(encoding="utf-8")
        block = html.split("marked.use(", 1)[1].split("});", 1)[0]
        self.assertIn("gfm: true", block)
        self.assertIn("breaks: true", block)

    def test_assistant_bubbles_are_the_markdown_surface(self):
        html = INDEX.read_text(encoding="utf-8")
        # User input stays plain text: it is the user's own typing and there is
        # no reason to give it an HTML surface.
        self.assertIn('bubble.classList.add("md")', html)
        self.assertNotRegex(html, r'render\("user"[^)]*\)[\s\S]{0,120}classList\.add\("md"\)')


if __name__ == "__main__":
    unittest.main()
