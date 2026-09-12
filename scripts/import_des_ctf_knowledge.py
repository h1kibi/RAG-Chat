"""Import safe, traceable text material from a security knowledge repository.

The importer only reads selected text files. It never executes upstream content.
It is intentionally usable for Markdown repositories and payload wordlists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence


SOURCE_REPOSITORY = "Des-CTF-Knowledge"
SOURCE_URL = "https://github.com/Dest1ny-Sec/Des-CTF-Knowledge"
MARKDOWN_SUFFIXES = {".md", ".markdown"}
PLAIN_TEXT_SUFFIXES = {".txt"}
STRUCTURED_TEXT_SUFFIXES = {".yml", ".yaml"}
EXCLUDED_SUFFIXES = {
    ".py", ".pyc", ".ps1", ".psm1", ".sh", ".bash", ".zsh", ".bat", ".cmd",
    ".js", ".mjs", ".cjs", ".php", ".pl", ".rb", ".go", ".c", ".cc", ".cpp",
    ".h", ".hpp", ".java", ".class", ".dll", ".exe", ".bin", ".zip", ".7z",
    ".rar", ".tar", ".gz", ".bz2", ".xz", ".png", ".jpg", ".jpeg", ".gif",
    ".webp", ".svg", ".ico", ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".xml", ".html", ".css", ".js", ".mvg", ".avi",
}
EXCLUDED_DIRS = {
    ".git", ".github", "__pycache__", "node_modules", "dist", "build", "target",
    "vendor", "theme", "images", "image", "assets",
}
EXCLUDED_NAMES = {
    "summary.md",
    "agents.md",
    "license",
    "license.md",
    "license.notice",
    "copying",
    "notice",
    "notice.md",
    "notice.txt",
}

_PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
_BEARER_RE = re.compile(r"(?i)(\b(?:authorization\s*:\s*bearer|bearer)\s+)[A-Za-z0-9._~+/=-]{8,}")
_ASSIGNMENT_RE = re.compile(
    r"(?im)(\b(?:api[_-]?key|secret[_-]?key|access[_-]?token|token|password|passwd|cookie)\b\s*[:=]\s*[\"']?)([^\s\"']{4,})"
)


def should_import(
    path: Path,
    include_plain_text: bool = False,
    include_structured_text: bool = False,
) -> bool:
    """Return whether a relative source path is safe and useful to index."""
    parts = {part.lower() for part in path.parts}
    if parts & EXCLUDED_DIRS:
        return False
    name = path.name.lower()
    suffix = path.suffix.lower()
    if name in EXCLUDED_NAMES:
        return False
    if name.endswith(".idx.md") or name.endswith(".index.md"):
        return False
    if suffix in EXCLUDED_SUFFIXES:
        return False
    if suffix in MARKDOWN_SUFFIXES:
        return True
    if include_plain_text and suffix in PLAIN_TEXT_SUFFIXES:
        if _is_flag_attachment_name(name):
            return False
        return True
    return include_structured_text and suffix in STRUCTURED_TEXT_SUFFIXES


def _is_flag_attachment_name(name: str) -> bool:
    """Exclude imported raw CTF attachment files (flag.txt, flag.fernet.txt).

    Markdown write-ups that merely mention flags are unaffected because they
    never reach this branch.
    """
    lowered = name.lower()
    stem = lowered.split(".", 1)[0] if "." in lowered else lowered
    return lowered.startswith("flag") and stem in {"flag", "flag1", "flag2", "flag3"}


def redact_sensitive_text(text: str) -> str:
    """Redact common credential material while leaving ordinary examples intact."""
    text = _PRIVATE_KEY_RE.sub("<REDACTED_PRIVATE_KEY>", text)
    text = _BEARER_RE.sub(r"\1<REDACTED>", text)
    return _ASSIGNMENT_RE.sub(r"\1<REDACTED>", text)


_MDBOOK_DIRECTIVE_RE = re.compile(
    r"\{\{#[a-z-]+(?:[^{}]*)\}\}\n?"
)
_INCLUDE_BLOCK_RE = re.compile(
    r"\{\{#ref\}\}\n.*?\n\{\{#endref\}\}\n?", re.DOTALL
)
_RELATIVE_MD_LINK_RE = re.compile(
    r"\[([^\]]*)\]\((?=[^)]*\.md(?:#|\?|\)|$))[^)]*\)"
)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_BLANK_LINES_RE = re.compile(r"\n{4,}")


def strip_markdown_noise(text: str) -> str:
    """Remove mdbook render scaffolding and dead relative links from Markdown.

    mdBook directives (``{{#include ../banners/...}}``, ``{{#ref}}`` blocks)
    are render-time instructions that leak into plain-text chunks; relative
    ``.md`` links point at pages that do not exist outside the book. The text
    label of such a link is kept, so surrounding prose stays readable.
    """
    text = _HTML_COMMENT_RE.sub("", text)
    text = _INCLUDE_BLOCK_RE.sub("", text)
    text = _MDBOOK_DIRECTIVE_RE.sub("", text)
    text = _RELATIVE_MD_LINK_RE.sub(r"\1", text)
    return _BLANK_LINES_RE.sub("\n\n\n", text)


def strip_provenance_frontmatter(text: str) -> str:
    """Remove the YAML provenance block the importer prepends to each file."""
    if not text.startswith("---\n"):
        return text
    end = text.find("\n---\n", 4)
    if end < 0 or end > 4096:
        return text
    return text[end + 5 :].lstrip("\n")


def build_provenance_header(
    source_path: str,
    commit: str,
    retrieved_at: str,
    source_repository: str = SOURCE_REPOSITORY,
    source_url: str | None = SOURCE_URL,
) -> str:
    url_line = f"source_url: {source_url}\n" if source_url else ""
    return (
        "---\n"
        f"source_repository: {source_repository}\n"
        f"{url_line}"
        f"source_path: {source_path}\n"
        f"source_commit: {commit}\n"
        f"retrieved_at: {retrieved_at}\n"
        "usage: authorized-lab-ctf-defense-research-only\n"
        "notice: Imported text is for authorized systems, isolated labs, CTFs, and defensive research.\n"
        "---\n\n"
    )


def _iter_source_files(
    source_root: Path,
    include_plain_text: bool = False,
    include_structured_text: bool = False,
) -> Iterable[Path]:
    for path in sorted(source_root.rglob("*")):
        if path.is_file() and should_import(
            path.relative_to(source_root),
            include_plain_text=include_plain_text,
            include_structured_text=include_structured_text,
        ):
            yield path

def _destination_relative(
    relative: Path,
    strip_prefix: str | None,
    normalize_structured_text: bool = False,
) -> Path:
    if strip_prefix:
        prefix = Path(strip_prefix)
        try:
            relative = relative.relative_to(prefix)
        except ValueError:
            pass
    if normalize_structured_text and relative.suffix.lower() in STRUCTURED_TEXT_SUFFIXES:
        return relative.with_suffix(".md")
    return relative


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _remove_previous_documents(destination_root: Path, metadata_root: Path) -> None:
    manifest_path = metadata_root / "import-manifest.json"
    if not manifest_path.is_file():
        return
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    for document in previous.get("documents", []):
        destination_path = document.get("destination_path")
        if not isinstance(destination_path, str) or not destination_path:
            continue
        candidate = (destination_root / Path(destination_path)).resolve()
        try:
            candidate.relative_to(destination_root)
        except ValueError:
            continue
        if candidate.is_file():
            candidate.unlink()


def _copy_license_files(
    source_root: Path,
    metadata_root: Path,
    license_files: Sequence[str] | None,
) -> list[str]:
    candidates = list(license_files or ("LICENSE", "LICENSE.notice", "LICENSE.md", "src/LICENSE.md"))
    copied: list[str] = []
    seen: set[Path] = set()
    for candidate_name in candidates:
        source_path = (source_root / candidate_name).resolve()
        try:
            source_path.relative_to(source_root)
        except ValueError:
            continue
        if source_path in seen or not source_path.is_file():
            continue
        seen.add(source_path)
        destination_name = f"UPSTREAM-{source_path.name}"
        (metadata_root / destination_name).write_bytes(source_path.read_bytes())
        copied.append(str(source_path.relative_to(source_root)).replace("\\", "/"))
    return copied


def import_documents(
    source_root: Path,
    destination_root: Path,
    commit: str,
    retrieved_at: str | None = None,
    metadata_root: Path | None = None,
    source_repository: str = SOURCE_REPOSITORY,
    source_url: str | None = SOURCE_URL,
    include_plain_text: bool = False,
    strip_prefix: str | None = None,
    license_files: Sequence[str] | None = None,
    include_structured_text: bool = False,
    provenance_header: bool = True,
    strip_markdown_noise_enabled: bool = False,
) -> dict:
    """Copy filtered text and keep import metadata outside indexed content."""
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()
    metadata_root = (Path(metadata_root) if metadata_root is not None else destination_root).resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source root does not exist: {source_root}")
    retrieved_at = retrieved_at or date.today().isoformat()
    destination_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)
    _remove_previous_documents(destination_root, metadata_root)

    documents = []
    used_destinations: dict[str, str] = {}
    for source_path in _iter_source_files(
        source_root,
        include_plain_text=include_plain_text,
        include_structured_text=include_structured_text,
    ):
        relative = source_path.relative_to(source_root)
        destination_relative = _destination_relative(
            relative,
            strip_prefix,
            normalize_structured_text=include_structured_text,
        )
        destination_key = str(destination_relative).replace("\\", "/").casefold()
        collision_with = None
        if destination_key in used_destinations:
            original_destination = destination_relative
            stem = destination_relative.stem
            suffix = destination_relative.suffix
            counter = 1
            while destination_key in used_destinations:
                destination_relative = destination_relative.with_name(f"{stem}__duplicate_{counter}{suffix}")
                destination_key = str(destination_relative).replace("\\", "/").casefold()
                counter += 1
            collision_with = used_destinations[str(original_destination).replace("\\", "/").casefold()]
        used_destinations[destination_key] = str(relative).replace("\\", "/")
        raw = source_path.read_bytes()
        text = raw.decode("utf-8-sig", errors="replace")
        text = redact_sensitive_text(text)
        if strip_markdown_noise_enabled:
            text = strip_markdown_noise(text)
        if provenance_header:
            rendered = build_provenance_header(
                str(relative).replace("\\", "/"), commit, retrieved_at, source_repository, source_url
            ) + text
        else:
            rendered = text
        destination = destination_root / destination_relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        documents.append(
            {
                "source_path": str(relative).replace("\\", "/"),
                "destination_path": str(destination_relative).replace("\\", "/"),
                "sha256": _sha256(raw),
                "bytes": len(raw),
                **({"destination_collision_with": collision_with} if collision_with else {}),
            }
        )

    copied_licenses = _copy_license_files(source_root, metadata_root, license_files)
    included_extensions = set(MARKDOWN_SUFFIXES)
    if include_plain_text:
        included_extensions |= PLAIN_TEXT_SUFFIXES
    if include_structured_text:
        included_extensions |= STRUCTURED_TEXT_SUFFIXES
    manifest = {
        "source_repository": source_repository,
        "source_url": source_url,
        "source_commit": commit,
        "retrieved_at": retrieved_at,
        "filter": {
            "included_extensions": sorted(included_extensions),
            "excluded_indexes": ["*.idx.md", "*.index.md", "SUMMARY.md"],
            "excluded_content": ["scripts", "binaries", "images", "archives", "caches", "licenses"],
            "include_plain_text": include_plain_text,
            "include_structured_text": include_structured_text,
            "structured_text_output_extension": ".md" if include_structured_text else None,
            "strip_prefix": strip_prefix,
        },
        "licenses": copied_licenses,
        "documents": documents,
    }
    (metadata_root / "import-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    upstream_line = f"- Upstream: {source_url}\n" if source_url else ""
    (metadata_root / "SOURCE-ATTRIBUTION.md").write_text(
        "# Imported source attribution\n\n"
        f"- Repository: {source_repository}\n"
        f"{upstream_line}"
        f"- Commit: `{commit}`\n"
        f"- Retrieved: `{retrieved_at}`\n"
        "- Use: authorized labs, CTFs, and defensive research only\n\n"
        "Imported documents retain their upstream relative path and SHA256 in the manifest. "
        "Individual articles, payload lists, and code examples may have their own authorship or license terms.\n",
        encoding="utf-8",
    )
    return {"imported_files": len(documents), "destination_root": str(destination_root), "manifest": manifest}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    import_parser = subparsers.add_parser("import")
    import_parser.add_argument("--source-root", required=True, type=Path)
    import_parser.add_argument("--destination-root", required=True, type=Path)
    import_parser.add_argument("--commit", required=True)
    import_parser.add_argument("--retrieved-at", default=None)
    import_parser.add_argument("--metadata-root", default=None, type=Path)
    import_parser.add_argument("--source-repository", default=SOURCE_REPOSITORY)
    import_parser.add_argument("--source-url", default=SOURCE_URL)
    import_parser.add_argument("--include-plain-text", action="store_true")
    import_parser.add_argument("--include-structured-text", action="store_true")
    import_parser.add_argument("--strip-prefix", default=None)
    import_parser.add_argument("--license-file", action="append", default=None)
    import_parser.add_argument("--no-provenance-header", action="store_true")
    import_parser.add_argument("--strip-markdown-noise", action="store_true")
    args = parser.parse_args()
    if args.command == "import":
        summary = import_documents(
            args.source_root,
            args.destination_root,
            args.commit,
            args.retrieved_at,
            args.metadata_root,
            args.source_repository,
            args.source_url,
            args.include_plain_text,
            args.strip_prefix,
            args.license_file,
            include_structured_text=args.include_structured_text,
            provenance_header=not args.no_provenance_header,
            strip_markdown_noise_enabled=args.strip_markdown_noise,
        )
        print(json.dumps({"imported_files": summary["imported_files"], "destination_root": summary["destination_root"]}, ensure_ascii=False))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())




