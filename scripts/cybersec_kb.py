"""Initialize the repository template for the local cybersecurity knowledge base."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any


_KB_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def validate_knowledge_base_name(name: str) -> str:
    """Validate a knowledge-base directory name before using it in a path."""
    if not isinstance(name, str) or not name or not _KB_NAME_PATTERN.fullmatch(name):
        raise ValueError(f"Unsafe knowledge base name: {name!r}")
    return name


def load_manifest(template_root: Path) -> dict[str, Any]:
    """Load and validate the template manifest."""
    manifest_path = Path(template_root) / "manifest.json"
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        manifest = json.load(handle)
    validate_knowledge_base_name(manifest.get("name"))
    if not manifest.get("seed_documents"):
        raise ValueError("Template manifest must declare seed_documents")
    return manifest


def _merge_unique(existing: list[Any], incoming: list[Any]) -> list[Any]:
    """Append template values while preserving runtime order and custom values."""
    merged = list(existing)
    for value in incoming:
        if value not in merged:
            merged.append(value)
    return merged


def _merge_manifest(template: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    """Merge importer-managed manifest metadata without replacing user fields."""
    merged = dict(runtime)
    if template.get("categories"):
        merged["categories"] = _merge_unique(
            list(runtime.get("categories", [])), list(template["categories"])
        )

    template_sources = list(template.get("external_sources", []))
    runtime_sources = list(runtime.get("external_sources", []))
    sources: list[Any] = []
    source_keys: dict[tuple[Any, Any], int] = {}
    for source in runtime_sources:
        if not isinstance(source, dict):
            sources.append(source)
            continue
        key = (source.get("name"), source.get("category"))
        if not all(key):
            sources.append(dict(source))
            continue
        if key in source_keys:
            current = dict(sources[source_keys[key]])
            current.update(source)
            sources[source_keys[key]] = current
        else:
            source_keys[key] = len(sources)
            sources.append(dict(source))

    for source in template_sources:
        if not isinstance(source, dict):
            continue
        key = (source.get("name"), source.get("category"))
        if key in source_keys:
            current = dict(sources[source_keys[key]])
            current.update(source)
            sources[source_keys[key]] = current
        else:
            source_keys[key] = len(sources)
            sources.append(dict(source))
    if sources:
        merged["external_sources"] = sources
    return merged


def _merge_manifest_file(source: Path, destination: Path) -> str:
    """Merge a template manifest into an existing runtime manifest."""
    template = json.loads(source.read_text(encoding="utf-8-sig"))
    runtime = json.loads(destination.read_text(encoding="utf-8-sig"))
    merged = _merge_manifest(template, runtime)
    if merged != runtime:
        destination.write_text(
            json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    return "merged"


def _copy_file(source: Path, destination: Path, overwrite: bool) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return "skipped"
    shutil.copy2(source, destination)
    return "overwritten" if destination.exists() and overwrite else "copied"


def initialize(
    template_root: Path,
    data_root: Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Copy the curated template into a Chatchat runtime data root.

    Existing runtime documents are preserved by default. Only files supplied by
    the template are eligible for replacement when ``overwrite=True``.
    """
    template_root = Path(template_root).resolve()
    data_root = Path(data_root).resolve()
    manifest = load_manifest(template_root)
    name = validate_knowledge_base_name(manifest["name"])

    if not template_root.is_dir():
        raise FileNotFoundError(f"Template root does not exist: {template_root}")

    destination_root = data_root / "data" / "knowledge_base" / name
    destination_content = destination_root / "content"
    source_files = [template_root / "manifest.json"]
    readme = template_root / "README.md"
    if readme.exists():
        source_files.append(readme)
    source_files.extend(
        path for path in (template_root / "content").rglob("*") if path.is_file()
    )

    summary: dict[str, Any] = {
        "knowledge_base": name,
        "destination": str(destination_root),
        "copied_files": 0,
        "overwritten_files": 0,
        "skipped_files": 0,
        "merged_files": 0,
        "files": [],
    }
    for source in source_files:
        relative = source.relative_to(template_root)
        destination = destination_root / relative
        if (
            relative.as_posix() == "manifest.json"
            and destination.exists()
            and not overwrite
        ):
            result = _merge_manifest_file(source, destination)
        else:
            result = _copy_file(source, destination, overwrite=overwrite)
        summary[f"{result}_files"] += 1
        summary["files"].append({"path": str(relative), "action": result})
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    init_parser = subparsers.add_parser("init", help="initialize a runtime knowledge base")
    init_parser.add_argument("--template-root", required=True, type=Path)
    init_parser.add_argument("--data-root", required=True, type=Path)
    init_parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "init":
        summary = initialize(args.template_root, args.data_root, args.overwrite)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
