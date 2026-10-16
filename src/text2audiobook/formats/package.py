"""Trantor package input: preface, one content file, then key takeaways per source."""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path

from text2audiobook.formats.html_extract import title_from_filename
from text2audiobook.formats.markdown import markdown_to_text
from text2audiobook.io import (
    BookMetadata,
    Chapter,
    assign_chapter_slugs,
    finalize_metadata,
)

_LIST_MARKER = re.compile(r"^(?:[-*+]|\d+[.)])\s+")
_TYPE_SUFFIXES = {
    "pdf": frozenset({".pdf"}),
    "markdown": frozenset({".md", ".markdown"}),
    "html": frozenset({".html", ".htm"}),
}


def resolve_package_path(path: Path) -> Path:
    """Return package.json for a package directory or a direct file path."""
    candidate = Path(path).expanduser()
    if candidate.is_dir():
        candidate = candidate / "package.json"
    if not candidate.is_file():
        raise FileNotFoundError(f"package.json not found at {candidate}")
    return candidate.resolve()


def _language(target_language: str) -> str:
    cleaned = target_language.strip() or "English"
    if cleaned.casefold() == "english":
        return "en"
    return cleaned


def _spoken_markdown(text: str) -> str:
    """Plain speech for a preface or takeaway file. Short texts are kept."""
    lines: list[str] = []
    for line in markdown_to_text(text).splitlines():
        cleaned = _LIST_MARKER.sub("", line.strip())
        if cleaned:
            lines.append(cleaned)
    spoken = "\n".join(lines).strip()
    if spoken:
        return spoken
    return text.strip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _member(package_root: Path, relative: str) -> Path:
    if not str(relative).strip():
        raise ValueError("Package path is empty")
    path = (package_root / relative).resolve()
    if not path.is_relative_to(package_root.resolve()):
        raise ValueError(f"Package path escapes the package directory: {relative}")
    if not path.is_file():
        raise FileNotFoundError(f"Package file missing: {relative}")
    return path


def _fallback_plain(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return markdown_to_text(_read_text(path)).strip()
    if suffix in {".html", ".htm"}:
        from text2audiobook.formats.html_extract import html_to_markdownish

        return markdown_to_text(html_to_markdownish(path.read_bytes())).strip()
    return ""


def _content_units(path: Path, content_type: str) -> list[tuple[str, str]]:
    """Read one content file with the existing reader. Do not merge it with other texts."""
    suffix = path.suffix.lower()
    allowed = _TYPE_SUFFIXES.get(content_type)
    if allowed is None or suffix not in allowed:
        raise ValueError(
            f"Unsupported package content {path.name} ({content_type or 'unknown type'})"
        )

    import text2audiobook.formats  # noqa: F401 — register PDF, Markdown, and HTML readers
    from text2audiobook.io import parse_source

    with tempfile.TemporaryDirectory(prefix="t2a-package-") as scratch:
        root = Path(scratch)
        try:
            _metadata, chapters = parse_source(
                path,
                staging_root=root / "staging",
                output_root=root / "output",
            )
        except ValueError as exc:
            if "No readable chapters" not in str(exc):
                raise
            text = _fallback_plain(path)
            if not text:
                raise
            return [(title_from_filename(path), text)]

    units = [(chapter.title, chapter.text.strip()) for chapter in chapters if chapter.text.strip()]
    if units:
        return units
    text = _fallback_plain(path)
    if not text:
        raise ValueError(f"No readable chapters found in {path.name}")
    return [(title_from_filename(path), text)]


def read_package(
    path: Path,
    *,
    staging_root: Path,
    output_root: Path,
) -> tuple[BookMetadata, list[Chapter]]:
    """One audiobook: for each source, Preface, then that file's chapters, then Key takeaways."""
    package_json = resolve_package_path(path)
    try:
        data = json.loads(package_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid package.json: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("package.json must be an object")
    if data.get("schema") != 1:
        raise ValueError(f"Unsupported package schema {data.get('schema')!r}")
    sources = data.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("package.json needs a non-empty sources list")

    package_root = package_json.parent
    title = str(data.get("title") or "").strip() or "Trantor package"
    many = len(sources) > 1
    units: list[tuple[str, str]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError("Each package source must be an object")
        citekey = str(source.get("citekey") or "").strip()
        if not citekey:
            raise ValueError("Package source is missing citekey")
        source_title = str(source.get("title") or citekey).strip()
        preface_name = "Preface" if not many else f"{source_title}: Preface"
        takeaway_name = "Key takeaways" if not many else f"{source_title}: Key takeaways"
        for label, key in ((preface_name, "preface"), (takeaway_name, "takeaways")):
            relative = source.get(key)
            if not isinstance(relative, str) or not relative.strip():
                raise ValueError(f"{citekey} is missing {key}")
        content_relative = source.get("content")
        if not isinstance(content_relative, str) or not content_relative.strip():
            raise ValueError(f"{citekey} is missing content")
        content_type = str(source.get("content_type") or "").strip().lower()

        preface = _spoken_markdown(_read_text(_member(package_root, source["preface"])))
        if not preface:
            raise ValueError(f"{citekey} preface is empty")
        content = _content_units(_member(package_root, content_relative), content_type)
        takeaways = _spoken_markdown(_read_text(_member(package_root, source["takeaways"])))
        if not takeaways:
            raise ValueError(f"{citekey} key takeaways are empty")
        units.append((preface_name, preface))
        units.extend(content)
        units.append((takeaway_name, takeaways))

    metadata = finalize_metadata(
        title=title,
        author="Trantor",
        language=_language(str(data.get("target_language") or "English")),
        cover_bytes=None,
        staging_root=staging_root,
        output_root=output_root,
    )
    return metadata, assign_chapter_slugs(units)
