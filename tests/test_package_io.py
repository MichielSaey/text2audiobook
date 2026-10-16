"""Package reader: preface, markdown content, and a short takeaway stay separate."""

import json
from pathlib import Path

from text2audiobook.formats.package import read_package, resolve_package_path
from text2audiobook.io import MIN_SECTION_WORDS

_BODY = " ".join(f"word{i}" for i in range(40))


def test_package_reader_keeps_short_takeaway(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    source = package / "sources" / "demo2020note"
    source.mkdir(parents=True)
    (source / "preface.md").write_text("A short preface.\n", encoding="utf-8")
    (source / "content.md").write_text(f"# Body chapter\n\n{_BODY}\n", encoding="utf-8")
    (source / "takeaways.md").write_text("- One short takeaway.\n", encoding="utf-8")
    (package / "package.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "target_language": "English",
                "title": "Demo package",
                "sources": [
                    {
                        "citekey": "demo2020note",
                        "title": "Demo Note",
                        "preface": "sources/demo2020note/preface.md",
                        "content": "sources/demo2020note/content.md",
                        "content_type": "markdown",
                        "takeaways": "sources/demo2020note/takeaways.md",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    resolved = resolve_package_path(package)
    assert resolved == resolve_package_path(package / "package.json")
    metadata, chapters = read_package(
        resolved,
        staging_root=tmp_path / "staging",
        output_root=tmp_path / "output",
    )

    assert metadata.title == "Demo package"
    assert [chapter.title for chapter in chapters] == [
        "Preface",
        "Body chapter",
        "Key takeaways",
    ]
    assert "A short preface." in chapters[0].text
    assert len(chapters[0].text.split()) < MIN_SECTION_WORDS
    assert "word0" in chapters[1].text
    assert "word0" not in chapters[0].text
    assert "word0" not in chapters[2].text
    assert "One short takeaway." in chapters[2].text
    assert len(chapters[2].text.split()) < MIN_SECTION_WORDS
    assert "One short takeaway." not in chapters[1].text
