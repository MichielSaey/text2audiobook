"""Trantor briefing intake: markdown chapters, fingerprint skip, no-GPU exit."""

import hashlib
import json
from pathlib import Path

from text2audiobook.briefing import (
    NO_GPU_MESSAGE,
    UNCHANGED_MESSAGE,
    briefing_fingerprint_file,
    expected_briefing_m4b,
    export_proposed_briefing,
    machine_has_nvidia_gpu,
    should_skip_briefing,
    write_briefing_fingerprint,
)
from text2audiobook.cli import main
from text2audiobook.formats.markdown import MarkdownReader
from text2audiobook.io import MIN_SECTION_WORDS

_BOOK_SENTENCE = "This is a book, so the full text is not included in the briefing."
_INTRO = "Proposed sources for the next listening session."
_BOOK_BODY = (
    "Citekey fisher2009 What it is: Mark Fisher's book on the cultural logic of "
    "late capitalism, summarized here so this heading body stays long enough for "
    "the markdown reader to keep it. Source text: "
    + _BOOK_SENTENCE
)
_ARTICLE_BODY = (
    "Citekey doe2024 What it is: A journal article on how researchers listen to "
    "archival recordings when the source is not a book. Source text: The attached "
    "essay says that a spoken summary should carry the argument even when the PDF "
    "stays on the shelf."
)
_BRIEFING = f"""# Proposed research sources

{_INTRO}

## Capitalist Realism

{_BOOK_BODY}

## Archival Listening

{_ARTICLE_BODY}
"""


def _config(tmp_path: Path) -> Path:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "paths": {
                    "input_dir": "input",
                    "staging_dir": "staging",
                    "output_dir": "output",
                    "runs_dir": "runs",
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def _markdown(tmp_path: Path, text: str | None = None) -> Path:
    path = tmp_path / "input" / "trantor-proposed.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text is not None else _BRIEFING, encoding="utf-8")
    return path


def test_markdown_reader_extracts_two_briefing_sources(tmp_path: Path) -> None:
    # The intro sits under the H1 and stays under MIN_SECTION_WORDS, so it is
    # dropped. Each source is one H2 body and stays above that minimum.
    assert len(_INTRO.split()) < MIN_SECTION_WORDS
    assert len(_BOOK_BODY.split()) >= MIN_SECTION_WORDS
    assert len(_ARTICLE_BODY.split()) >= MIN_SECTION_WORDS

    path = _markdown(tmp_path)
    metadata, chapters = MarkdownReader().read(
        path,
        staging_root=tmp_path / "staging",
        output_root=tmp_path / "output",
    )
    assert metadata.title == "Proposed research sources"
    assert [chapter.title for chapter in chapters] == [
        "Capitalist Realism",
        "Archival Listening",
    ]
    assert _BOOK_SENTENCE in chapters[0].text
    assert _BOOK_SENTENCE not in chapters[1].text
    assert "Citekey fisher2009" in chapters[0].text
    assert "Citekey doe2024" in chapters[1].text
    assert "The attached essay" in chapters[1].text


def test_expected_m4b_uses_briefing_h1(tmp_path: Path) -> None:
    path = _markdown(tmp_path, "# Proposed research sources\n\nShort intro.\n")
    assert expected_briefing_m4b(path, tmp_path / "output") == (
        tmp_path / "output" / "proposed_research_sources.m4b"
    )


def test_fingerprint_skip_matches_hash_and_existing_m4b(tmp_path: Path) -> None:
    markdown = _markdown(tmp_path)
    m4b = tmp_path / "output" / "proposed_research_sources.m4b"
    m4b.parent.mkdir()
    m4b.write_bytes(b"audio")
    fingerprint = briefing_fingerprint_file(tmp_path / "runs")
    write_briefing_fingerprint(markdown, fingerprint)

    digest = hashlib.sha256(markdown.read_bytes()).hexdigest()
    assert fingerprint.read_text(encoding="utf-8") == digest + "\n"
    assert should_skip_briefing(
        markdown, fingerprint_path=fingerprint, m4b_path=m4b
    )
    assert (
        should_skip_briefing(
            markdown, fingerprint_path=fingerprint, m4b_path=m4b, force=True
        )
        is False
    )

    markdown.write_text(markdown.read_text(encoding="utf-8") + "\nMore.\n", encoding="utf-8")
    assert (
        should_skip_briefing(markdown, fingerprint_path=fingerprint, m4b_path=m4b)
        is False
    )
    m4b.unlink()
    write_briefing_fingerprint(markdown, fingerprint)
    assert (
        should_skip_briefing(markdown, fingerprint_path=fingerprint, m4b_path=m4b)
        is False
    )


def test_gpu_gate_requires_device_or_nvidia_smi(tmp_path: Path) -> None:
    missing = tmp_path / "no-nvidia0"
    present = tmp_path / "nvidia0"
    present.write_text("", encoding="utf-8")
    called = {"smi": 0}

    def failing_smi() -> int:
        called["smi"] += 1
        return 1

    assert machine_has_nvidia_gpu(device=present, nvidia_smi=failing_smi) is True
    assert called["smi"] == 0
    assert machine_has_nvidia_gpu(device=missing, nvidia_smi=lambda: 1) is False
    assert machine_has_nvidia_gpu(device=missing, nvidia_smi=lambda: 0) is True

    def broken() -> int:
        raise OSError("nvidia-smi missing")

    assert machine_has_nvidia_gpu(device=missing, nvidia_smi=broken) is False


def test_export_runs_trantor_with_repo_relative_out(tmp_path: Path) -> None:
    out = tmp_path / "data" / "input" / "trantor-proposed.md"
    captured: dict[str, object] = {}

    def which(name: str) -> str | None:
        assert name == "trantor"
        return "/opt/trantor"

    def runner(argv: list[str], **kwargs: object) -> object:
        captured["argv"] = argv
        captured["kwargs"] = kwargs

        class Result:
            returncode = 0

        return Result()

    code = export_proposed_briefing(out, cwd=tmp_path, runner=runner, which=which)
    assert code == 0
    assert captured["argv"] == [
        "/opt/trantor",
        "briefing",
        "--status",
        "proposed",
        "--out",
        "data/input/trantor-proposed.md",
    ]
    assert captured["kwargs"] == {"check": False, "cwd": tmp_path}
    assert out.parent.is_dir()


def test_export_returns_none_when_trantor_is_absent(tmp_path: Path) -> None:
    def runner(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("trantor should not run")

    code = export_proposed_briefing(
        tmp_path / "out.md",
        runner=runner,
        which=lambda _name: None,
    )
    assert code is None


def test_briefing_cli_skips_without_gpu(monkeypatch, capsys) -> None:
    monkeypatch.setattr("text2audiobook.briefing.machine_has_nvidia_gpu", lambda **_kw: False)

    def explode(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("trantor should not run")

    monkeypatch.setattr("text2audiobook.briefing.export_proposed_briefing", explode)
    assert main(["--briefing"]) == 0
    captured = capsys.readouterr()
    assert NO_GPU_MESSAGE in captured.out


def test_briefing_cli_skips_unchanged_without_pipeline(monkeypatch, tmp_path: Path, capsys) -> None:
    markdown = _markdown(tmp_path, "# Proposed research sources\n\nShort intro.\n")
    m4b = tmp_path / "output" / "proposed_research_sources.m4b"
    m4b.parent.mkdir()
    m4b.write_bytes(b"audio")
    write_briefing_fingerprint(markdown, briefing_fingerprint_file(tmp_path / "runs"))

    monkeypatch.setattr("text2audiobook.briefing.machine_has_nvidia_gpu", lambda **_kw: True)
    monkeypatch.setattr(
        "text2audiobook.briefing.export_proposed_briefing",
        lambda *_args, **_kwargs: 0,
    )

    def boom(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("pipeline should not run")

    monkeypatch.setattr("text2audiobook.pipeline.run", boom)
    monkeypatch.setattr("text2audiobook.gpu.prepare_gpu_env", lambda: True)

    code = main(["--briefing", "--config", str(_config(tmp_path))])
    assert code == 0
    assert UNCHANGED_MESSAGE in capsys.readouterr().out


def test_briefing_cli_processes_only_the_markdown(monkeypatch, tmp_path: Path) -> None:
    markdown = _markdown(tmp_path, "# Proposed research sources\n\nShort intro.\n")
    m4b = tmp_path / "output" / "proposed_research_sources.m4b"
    m4b.parent.mkdir()
    m4b.write_bytes(b"audio")
    seen: dict[str, object] = {}

    monkeypatch.setattr("text2audiobook.briefing.machine_has_nvidia_gpu", lambda **_kw: True)
    monkeypatch.setattr(
        "text2audiobook.briefing.export_proposed_briefing",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr("text2audiobook.gpu.prepare_gpu_env", lambda: True)

    def fake_run(_config_obj: object, **kwargs: object) -> int:
        seen["source_paths"] = kwargs.get("source_paths")
        return 0

    monkeypatch.setattr("text2audiobook.pipeline.run", fake_run)
    code = main(["--briefing", "--config", str(_config(tmp_path))])
    assert code == 0
    assert seen["source_paths"] == [markdown.resolve()]
    fingerprint = briefing_fingerprint_file(tmp_path / "runs")
    digest = hashlib.sha256(markdown.read_bytes()).hexdigest()
    assert fingerprint.read_text(encoding="utf-8") == digest + "\n"


def test_briefing_cli_does_not_record_fingerprint_on_failure(
    monkeypatch, tmp_path: Path
) -> None:
    _markdown(tmp_path, "# Proposed research sources\n\nShort intro.\n")
    monkeypatch.setattr("text2audiobook.briefing.machine_has_nvidia_gpu", lambda **_kw: True)
    monkeypatch.setattr(
        "text2audiobook.briefing.export_proposed_briefing",
        lambda *_args, **_kwargs: 0,
    )
    monkeypatch.setattr("text2audiobook.gpu.prepare_gpu_env", lambda: True)
    monkeypatch.setattr("text2audiobook.pipeline.run", lambda *_args, **_kwargs: 1)

    code = main(["--briefing", "--config", str(_config(tmp_path))])
    assert code == 1
    assert briefing_fingerprint_file(tmp_path / "runs").is_file() is False
