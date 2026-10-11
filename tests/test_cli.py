"""CLI stage-flag selection."""

import argparse
from pathlib import Path

from text2audiobook.cli import _selected_stages, explicit_sources, main


def _ns(**flags: bool) -> argparse.Namespace:
    base = {name: False for name in ("extract", "clean", "format", "speak")}
    base.update(flags)
    return argparse.Namespace(**base)


def test_selected_stages_defaults_to_full_pipeline() -> None:
    assert _selected_stages(_ns()) is None


def test_selected_stages_combinable_and_ordered() -> None:
    assert _selected_stages(_ns(speak=True, format=True)) == ["format", "speak"]
    assert _selected_stages(_ns(clean=True)) == ["clean"]
    assert _selected_stages(_ns(extract=True, speak=True, clean=True, format=True)) == [
        "extract",
        "clean",
        "format",
        "speak",
    ]


def test_explicit_sources_input_is_one_file() -> None:
    chosen = Path("notes.md")
    assert explicit_sources(chosen) == [chosen]
    assert explicit_sources(chosen, [Path("other.url")]) == [chosen]


def test_explicit_sources_keeps_url_list_and_folder_scan() -> None:
    urls = [Path("clip.url")]
    assert explicit_sources(None, urls) == urls
    assert explicit_sources(None, []) is None
    assert explicit_sources(None, None) is None


def test_missing_input_exits_before_gpu(monkeypatch, tmp_path: Path) -> None:
    def boom() -> bool:
        raise AssertionError("GPU prep should not run")

    monkeypatch.setattr("text2audiobook.gpu.prepare_gpu_env", boom)
    code = main(["--input", str(tmp_path / "missing.md")])
    assert code == 1


def test_input_flag_passes_only_that_file(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Title\n\nhello\n", encoding="utf-8")
    seen: dict[str, object] = {}

    def fake_run(_config: object, **kwargs: object) -> int:
        seen["source_paths"] = kwargs.get("source_paths")
        return 0

    monkeypatch.setattr("text2audiobook.gpu.prepare_gpu_env", lambda: True)
    monkeypatch.setattr("text2audiobook.pipeline.run", fake_run)
    code = main(["--input", str(source)])
    assert code == 0
    assert seen["source_paths"] == [source]
