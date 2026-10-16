"""Translate stage: non-English text is translated before clean; English is copied."""

import json
from pathlib import Path

from text2audiobook.cli import main
from text2audiobook.config import load_config
from text2audiobook.pipeline import process_source, run
from text2audiobook.stems import BookStems, load_clean_sections
from text2audiobook.tracking import RunTracker
from text2audiobook.translate import (
    TRANSLATE_PROMPT,
    is_already_english,
    translation_target,
)

_POLISH = (
    "Post-metal jest nazwą hybrydowej praktyki wewnątrz współczesnego heavy metalu, "
    "a nie pojedynczą techniką wykonawczą. Kasperski stwierdza, że krytycy zwykle "
    "traktują ten prąd jako osobny gatunek, lecz jego postmodernistyczna nierówność "
    "czyni go także estetycznym nurtem całej sceny. Wczesne nagrania powstały na "
    "początku lat dziewięćdziesiątych, głównie w muzyce amerykańskiej, zanim sama "
    "nazwa weszła do szerokiego obiegu i zanim krytyka opisała długie formy, "
    "narastanie, powtórzenie oraz kontrast między gęstym hałasem a cichym fragmentem."
)
_POLISH_ASCII = (
    "Post-metal jest nazwa hybrydowej praktyki wewnatrz heavy metalu a nie "
    "pojedyncza technika. Kasperski stwierdza ze krytycy traktuja ten prad jako "
    "osobny gatunek lecz jego nierownosc czyni go takze estetycznym nurtem. "
    "Wczesne nagrania powstaly na poczatku lat glownie w muzyce amerykanskiej "
    "zanim sama nazwa weszla do obiegu i zanim krytyka opisala dlugie formy."
)
_ENGLISH = (
    "Post-metal is a name for a hybrid practice inside recent heavy metal, "
    "and not a single performance technique. Critics usually treat the current "
    "as its own genre, but its unevenness also makes it an aesthetic tendency "
    "across the wider scene. The early records were made in the early nineteen "
    "nineties, mainly in American music, before the name was in wide use and "
    "before critics described the long forms, the slow build, the repetition, "
    "and the contrast between a dense wall of noise and a quiet passage."
)
_TRANSLATED = "The translated article keeps every sentence of the Polish source."


def test_translate_prompt_only_translates_into_english() -> None:
    lowered = TRANSLATE_PROMPT.lower()
    assert "into english" in lowered
    assert "summar" in lowered
    assert "omit" in lowered
    assert "commentary" in lowered
    assert "{text}" in TRANSLATE_PROMPT


def test_translation_target_reuses_english_tts_lang() -> None:
    assert translation_target("English") == "English"
    assert translation_target(" english ") == "English"
    assert translation_target(None) == "English"
    assert translation_target("") == "English"
    assert translation_target("Chinese") == "English"


def test_language_heuristic_rejects_polish_and_keeps_english() -> None:
    assert is_already_english(_POLISH) is False
    assert is_already_english(_POLISH_ASCII) is False
    assert is_already_english(_ENGLISH) is True
    assert is_already_english(" ".join(f"word{i}" for i in range(40))) is True
    assert is_already_english(
        "Kasperski treats post-metal as both a genre name and a hybrid current."
    ) is True
    french = (
        "Le post-metal est une pratique hybride dans le heavy metal recent et "
        "les critiques traitent ce courant comme un genre a part mais les "
        "premiers disques sont apparus dans la musique americaine."
    )
    assert is_already_english(french) is False


def _config(tmp_path: Path) -> object:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {
                    "input_dir": "input",
                    "staging_dir": "staging",
                    "output_dir": "output",
                    "runs_dir": "runs",
                },
                "selection": {"keep_chapter_indices": [0]},
                "llm": {"cleanup": False, "direction": False},
                "tts": {"device": "cpu", "lang": "English"},
            }
        ),
        encoding="utf-8",
    )
    return load_config(config_path)


def _markdown(tmp_path: Path, body: str) -> Path:
    source = tmp_path / "input" / "article.md"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(f"# Article\n\n{body}\n", encoding="utf-8")
    return source


def test_polish_text_is_translated_before_clean(tmp_path: Path, monkeypatch) -> None:
    source = _markdown(tmp_path, _POLISH)
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_translate(_llm, texts, _config, **_kwargs):
        calls.extend(texts)
        return [_TRANSLATED for _ in texts]

    monkeypatch.setattr("text2audiobook.translate.translate_texts_batch", fake_translate)
    monkeypatch.setattr("text2audiobook.pipeline.load_llm", lambda *_a, **_k: object())
    monkeypatch.setattr("text2audiobook.pipeline.unload_llm", lambda *_a, **_k: None)

    tracker = RunTracker(config.paths.runs_dir, config.to_dict())
    record = tracker.start_book(source)
    process_source(
        source,
        config,
        tracker=tracker,
        record=record,
        stages=("extract", "translate", "clean"),
        tts_device="cpu",
    )
    assert calls
    assert any("Kasperski" in text and "stwierdza" in text for text in calls)
    translated_calls = len(calls)

    stems = BookStems(next(p for p in config.paths.staging_dir.iterdir() if p.is_dir()))
    assert stems.translate_manifest.exists()
    sections = load_clean_sections(stems)
    assert sections
    assert _TRANSLATED in sections[0].text
    assert "stwierdza" not in sections[0].text

    record = tracker.start_book(source)
    process_source(
        source,
        config,
        tracker=tracker,
        record=record,
        stages=("translate",),
        tts_device="cpu",
    )
    assert len(calls) == translated_calls


def test_english_text_skips_translation(tmp_path: Path, monkeypatch) -> None:
    source = _markdown(tmp_path, _ENGLISH)
    config = _config(tmp_path)
    calls: list[str] = []

    def fail_translate(*_a, **_k):
        calls.append("translate")
        raise AssertionError("English text must not call the translator")

    monkeypatch.setattr("text2audiobook.translate.translate_texts_batch", fail_translate)
    monkeypatch.setattr("text2audiobook.pipeline.load_llm", fail_translate)

    tracker = RunTracker(config.paths.runs_dir, config.to_dict())
    record = tracker.start_book(source)
    process_source(
        source,
        config,
        tracker=tracker,
        record=record,
        stages=("extract", "translate", "clean"),
        tts_device="cpu",
    )
    assert calls == []
    stems = BookStems(next(p for p in config.paths.staging_dir.iterdir() if p.is_dir()))
    sections = load_clean_sections(stems)
    assert "mainly in American music" in sections[0].text
    windows = stems.translate_windows_jsonl.read_text(encoding="utf-8")
    assert '"copied": true' in windows


def _package(tmp_path: Path, body: str) -> Path:
    package = tmp_path / "pkg"
    source = package / "sources" / "demo2020note"
    source.mkdir(parents=True)
    (source / "preface.md").write_text("A short preface.\n", encoding="utf-8")
    (source / "content.md").write_text(f"# Body chapter\n\n{body}\n", encoding="utf-8")
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
    return package


def test_package_content_is_translated_and_sections_stay_separate(
    tmp_path: Path, monkeypatch
) -> None:
    package = _package(tmp_path, _POLISH)
    config = _config(tmp_path)
    calls: list[str] = []

    def fake_translate(_llm, texts, _config, **_kwargs):
        calls.extend(texts)
        return [_TRANSLATED for _ in texts]

    monkeypatch.setattr("text2audiobook.translate.translate_texts_batch", fake_translate)
    monkeypatch.setattr("text2audiobook.pipeline.load_llm", lambda *_a, **_k: object())
    monkeypatch.setattr("text2audiobook.pipeline.unload_llm", lambda *_a, **_k: None)

    assert (
        run(
            config,
            stages=("extract", "translate", "clean"),
            package_path=package,
        )
        == 0
    )
    assert calls
    assert all("stwierdza" in text for text in calls)
    assert all("short preface" not in text for text in calls)
    assert all("takeaway" not in text.lower() for text in calls)

    stems = BookStems(config.paths.staging_dir / "demo_package")
    sections = load_clean_sections(stems)
    titles: list[str] = []
    for section in sections:
        if section.chapter_title not in titles:
            titles.append(section.chapter_title)
    assert titles == ["Preface", "Body chapter", "Key takeaways"]
    by_title = {section.chapter_title: section.text for section in sections}
    assert "A short preface." in by_title["Preface"]
    assert _TRANSLATED in by_title["Body chapter"]
    assert "stwierdza" not in by_title["Body chapter"]
    assert "One short takeaway." in by_title["Key takeaways"]


def test_translate_flag_is_recognized() -> None:
    assert main(["--list-voices", "--translate"]) == 0
