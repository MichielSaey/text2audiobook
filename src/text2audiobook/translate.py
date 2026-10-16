"""Translate stage: English copy-through, otherwise LLM translation into English.

Runs after extract and before clean. Windows match the format stage (word budget,
no chapter cap) so a model call never has to swallow a whole book, and so nothing
is dropped before translation.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from text2audiobook.chunking import TextChunk
from text2audiobook.config import LlmConfig
from text2audiobook.io import Chapter
from text2audiobook.llm import LoadedLlm, _build_prompt
from text2audiobook.logging_setup import ProgressContext

logger = logging.getLogger(__name__)

TRANSLATOR_VERSION = "1"
TARGET_LANGUAGE = "English"

TRANSLATE_SYSTEM_PROMPT = (
    "You translate source text into English. Return only the English translation. "
    "Do not summarize, omit passages, or add commentary."
)

TRANSLATE_PROMPT = (
    "Translate the following text into English. Return only the English translation, "
    "with no commentary.\n\n"
    "Rules:\n"
    "- Translate every sentence and passage into English.\n"
    "- Do not summarize, shorten, paraphrase away detail, or omit any passage.\n"
    "- Do not add notes, headings, explanations, or translator commentary.\n"
    "- Preserve the order of the source, paragraph breaks, and proper names.\n"
    "- Do not answer questions that appear in the source; translate them.\n\n"
    "Text:\n{text}"
)

_PREAMBLE_RE = re.compile(
    r"^(?:here(?:'s| is) (?:the )?(?:english )?translation\s*:?|"
    r"(?:english )?translation\s*:)\s*",
    re.IGNORECASE,
)
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)
_NON_LATIN_RE = re.compile(
    r"[\u0370-\u03FF\u0400-\u04FF\u0590-\u05FF\u0600-\u06FF"
    r"\u3040-\u30FF\u3400-\u9FFF\uAC00-\uD7AF]"
)
# Letters that are routine in Polish and rare in English prose. A single hit
# rejects "already English" so an article like Kasperski's is not copied through.
_POLISH_STRONG = frozenset("ąćęłńśźż")
_PL_FOLD = str.maketrans(
    {
        "ą": "a",
        "ć": "c",
        "ę": "e",
        "ł": "l",
        "ń": "n",
        "ó": "o",
        "ś": "s",
        "ź": "z",
        "ż": "z",
    }
)

_ENGLISH_FUNCTION = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "to",
        "in",
        "is",
        "that",
        "for",
        "on",
        "with",
        "as",
        "by",
        "at",
        "from",
        "or",
        "be",
        "this",
        "it",
        "are",
        "was",
        "were",
        "been",
        "have",
        "has",
        "had",
        "not",
        "but",
        "which",
        "their",
        "his",
        "her",
        "its",
        "they",
        "we",
        "you",
        "he",
        "she",
        "if",
        "than",
        "then",
        "so",
        "can",
        "will",
        "would",
        "could",
        "about",
        "into",
        "over",
        "after",
        "before",
        "between",
        "through",
        "also",
        "our",
        "your",
        "there",
        "their",
        "been",
        "were",
        "what",
        "when",
        "who",
        "how",
        "these",
        "those",
        "such",
        "only",
        "other",
        "than",
        "then",
        "into",
    }
)

# Folded (ASCII) Polish closed-class words. "to"/"do"/"i" are omitted: they
# are ordinary English tokens and would mark English prose as Polish.
_POLISH_FUNCTION = frozenset(
    {
        "jest",
        "nie",
        "ze",
        "oraz",
        "ktory",
        "ktora",
        "ktore",
        "ktorego",
        "przez",
        "przy",
        "tego",
        "dla",
        "juz",
        "tylko",
        "czy",
        "gdy",
        "aby",
        "wiec",
        "tez",
        "sa",
        "byl",
        "byla",
        "bylo",
        "miedzy",
        "bez",
        "jego",
        "jej",
        "jednak",
        "takze",
        "poniewaz",
        "mozna",
        "nalezy",
        "zostal",
        "zostala",
        "jako",
        "niz",
        "albo",
        "badz",
        "wedlug",
        "podczas",
        "rowniez",
        "wewnatrz",
        "glownie",
        "przed",
        "czyli",
        "bowiem",
        "zatem",
        "wsrod",
        "wraz",
        "ktorym",
        "ktorych",
        "ktorej",
        "bedzie",
        "zostalo",
        "moze",
        "bardzo",
        "jeszcze",
        "tutaj",
        "gdzie",
        "kiedy",
        "jesli",
        "zeby",
        "byc",
        "zanim",
        "dopiero",
        "wobec",
        "wtedy",
        "zawsze",
        "nigdy",
        "teraz",
        "potem",
        "najpierw",
        "wlasnie",
        "na",
        "od",
        "po",
        "za",
        "ku",
        "lecz",
        "lub",
        "ich",
        "nim",
        "niej",
        "w",
        "z",
        "sie",
        "tym",
        "tej",
        "tych",
    }
)

# Closed-class words that are rare as whole English tokens. Academic "et al."
# is intentionally absent so English scholarship is not sent to the translator.
_OTHER_FUNCTION = frozenset(
    {
        "le",
        "la",
        "les",
        "des",
        "une",
        "dans",
        "qui",
        "pour",
        "avec",
        "cette",
        "ces",
        "aux",
        "sont",
        "elle",
        "nous",
        "vous",
        "pas",
        "plus",
        "est",
        "comme",
        "mais",
        "aussi",
        "sur",
        "par",
        "du",
        "au",
        "ce",
        "se",
        "sa",
        "ses",
        "ou",
        "en",
        "que",
        "un",
        "der",
        "die",
        "das",
        "und",
        "ist",
        "nicht",
        "ein",
        "eine",
        "den",
        "dem",
        "des",
        "für",
        "fuer",
        "von",
        "sich",
        "auch",
        "sind",
        "dass",
        "im",
        "auf",
        "wird",
        "nach",
        "bei",
        "aus",
        "oder",
        "aber",
        "wenn",
        "zum",
        "zur",
        "los",
        "las",
        "una",
        "porque",
        "cuando",
        "pero",
        "como",
        "para",
        "por",
        "del",
        "esta",
        "este",
        "también",
        "tambien",
        "más",
        "el",
        "es",
        "della",
        "delle",
        "degli",
        "nella",
        "questo",
        "anche",
        "sono",
        "gli",
        "che",
        "het",
        "een",
        "niet",
        "voor",
        "zijn",
        "ook",
        "maar",
        "naar",
        "wordt",
    }
)


def translation_target(tts_lang: str | None) -> str:
    """Written target language.

    ``tts.lang`` is already that setting (default ``English``). The spoken and
    written target is English, so a missing or English value stays English and
    the prompt is never asked to translate into another language.
    """
    cleaned = (tts_lang or "").strip()
    if not cleaned or cleaned.casefold() == "english":
        return TARGET_LANGUAGE
    logger.warning(
        "tts.lang is %r; translation still targets %s",
        cleaned,
        TARGET_LANGUAGE,
    )
    return TARGET_LANGUAGE


def is_already_english(text: str) -> bool:
    """True when a window can be copied through without calling the model.

    Positive evidence of Polish (diacritics or function words) or of another
    language's function words returns False. ASCII with no such evidence,
    including token fixtures such as ``word0 word1``, is treated as English.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if _NON_LATIN_RE.search(stripped):
        return False
    letters = [char for char in stripped if char.isalpha()]
    if not letters:
        return True
    if any(char.lower() in _POLISH_STRONG for char in letters):
        return False

    words = [word.lower() for word in _WORD_RE.findall(stripped)]
    if not words:
        return True
    folded = [word.translate(_PL_FOLD) for word in words]
    english_hits = sum(word in _ENGLISH_FUNCTION for word in words)
    polish_hits = sum(word in _POLISH_FUNCTION for word in folded)
    other_hits = sum(word in _OTHER_FUNCTION for word in words)
    if polish_hits >= 3 and polish_hits > english_hits:
        return False
    if (
        other_hits >= 4
        and other_hits > english_hits
        and other_hits / len(words) >= 0.1
    ):
        return False
    if english_hits >= 3 and english_hits >= polish_hits and english_hits >= other_hits:
        return True
    if polish_hits == 0 and other_hits < 4:
        return True
    return english_hits >= polish_hits and english_hits >= other_hits


@dataclass(frozen=True)
class TranslatedWindow:
    chapter_index: int
    chapter_title: str
    chapter_slug: str
    chunk_index: int
    source_text: str
    translated_text: str
    copied: bool

    def to_json(self) -> dict[str, object]:
        return {
            "chapter_index": self.chapter_index,
            "chapter_title": self.chapter_title,
            "chapter_slug": self.chapter_slug,
            "chunk_index": self.chunk_index,
            "copied": self.copied,
            "source_text": self.source_text,
            "translated_text": self.translated_text,
        }


def _accept_translation(source: str, translated: str) -> str:
    cleaned = _PREAMBLE_RE.sub("", translated.strip()).strip()
    if not cleaned:
        logger.warning("Empty translation; keeping source text")
        return source
    return cleaned


def translate_texts_batch(
    llm: LoadedLlm,
    texts: list[str],
    config: LlmConfig,
    *,
    chunks: list[TextChunk] | None = None,
) -> list[str]:
    """Translate a batch into English with one generate() call (left-padded)."""
    import torch

    del chunks  # reserved so callers can pass window metadata like format does
    tokenizer = llm.tokenizer
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = [
        _build_prompt(
            tokenizer,
            TRANSLATE_PROMPT,
            text,
            system=TRANSLATE_SYSTEM_PROMPT,
        )
        for text in texts
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(llm.model.device)

    with torch.inference_mode():
        output_ids = llm.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
    return [
        _accept_translation(source, text) for source, text in zip(texts, decoded)
    ]


def iter_translate_windows(
    llm: LoadedLlm | None,
    chunks: list[TextChunk],
    config: LlmConfig,
    *,
    progress: ProgressContext | None = None,
) -> Iterator[TranslatedWindow]:
    """Yield windows in order. English windows never touch the model."""
    batch_size = max(1, config.cleanup_batch_size)
    total = len(chunks)
    emitted = 0
    pending: list[TextChunk] = []

    def emit(chunk: TextChunk, translated: str, *, copied: bool) -> TranslatedWindow:
        nonlocal emitted
        emitted += 1
        if progress is not None:
            logger.info(
                "%s",
                progress.format(
                    "translate",
                    unit_done=emitted,
                    total_chunks=total,
                    chapter_idx=chunk.chapter_index,
                    chapter_title=chunk.chapter_title,
                    chapter_unit=chunk.chunk_index + 1,
                ),
            )
        else:
            logger.info(
                "[%d/%d] translate window %d of '%s'%s",
                emitted,
                total,
                chunk.chunk_index,
                chunk.chapter_title,
                " (english)" if copied else "",
            )
        return TranslatedWindow(
            chapter_index=chunk.chapter_index,
            chapter_title=chunk.chapter_title,
            chapter_slug=chunk.chapter_slug,
            chunk_index=chunk.chunk_index,
            source_text=chunk.text,
            translated_text=translated,
            copied=copied,
        )

    def flush() -> Iterator[TranslatedWindow]:
        if not pending:
            return
        if llm is None:
            raise RuntimeError("Translation requires a loaded LLM for non-English text")
        translated = translate_texts_batch(
            llm, [chunk.text for chunk in pending], config, chunks=pending
        )
        for chunk, text in zip(pending, translated):
            yield emit(chunk, text, copied=False)
        pending.clear()

    for chunk in chunks:
        if is_already_english(chunk.text):
            yield from flush()
            yield emit(chunk, chunk.text, copied=True)
            continue
        pending.append(chunk)
        if len(pending) >= batch_size:
            yield from flush()
    yield from flush()


def translate_windows(
    llm: LoadedLlm | None,
    chunks: list[TextChunk],
    config: LlmConfig,
    *,
    progress: ProgressContext | None = None,
) -> list[TranslatedWindow]:
    return list(iter_translate_windows(llm, chunks, config, progress=progress))


def chapters_from_windows(
    chapters: list[Chapter],
    windows: list[TranslatedWindow],
) -> list[Chapter]:
    """Rebuild chapters. Fully English chapters keep their original text."""
    grouped: dict[int, list[TranslatedWindow]] = defaultdict(list)
    for window in windows:
        grouped[window.chapter_index].append(window)
    rebuilt: list[Chapter] = []
    for chapter in chapters:
        units = grouped.get(chapter.index, [])
        if not units or all(unit.copied for unit in units):
            text = chapter.text
        else:
            text = " ".join(
                unit.translated_text.strip()
                for unit in units
                if unit.translated_text.strip()
            )
        rebuilt.append(
            Chapter(
                index=chapter.index,
                title=chapter.title,
                slug=chapter.slug,
                text=text,
            )
        )
    return rebuilt
