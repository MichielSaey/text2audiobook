"""Command-line entry point: folder scan, one --input file, --url, or --briefing."""

import argparse
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from text2audiobook.config import DEFAULT_CONFIG_PATH
from text2audiobook.stems import PIPELINE_STAGES


def _selected_stages(args: argparse.Namespace) -> list[str] | None:
    """Return requested stage names, or None for the full default pipeline."""
    selected = [name for name in PIPELINE_STAGES if getattr(args, name, False)]
    return selected or None


def explicit_sources(
    input_path: Path | None,
    url_paths: Sequence[Path] | None = None,
) -> list[Path] | None:
    """One ``--input`` file, or ``--url`` files. None scans the input folder."""
    if input_path is not None:
        return [input_path]
    if url_paths:
        return list(url_paths)
    return None


def _run_briefing(args: argparse.Namespace) -> int:
    """Export the proposed Trantor briefing and speak that markdown file."""
    from text2audiobook import briefing

    if not briefing.machine_has_nvidia_gpu():
        print(briefing.NO_GPU_MESSAGE)
        return 0

    from text2audiobook.config import load_config

    config = load_config(args.config)
    config_path = config.config_path
    tool_root = config_path.parent if config_path is not None else Path.cwd()
    markdown_path = briefing.briefing_markdown_path(config.paths.input_dir)
    exported = briefing.export_proposed_briefing(markdown_path, cwd=tool_root)
    if exported is None:
        print("error: trantor executable not found on PATH", file=sys.stderr)
        return 1
    if exported != 0:
        print(f"error: trantor briefing exited {exported}", file=sys.stderr)
        return exported
    if not markdown_path.is_file():
        print(
            f"error: briefing markdown was not written: {markdown_path}",
            file=sys.stderr,
        )
        return 1

    m4b_path = briefing.expected_briefing_m4b(markdown_path, config.paths.output_dir)
    fingerprint_path = briefing.briefing_fingerprint_file(config.paths.runs_dir)
    if briefing.should_skip_briefing(
        markdown_path,
        fingerprint_path=fingerprint_path,
        m4b_path=m4b_path,
        force=args.force,
    ):
        print(briefing.UNCHANGED_MESSAGE)
        return 0

    from text2audiobook.gpu import prepare_gpu_env

    prepare_gpu_env()

    from text2audiobook.logging_setup import setup_logging
    from text2audiobook.pipeline import run

    setup_logging()
    stages = _selected_stages(args)
    code = run(
        config,
        source_paths=[markdown_path],
        stages=stages,
        force=args.force,
        voice=args.voice,
        speak_footnote_cues=args.footnote_cues,
    )
    if (
        code == 0
        and (stages is None or stages == list(PIPELINE_STAGES))
        and markdown_path.is_file()
        and m4b_path.is_file()
    ):
        briefing.write_briefing_fingerprint(markdown_path, fingerprint_path)
    return code


def _run_batch_vram_calibration(config_path: Path | None) -> int:
    """Load Base TTS, run VRAM batch calibration, print fitted knobs, exit."""
    from text2audiobook.batch_vram import run_batch_vram_calibration
    from text2audiobook.config import load_config
    from text2audiobook.gpu import prepare_gpu_env
    from text2audiobook.logging_setup import setup_logging
    from text2audiobook.tts import (
        is_base,
        load_tts,
        unload_tts,
    )

    prepare_gpu_env()
    setup_logging()
    config = load_config(config_path)
    tts = config.tts

    if not is_base(tts.model_id):
        print(
            "error: --calibrate-tts-batch requires a Base (voice-clone) model_id "
            f"(got {tts.model_id!r})",
            file=sys.stderr,
        )
        return 2
    if not tts.ref_audio:
        print("error: tts.ref_audio is required for batch VRAM calibration", file=sys.stderr)
        return 2
    ref_audio = Path(tts.ref_audio)
    if not ref_audio.is_file():
        print(f"error: tts.ref_audio not found: {ref_audio}", file=sys.stderr)
        return 2
    if not tts.x_vector_only and not (tts.ref_text and tts.ref_text.strip()):
        print(
            "error: tts.ref_text is required unless tts.x_vector_only is true",
            file=sys.stderr,
        )
        return 2

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = config.paths.runs_dir / f"{stamp}_batch_vram"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = load_tts(tts)

    def _hard_unload(current) -> None:
        unload_tts(current)
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            torch.cuda.empty_cache()
        gc.collect()

    def reload_model():
        """Fully drop the old model; return (model, None) and use ref_audio per trial."""
        nonlocal model
        _hard_unload(model)
        model = None
        model = load_tts(tts)
        # Skip rebuild of voice_clone_prompt here — encoding the ref right after an
        # OOM often OOMs again. synthesize uses ref_audio when prompt is None.
        return model, None

    try:
        result = run_batch_vram_calibration(
            model,
            voice=tts.voice,
            language=tts.lang,
            model_id=tts.model_id,
            voice_clone_prompt=None,
            ref_audio=ref_audio,
            ref_text=tts.ref_text,
            x_vector_only=tts.x_vector_only,
            out_dir=out_dir,
            reload_model=reload_model,
        )
    finally:
        if model is not None:
            unload_tts(model)

    print()
    print("Suggested config.tts values:")
    print(f'  "batch_max_pad_chars": {result["budget_chars"]},')
    print(f'  "batch_vram_overhead": {result["overhead_chars"]},')
    print(f'  "batch_max_items": {result["max_items_suggested"]},')
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="text2audiobook",
        description=(
            "Convert supported text sources (EPUB, PDF, Markdown, HTML, .url, "
            "or a Trantor package) to M4B audiobooks."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"Path to a config JSON file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--url",
        action="append",
        default=[],
        metavar="URL",
        help=(
            "Fetch and convert a web page (repeatable). "
            "When set, the input folder is not scanned."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        metavar="PATH",
        help="Convert this one file. When set, the input folder is not scanned.",
    )
    parser.add_argument(
        "--briefing",
        action="store_true",
        help=(
            "Export the Trantor proposed briefing "
            "(trantor briefing --status proposed) and convert that markdown file. "
            "Skips model load when the briefing hash and M4B are unchanged. "
            "Exits 0 without TTS when this machine has no GPU."
        ),
    )
    parser.add_argument(
        "--package",
        type=Path,
        default=None,
        metavar="DIR",
        help=(
            "Read a Trantor package directory or package.json. "
            "When set, the input folder is not scanned."
        ),
    )
    parser.add_argument(
        "--extract",
        action="store_true",
        help="Run the extract stage (combinable with other stage flags).",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help=(
            "Run the translate stage (combinable; requires extract stem if extract "
            "is skipped). English text is copied through; other languages are "
            "translated into English."
        ),
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Run the clean stage (combinable; requires translate stem if translate is skipped).",
    )
    parser.add_argument(
        "--format",
        action="store_true",
        dest="format",
        help="Run the format stage (combinable; requires clean stem if clean is skipped).",
    )
    parser.add_argument(
        "--speak",
        action="store_true",
        help=(
            "Run the speak stage (combinable; requires format stem if format is skipped). "
            "Speak-only does not load the LLM."
        ),
    )
    parser.add_argument(
        "--calibrate-tts-batch",
        action="store_true",
        help=(
            "Measure TTS batch VRAM frontiers on the voice-clone path, fit "
            "batch_max_pad_chars / batch_vram_overhead, write calibration.json, "
            "and exit (does not process books)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redo requested stages even when stems or the M4B already exist.",
    )
    parser.add_argument(
        "--voice",
        default=None,
        help=(
            "CustomVoice speaker (e.g. Serena, Ryan, Aiden), 'random' for the "
            "CustomVoice pool, 'cloned' for Base voice clone (uses tts.ref_audio), "
            "or 'designed' for VoiceDesign. Content language stays from config "
            "tts.lang."
        ),
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help=(
            "Print CustomVoice speakers plus Base/VoiceDesign notes, then exit "
            "(no GPU / LLM load)."
        ),
    )
    cues = parser.add_mutually_exclusive_group()
    cues.add_argument(
        "--footnote-cues",
        action="store_true",
        dest="footnote_cues",
        help="Speak footnote start/end markers (overrides config).",
    )
    cues.add_argument(
        "--no-footnote-cues",
        action="store_false",
        dest="footnote_cues",
        help="Omit spoken footnote markers (overrides config).",
    )
    parser.set_defaults(footnote_cues=None)
    args = parser.parse_args(argv)

    if args.list_voices:
        from text2audiobook.voices import format_voice_table

        print(format_voice_table())
        return 0

    if args.briefing and (args.input is not None or args.url or args.calibrate_tts_batch):
        parser.error(
            "--briefing cannot be combined with --input, --url, or --calibrate-tts-batch"
        )
    if args.input is not None and args.url:
        parser.error("--input and --url cannot be used together")

    if args.calibrate_tts_batch:
        return _run_batch_vram_calibration(args.config)

    if args.briefing:
        return _run_briefing(args)

    if args.input is not None and not args.input.is_file():
        print(f"error: --input is not a file: {args.input}", file=sys.stderr)
        return 1
    if args.package is not None and args.url:
        print("error: --package cannot be combined with --url", file=sys.stderr)
        return 2

    from text2audiobook.gpu import prepare_gpu_env

    prepare_gpu_env()

    from text2audiobook.config import load_config
    from text2audiobook.formats.url import write_url_source
    from text2audiobook.logging_setup import setup_logging
    from text2audiobook.pipeline import run

    setup_logging()
    config = load_config(args.config)

    url_paths: list[Path] | None = None
    if args.url:
        url_dir = config.paths.staging_dir / "_cli_urls"
        url_paths = [
            write_url_source(url_dir, url, stem=f"cli_{index}")
            for index, url in enumerate(args.url, start=1)
        ]

    return run(
        config,
        source_paths=explicit_sources(args.input, url_paths),
        stages=_selected_stages(args),
        force=args.force,
        voice=args.voice,
        speak_footnote_cues=args.footnote_cues,
        package_path=args.package,
    )


if __name__ == "__main__":
    sys.exit(main())
