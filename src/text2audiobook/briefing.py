"""Trantor proposed-briefing export, GPU gate, and unchanged-file skip."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from text2audiobook.formats.markdown import book_title
from text2audiobook.io import slugify
from text2audiobook.stems import file_sha256

BRIEFING_FILENAME = "trantor-proposed.md"
FINGERPRINT_FILENAME = "briefing-fingerprint.txt"
NO_GPU_MESSAGE = "Briefing skipped because this machine has no GPU."
UNCHANGED_MESSAGE = "Briefing unchanged; skipped without loading models."

_NVIDIA_DEVICE = Path("/dev/nvidia0")
_NVIDIA_SMI_TIMEOUT_S = 30


def briefing_markdown_path(input_dir: Path) -> Path:
    return Path(input_dir) / BRIEFING_FILENAME


def briefing_fingerprint_file(runs_dir: Path) -> Path:
    return Path(runs_dir) / FINGERPRINT_FILENAME


def expected_briefing_m4b(markdown_path: Path, output_dir: Path) -> Path:
    """M4B path the markdown reader would write for this briefing."""
    content = markdown_path.read_text(encoding="utf-8")
    title = book_title(markdown_path, content)
    return Path(output_dir) / f"{slugify(title)}.m4b"


def write_briefing_fingerprint(markdown_path: Path, fingerprint_path: Path) -> None:
    """Store a sha256 of the briefing markdown."""
    path = Path(fingerprint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(file_sha256(markdown_path) + "\n", encoding="utf-8")


def should_skip_briefing(
    markdown_path: Path,
    *,
    fingerprint_path: Path,
    m4b_path: Path,
    force: bool = False,
) -> bool:
    """True when the stored hash matches the markdown and the M4B is already built."""
    if force:
        return False
    if not markdown_path.is_file() or not m4b_path.is_file() or not fingerprint_path.is_file():
        return False
    stored = fingerprint_path.read_text(encoding="utf-8").strip()
    if not stored:
        return False
    return stored == file_sha256(markdown_path)


def machine_has_nvidia_gpu(
    *,
    device: Path = _NVIDIA_DEVICE,
    nvidia_smi: Callable[[], int] | None = None,
) -> bool:
    """True when ``/dev/nvidia0`` exists or ``nvidia-smi`` exits 0.

    Missing device and a failing or absent ``nvidia-smi`` means no GPU.
    """
    if Path(device).exists():
        return True
    if nvidia_smi is None:
        return _nvidia_smi_status() == 0
    try:
        return nvidia_smi() == 0
    except OSError:
        return False


def _nvidia_smi_status() -> int:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return 1
    try:
        completed = subprocess.run(
            [executable],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 1
    return completed.returncode


def _briefing_out_argument(out_path: Path, cwd: Path | None) -> str:
    if cwd is None:
        return str(out_path)
    try:
        return out_path.resolve().relative_to(cwd.resolve()).as_posix()
    except ValueError:
        return str(out_path.resolve())


def export_proposed_briefing(
    out_path: Path,
    *,
    cwd: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
    which: Callable[[str], str | None] = shutil.which,
) -> int | None:
    """Run ``trantor briefing --status proposed`` when trantor is on PATH.

    Returns the process exit code, or None when the executable is absent.
    ``--out`` is ``data/input/trantor-proposed.md`` when ``cwd`` is the repo root.
    """
    executable = which("trantor")
    if not executable:
        return None
    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "briefing",
        "--status",
        "proposed",
        "--out",
        _briefing_out_argument(destination, cwd),
    ]
    kwargs: dict[str, Any] = {"check": False}
    if cwd is not None:
        kwargs["cwd"] = cwd
    completed = runner(command, **kwargs)
    return int(completed.returncode)
