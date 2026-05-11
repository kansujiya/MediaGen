"""TTS using Coqui XTTS-v2.

Two modes, picked automatically:

  1. Voice cloning  — if `speaker_wav` is a real file, XTTS clones it. This
     is how you lock a per-character voice (preferred).
  2. Built-in speaker — if no clone target, XTTS uses one of its bundled
     multilingual speakers. Quality is still good; accent isn't Indian but
     it lets a first run produce a finished mp4 with zero manual assets.

XTTS-v2 handles Hinglish well when `language="hi"` is set — it reads Roman
Hindi tokens naturally.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch


_tts_singleton = None


def _get_tts(model_name: str):
    global _tts_singleton
    if _tts_singleton is None:
        from TTS.api import TTS  # lazy import — heavy
        _tts_singleton = TTS(model_name).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    return _tts_singleton


def _has_clone_target(speaker_wav: Path | None) -> bool:
    return speaker_wav is not None and Path(speaker_wav).exists()


def synth_line(
    text: str,
    out_wav: Path,
    speaker_wav: Path | None,
    cfg: dict,
) -> Path:
    tts = _get_tts(cfg["tts"]["xtts_model"])
    kwargs = dict(
        text=text,
        language=cfg["tts"]["language"],
        file_path=str(out_wav),
    )
    if _has_clone_target(speaker_wav):
        kwargs["speaker_wav"] = str(speaker_wav)
    else:
        # XTTS-v2 ships ~50 preset speakers; pick one and stick to it for the
        # whole reel so the "character" stays consistent within a video.
        kwargs["speaker"] = cfg["tts"].get("fallback_speaker", "Ana Florence")
    tts.tts_to_file(**kwargs)
    return out_wav


def synth_lines(
    lines: Iterable[str],
    workspace: Path,
    speaker_wav: Path | None,
    cfg: dict,
) -> list[Path]:
    out: list[Path] = []
    for i, text in enumerate(lines):
        wav = workspace / f"line_{i:02d}.wav"
        synth_line(text, wav, speaker_wav, cfg)
        out.append(wav)
    return out
