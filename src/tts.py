"""TTS using Coqui XTTS-v2. Handles Hinglish by treating the text as Hindi
(XTTS-v2 is robust to code-mixed Roman/Devanagari input)."""
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


def synth_line(
    text: str,
    out_wav: Path,
    speaker_wav: Path,
    cfg: dict,
) -> Path:
    tts = _get_tts(cfg["tts"]["xtts_model"])
    tts.tts_to_file(
        text=text,
        speaker_wav=str(speaker_wav),
        language=cfg["tts"]["language"],
        file_path=str(out_wav),
    )
    return out_wav


def synth_lines(
    lines: Iterable[str],
    workspace: Path,
    speaker_wav: Path,
    cfg: dict,
) -> list[Path]:
    out: list[Path] = []
    for i, text in enumerate(lines):
        wav = workspace / f"line_{i:02d}.wav"
        synth_line(text, wav, speaker_wav, cfg)
        out.append(wav)
    return out
