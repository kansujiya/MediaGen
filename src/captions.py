"""Word-level captions via faster-whisper. Outputs an SRT file the composer
can burn into the final video."""
from __future__ import annotations

from pathlib import Path
from datetime import timedelta


def _ts(seconds: float) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    total = int(td.total_seconds())
    ms = int((seconds - total) * 1000)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def transcribe_to_srt(audio_path: Path, out_srt: Path, cfg: dict) -> Path:
    from faster_whisper import WhisperModel

    model = WhisperModel(
        cfg["captions"]["whisper_model"],
        device="auto",
        compute_type="auto",
    )
    segments, _ = model.transcribe(str(audio_path), word_timestamps=True, language=None)

    max_chars = cfg["captions"]["max_chars_per_line"]
    lines: list[tuple[float, float, str]] = []
    buf: list = []          # list[(start, end, word)]
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        start = buf[0][0]
        end = buf[-1][1]
        text = " ".join(w for _, _, w in buf).strip()
        lines.append((start, end, text))
        buf = []
        buf_len = 0

    for seg in segments:
        for w in seg.words or []:
            word = w.word.strip()
            if not word:
                continue
            if buf_len + len(word) + 1 > max_chars and buf:
                flush()
            buf.append((w.start, w.end, word))
            buf_len += len(word) + 1
    flush()

    with open(out_srt, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(lines, 1):
            f.write(f"{i}\n{_ts(start)} --> {_ts(end)}\n{text}\n\n")
    return out_srt
