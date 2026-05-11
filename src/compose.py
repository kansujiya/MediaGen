"""Stitch clips, voiceover, optional BGM, and burned-in captions into a final
9:16 mp4 using ffmpeg via moviepy."""
from __future__ import annotations

import random
from pathlib import Path

from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    CompositeAudioClip,
    concatenate_videoclips,
    afx,
)


def _pick_bgm(music_dir: Path) -> Path | None:
    if not music_dir.exists():
        return None
    tracks = [p for p in music_dir.iterdir() if p.suffix.lower() in {".mp3", ".wav", ".m4a"}]
    return random.choice(tracks) if tracks else None


def compose_reel(
    clip_paths: list[Path],
    voice_paths: list[Path],
    out_mp4: Path,
    cfg: dict,
    srt_path: Path | None = None,
) -> Path:
    """Each clip is paired with its line's voiceover. Clip is stretched/trimmed
    to match voiceover length so audio and visuals stay in sync."""
    if len(clip_paths) != len(voice_paths):
        raise ValueError("clip/voice count mismatch")

    segments = []
    for clip_path, voice_path in zip(clip_paths, voice_paths):
        v = VideoFileClip(str(clip_path))
        a = AudioFileClip(str(voice_path))
        if v.duration < a.duration:
            # Loop the clip if voice is longer.
            n = int(a.duration // v.duration) + 1
            v = concatenate_videoclips([v] * n).subclip(0, a.duration)
        else:
            v = v.subclip(0, a.duration)
        v = v.set_audio(a)
        segments.append(v)

    final = concatenate_videoclips(segments, method="compose")

    bgm_path = _pick_bgm(Path("assets/music"))
    if bgm_path is not None:
        bgm = AudioFileClip(str(bgm_path)).fx(
            afx.volumex, 10 ** (cfg["audio"]["bgm_volume_db"] / 20)
        )
        if bgm.duration < final.duration:
            bgm = afx.audio_loop(bgm, duration=final.duration)
        bgm = bgm.subclip(0, final.duration)
        final = final.set_audio(CompositeAudioClip([final.audio, bgm]))

    tmp = out_mp4.with_suffix(".raw.mp4")
    final.write_videofile(
        str(tmp),
        codec="libx264",
        audio_codec="aac",
        fps=cfg["reel"]["fps"],
        preset="medium",
        threads=4,
    )

    if srt_path and cfg["captions"]["burn_in"]:
        _burn_subs(tmp, srt_path, out_mp4)
        tmp.unlink(missing_ok=True)
    else:
        tmp.rename(out_mp4)

    return out_mp4


def _burn_subs(in_mp4: Path, srt: Path, out_mp4: Path) -> None:
    import subprocess
    style = (
        "FontName=DejaVu Sans,FontSize=14,PrimaryColour=&H00FFFFFF,"
        "OutlineColour=&H80000000,BorderStyle=3,Outline=2,Shadow=0,"
        "Alignment=2,MarginV=80"
    )
    cmd = [
        "ffmpeg", "-y", "-i", str(in_mp4),
        "-vf", f"subtitles={srt}:force_style='{style}'",
        "-c:a", "copy", str(out_mp4),
    ]
    subprocess.run(cmd, check=True)
