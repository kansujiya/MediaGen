"""Producer: takes a finalised script dict and runs the existing media
pipeline (TTS -> stills -> clips -> compose -> captions). Returns the final
mp4 path."""
from __future__ import annotations

from pathlib import Path

from ...compose import compose_reel
from ...captions import transcribe_to_srt
from ...image_gen import generate_shot
from ...tts import synth_lines
from ...utils import output_dir, slugify, save_json, workspace_dir
from ...video_gen import make_clip


def run(cfg: dict, script: dict) -> dict:
    reel_id = slugify(script["title"])
    ws = workspace_dir(reel_id)
    save_json(script, ws / "script.json")

    speaker_path = Path(cfg["_genre"]["character"]["voice_ref_wav"])
    speaker_wav = speaker_path if speaker_path.exists() else None

    voice_paths = synth_lines(
        [ln["text"] for ln in script["lines"]], ws, speaker_wav, cfg
    )

    shot_pngs = []
    for i, ln in enumerate(script["lines"]):
        out = ws / f"shot_{i:02d}.png"
        generate_shot(ln["visual_prompt"], out, cfg, seed=1000 + i)
        shot_pngs.append(out)

    total_sec = cfg["_genre"]["script"].get("total_seconds", 15)
    clip_dur = max(3.0, total_sec / max(len(script["lines"]), 1))
    clip_paths = []
    for i, (png, ln) in enumerate(zip(shot_pngs, script["lines"])):
        out = ws / f"clip_{i:02d}.mp4"
        make_clip(png, ln["visual_prompt"], out, duration_s=clip_dur, cfg=cfg)
        clip_paths.append(out)

    raw_mp4 = ws / "reel_raw.mp4"
    compose_reel(clip_paths, voice_paths, raw_mp4, cfg, srt_path=None)

    srt_path = None
    if cfg["captions"]["enabled"]:
        srt_path = ws / "captions.srt"
        transcribe_to_srt(raw_mp4, srt_path, cfg)

    final_mp4 = output_dir() / f"{reel_id}.mp4"
    compose_reel(clip_paths, voice_paths, final_mp4, cfg, srt_path=srt_path)

    return {
        "reel_id": reel_id,
        "final_video_path": str(final_mp4),
        "workspace": str(ws),
        "num_shots": len(shot_pngs),
    }
