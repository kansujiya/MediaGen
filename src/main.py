"""Orchestrator: topic -> script -> voice -> stills -> clips -> captions -> mp4 -> upload.

Examples:
    python -m src.main --genre motivation --topic "Failure is data, not identity"
    python -m src.main --genre motivation --topic "..." --no-upload
    python -m src.main --genre motivation --topic "..." --video-backend image_kenburns
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import (
    load_config, setup_logging, slugify, workspace_dir, output_dir, save_json
)
from .script_gen import generate_script
from .tts import synth_lines
from .image_gen import generate_shot
from .video_gen import make_clip
from .compose import compose_reel
from .captions import transcribe_to_srt
from .youtube_upload import upload_short


def run(topic: str, genre: str, do_upload: bool, video_backend: str | None) -> Path:
    log = setup_logging()
    cfg = load_config(genre)
    if video_backend:
        cfg["video"]["pipeline"] = video_backend

    log.info("Generating script for: %s", topic)
    script = generate_script(topic, cfg)
    reel_id = slugify(script["title"])
    ws = workspace_dir(reel_id)
    save_json(script, ws / "script.json")
    log.info("Script -> %s lines, title=%r", len(script["lines"]), script["title"])

    speaker_wav = Path(cfg["_genre"]["character"]["voice_ref_wav"])
    if not speaker_wav.exists():
        raise FileNotFoundError(
            f"Voice reference wav missing: {speaker_wav}. "
            f"Record/download a 6-10s clean clip and place it there."
        )

    log.info("Synthesizing voiceover…")
    voice_paths = synth_lines(
        [ln["text"] for ln in script["lines"]], ws, speaker_wav, cfg
    )

    log.info("Generating per-line stills…")
    shot_pngs = []
    for i, ln in enumerate(script["lines"]):
        out = ws / f"shot_{i:02d}.png"
        generate_shot(ln["visual_prompt"], out, cfg, seed=1000 + i)
        shot_pngs.append(out)

    log.info("Animating stills (%s)…", cfg["video"]["pipeline"])
    clip_paths = []
    for i, (png, ln) in enumerate(zip(shot_pngs, script["lines"])):
        out = ws / f"clip_{i:02d}.mp4"
        # Duration here is a hint for the Ken Burns fallback; LTX/CogVideoX
        # produce a fixed number of frames and compose.py syncs to voice length.
        make_clip(png, ln["visual_prompt"], out, duration_s=6.0, cfg=cfg)
        clip_paths.append(out)

    log.info("Composing final reel…")
    raw_mp4 = ws / "reel_raw.mp4"
    compose_reel(clip_paths, voice_paths, raw_mp4, cfg, srt_path=None)

    srt_path = None
    if cfg["captions"]["enabled"]:
        log.info("Transcribing captions…")
        srt_path = ws / "captions.srt"
        transcribe_to_srt(raw_mp4, srt_path, cfg)

    final_mp4 = output_dir() / f"{reel_id}.mp4"
    compose_reel(clip_paths, voice_paths, final_mp4, cfg, srt_path=srt_path)
    log.info("Final: %s", final_mp4)

    if do_upload and cfg["upload"]["youtube"]["enabled"]:
        log.info("Uploading to YouTube…")
        tags = list({*script.get("hashtags", []), *cfg["upload"]["youtube"]["tags_extra"]})
        tags = [t.lstrip("#") for t in tags]
        vid_id = upload_short(
            final_mp4,
            title=script["title"],
            description=" ".join(ln["text"] for ln in script["lines"])
                       + "\n\n" + " ".join(f"#{t}" for t in tags),
            tags=tags,
            cfg=cfg,
        )
        log.info("Uploaded: https://youtube.com/shorts/%s", vid_id)

    return final_mp4


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topic", required=True)
    p.add_argument("--genre", default="motivation")
    p.add_argument("--no-upload", action="store_true")
    p.add_argument("--video-backend", choices=["ltx", "cogvideox2b", "image_kenburns"])
    args = p.parse_args()
    run(args.topic, args.genre, do_upload=not args.no_upload, video_backend=args.video_backend)


if __name__ == "__main__":
    main()
