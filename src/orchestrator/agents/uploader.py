"""Uploader: pushes the final mp4 to YouTube Shorts using existing helper."""
from __future__ import annotations

from pathlib import Path

from ...youtube_upload import upload_short


def run(cfg: dict, script: dict, video_path: str) -> dict:
    if not cfg["upload"]["youtube"]["enabled"]:
        return {"skipped": True, "reason": "youtube upload disabled in config"}

    tags = list({
        *script.get("hashtags", []),
        *cfg["upload"]["youtube"]["tags_extra"],
    })
    tags = [t.lstrip("#") for t in tags]
    description = (
        " ".join(ln["text"] for ln in script["lines"])
        + "\n\n"
        + " ".join(f"#{t}" for t in tags)
    )
    vid_id = upload_short(
        Path(video_path),
        title=script["title"],
        description=description,
        tags=tags,
        cfg=cfg,
    )
    return {
        "youtube_video_id": vid_id,
        "url": f"https://youtube.com/shorts/{vid_id}",
    }
