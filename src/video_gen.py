"""Image -> short video clip. Three backends:

  - ltx              : Lightricks LTX-Video (fast, 8GB-friendly with offload)
  - cogvideox2b      : THUDM/CogVideoX-2b (good motion, slightly heavier)
  - image_kenburns   : no GPU video gen, just pan/zoom the still frame.
                       Use this as a fallback or when VRAM is too tight.

The first two are image-to-video conditioned on the per-shot still frame so
the character stays consistent.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import numpy as np
from PIL import Image


def _kenburns(image_path: Path, out_mp4: Path, duration_s: float, fps: int) -> Path:
    """Cheap CPU fallback: subtle zoom-in on the still."""
    import imageio.v2 as imageio

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    total = max(1, int(duration_s * fps))
    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=8)
    for f in range(total):
        t = f / max(1, total - 1)
        zoom = 1.0 + 0.08 * t
        cw, ch = int(W / zoom), int(H / zoom)
        x = (W - cw) // 2
        y = (H - ch) // 2
        frame = img.crop((x, y, x + cw, y + ch)).resize((W, H), Image.LANCZOS)
        writer.append_data(np.array(frame))
    writer.close()
    return out_mp4


def _ltx_video(image_path: Path, prompt: str, out_mp4: Path, cfg: dict) -> Path:
    from diffusers import LTXImageToVideoPipeline
    from diffusers.utils import export_to_video

    pipe = LTXImageToVideoPipeline.from_pretrained(
        cfg["video"]["ltx_model"], torch_dtype=torch.bfloat16
    )
    if cfg["video"]["low_vram"] and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")

    image = Image.open(image_path).convert("RGB").resize(
        (cfg["video"]["width"], cfg["video"]["height"]), Image.LANCZOS
    )
    frames = pipe(
        image=image,
        prompt=prompt,
        num_frames=cfg["video"]["num_frames"],
        width=cfg["video"]["width"],
        height=cfg["video"]["height"],
    ).frames[0]
    export_to_video(frames, str(out_mp4), fps=cfg["reel"]["fps"])
    return out_mp4


def _cogvideox_2b(image_path: Path, prompt: str, out_mp4: Path, cfg: dict) -> Path:
    from diffusers import CogVideoXImageToVideoPipeline
    from diffusers.utils import export_to_video

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        cfg["video"]["cogvideox_model"], torch_dtype=torch.bfloat16
    )
    if cfg["video"]["low_vram"] and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
    elif torch.cuda.is_available():
        pipe = pipe.to("cuda")

    image = Image.open(image_path).convert("RGB")
    frames = pipe(
        image=image,
        prompt=prompt,
        num_frames=49,           # CogVideoX-2b sweet spot
        guidance_scale=6.0,
    ).frames[0]
    export_to_video(frames, str(out_mp4), fps=8)
    return out_mp4


def make_clip(
    image_path: Path,
    visual_prompt: str,
    out_mp4: Path,
    duration_s: float,
    cfg: dict,
) -> Path:
    backend = cfg["video"]["pipeline"]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    if backend == "image_kenburns":
        return _kenburns(image_path, out_mp4, duration_s, cfg["reel"]["fps"])
    if backend == "ltx":
        return _ltx_video(image_path, visual_prompt, out_mp4, cfg)
    if backend == "cogvideox2b":
        return _cogvideox_2b(image_path, visual_prompt, out_mp4, cfg)
    raise ValueError(f"Unknown video pipeline: {backend}")
