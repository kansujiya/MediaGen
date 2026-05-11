"""Per-shot character image generation. SDXL by default, with optional
character LoRA for identity lock. Uses CPU offload + VAE tiling so it runs
on 6-8 GB VRAM."""
from __future__ import annotations

from pathlib import Path

import torch


_pipe = None


def _build_pipe(cfg: dict):
    global _pipe
    if _pipe is not None:
        return _pipe

    from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline

    if cfg["image"]["pipeline"] == "sdxl":
        _pipe = StableDiffusionXLPipeline.from_pretrained(
            cfg["image"]["sdxl_model"],
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
    else:
        _pipe = StableDiffusionPipeline.from_pretrained(
            cfg["image"]["sd15_model"],
            torch_dtype=torch.float16,
        )

    if cfg["image"]["low_vram"] and torch.cuda.is_available():
        _pipe.enable_model_cpu_offload()
        _pipe.enable_vae_tiling()
    elif torch.cuda.is_available():
        _pipe = _pipe.to("cuda")

    return _pipe


def _maybe_load_lora(pipe, cfg: dict):
    lora_path = cfg["_genre"]["character"].get("lora_path")
    if not lora_path:
        return
    p = Path(lora_path)
    if not p.exists():
        # Soft-fail: surfacing a print is enough; we still get a usable image.
        print(f"[image_gen] LoRA not found at {p}, skipping.")
        return
    pipe.load_lora_weights(str(p.parent), weight_name=p.name)
    pipe.fuse_lora(lora_scale=cfg["_genre"]["character"].get("lora_weight", 0.8))


def generate_shot(
    visual_prompt: str,
    out_png: Path,
    cfg: dict,
    seed: int | None = None,
) -> Path:
    pipe = _build_pipe(cfg)
    _maybe_load_lora(pipe, cfg)

    char = cfg["_genre"]["character"]
    style = cfg["_genre"]["style"]["visual_keywords"]
    prompt = f"{visual_prompt}, {char['visual_prompt']}, {style}"
    neg = char["negative_prompt"]

    w, h = cfg["video"]["width"], cfg["video"]["height"]
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    if seed is not None:
        gen = gen.manual_seed(seed)

    img = pipe(
        prompt=prompt,
        negative_prompt=neg,
        num_inference_steps=cfg["image"]["steps"],
        guidance_scale=cfg["image"]["cfg"],
        width=w,
        height=h,
        generator=gen,
    ).images[0]

    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    return out_png
