"""Generate a structured reel script via a local LLM (Ollama)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import ollama

from .utils import REPO_ROOT, env


def _strip_code_fence(s: str) -> str:
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", s, re.S)
    return m.group(1) if m else s


def _coerce_json(raw: str) -> dict:
    raw = _strip_code_fence(raw).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"LLM did not return JSON. Got:\n{raw[:400]}")
    return json.loads(raw[start : end + 1])


def generate_script(topic: str, cfg: dict) -> dict:
    """Return a dict with title/lines/hashtags. Validates structure."""
    genre = cfg["_genre"]
    prompt_path = REPO_ROOT / "prompts" / f"script_{genre['genre']}.txt"
    template = prompt_path.read_text(encoding="utf-8")

    prompt = template.format(
        topic=topic,
        character_name=genre["character"]["display_name"],
        total_seconds=genre["script"]["total_seconds"],
        tone=genre["script"]["tone"],
        cta=genre["script"]["cta"],
    )

    client = ollama.Client(host=env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    resp = client.generate(
        model=env("OLLAMA_MODEL", cfg["llm"]["model"]),
        prompt=prompt,
        options={"temperature": cfg["llm"]["temperature"]},
        format="json",
    )
    data = _coerce_json(resp["response"])

    # Light validation — fail fast rather than push bad data downstream.
    if "lines" not in data or not isinstance(data["lines"], list) or not data["lines"]:
        raise ValueError(f"Bad script structure: {data}")
    for i, line in enumerate(data["lines"]):
        for k in ("text", "visual_prompt", "on_screen_text"):
            if k not in line:
                raise ValueError(f"Line {i} missing {k}: {line}")

    return data
