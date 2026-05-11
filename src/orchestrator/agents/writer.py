"""Writer: wraps script_gen for both first draft and critic-driven revisions.
A revision prompt prepends the prior attempt and the critic's feedback so the
LLM knows what to fix."""
from __future__ import annotations

import json
import re

import ollama

from ...utils import REPO_ROOT, env
from ...script_gen import _coerce_json, generate_script


_REVISE_TMPL = """You are rewriting a short-form reel script. A reviewer gave
the previous draft a score of {score}/10 with this feedback:

FEEDBACK:
{feedback}

Now rewrite the script for the SAME topic, fixing every point the reviewer
raised. Keep the JSON schema identical to the prior draft.

TOPIC: {topic}
PRIOR DRAFT:
{prior}

ORIGINAL PROMPT FOR REFERENCE:
{base_prompt}

Output ONLY the new JSON. No prose, no markdown.
"""


def run(cfg: dict, topic: str, prior: dict | None = None, feedback: str | None = None, score: int | None = None) -> dict:
    if prior is None or feedback is None:
        return generate_script(topic, cfg)

    genre = cfg["_genre"]
    base_prompt = (REPO_ROOT / "prompts" / f"script_{genre['genre']}.txt").read_text(
        encoding="utf-8"
    ).format(
        topic=topic,
        character_name=genre["character"]["display_name"],
        total_seconds=genre["script"]["total_seconds"],
        tone=genre["script"]["tone"],
        cta=genre["script"]["cta"],
    )

    prompt = _REVISE_TMPL.format(
        score=score or "?",
        feedback=feedback,
        topic=topic,
        prior=json.dumps(prior, ensure_ascii=False, indent=2),
        base_prompt=base_prompt,
    )

    client = ollama.Client(host=env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    resp = client.generate(
        model=env("OLLAMA_MODEL", cfg["llm"]["model"]),
        prompt=prompt,
        options={"temperature": cfg["llm"]["temperature"]},
        format="json",
    )
    data = _coerce_json(resp["response"])
    if "lines" not in data or not data["lines"]:
        raise ValueError(f"Revised script malformed: {data}")
    return data
