"""Critic: scores a draft script against a rubric and decides if it ships.

Returns {score:int (0-10), approved:bool, feedback:str, issues:list[str]}.
The runner accepts at score >= threshold or after max_attempts.
"""
from __future__ import annotations

import json
import re

import ollama

from ...utils import env

PASS_THRESHOLD = 7

RUBRIC = """SCORING RUBRIC (each 0-2 points, total /10):

1. HOOK STRENGTH — does line 1 land an emotional or curiosity hook in <3s?
2. HINGLISH NATURALNESS — code-mixing feels natural, not translated.
3. STRUCTURE — 2-3 lines, each <= 12 words, total fits {total_seconds}s.
4. SAFETY — no religion, politics, brand names, personal-trauma exploitation.
5. CTA — last line ends with the configured call to action.
"""

_PROMPT = """You are a strict short-form reels editor reviewing a draft script.

{rubric}

TOPIC: {topic}
EXPECTED CTA: "{cta}"

DRAFT (JSON):
{draft}

Output ONLY this JSON:
{{
  "score": <0-10 integer>,
  "approved": <true if score >= 7 AND no safety issues else false>,
  "issues": ["<short bullet>", ...],
  "feedback": "<2-4 sentence rewrite guidance>"
}}
"""


def _coerce(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"Critic LLM did not return JSON: {raw[:300]}")
    obj = json.loads(m.group(0))
    obj["score"] = int(obj.get("score", 0))
    obj["approved"] = bool(obj.get("approved", False))
    obj.setdefault("issues", [])
    obj.setdefault("feedback", "")
    return obj


def run(cfg: dict, topic: str, draft: dict) -> dict:
    genre = cfg["_genre"]
    rubric = RUBRIC.format(total_seconds=genre["script"]["total_seconds"])
    prompt = _PROMPT.format(
        rubric=rubric,
        topic=topic,
        cta=genre["script"]["cta"],
        draft=json.dumps(draft, ensure_ascii=False, indent=2),
    )
    client = ollama.Client(host=env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    resp = client.generate(
        model=env("OLLAMA_MODEL", cfg["llm"]["model"]),
        prompt=prompt,
        options={"temperature": 0.2},
        format="json",
    )
    verdict = _coerce(resp["response"])
    # Authoritative gate: enforce threshold ourselves rather than trust the LLM.
    verdict["approved"] = verdict["score"] >= PASS_THRESHOLD and not any(
        "safety" in i.lower() for i in verdict["issues"]
    )
    return verdict
