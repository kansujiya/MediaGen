"""Researcher: pulls candidate topics from Reddit (top posts of the last 24h
for genre-configured subreddits) and asks the LLM to pick the most reel-worthy
angle. Falls back to LLM brainstorm if Reddit is unreachable."""
from __future__ import annotations

import json
import random
import re

import ollama
import requests

from ...utils import env

USER_AGENT = "mediagen/0.1 (local reel pipeline)"
HEADERS = {"User-Agent": USER_AGENT}


def _fetch_subreddit_top(sub: str, limit: int = 15) -> list[dict]:
    url = f"https://www.reddit.com/r/{sub}/top.json?t=day&limit={limit}"
    r = requests.get(url, headers=HEADERS, timeout=10)
    r.raise_for_status()
    posts = []
    for child in r.json().get("data", {}).get("children", []):
        d = child["data"]
        posts.append({
            "subreddit": sub,
            "title": d.get("title", ""),
            "selftext": (d.get("selftext") or "")[:400],
            "score": d.get("score", 0),
            "url": "https://reddit.com" + d.get("permalink", ""),
        })
    return posts


def gather_candidates(subreddits: list[str], per_sub: int = 10) -> list[dict]:
    out: list[dict] = []
    for sub in subreddits:
        try:
            out.extend(_fetch_subreddit_top(sub, per_sub))
        except Exception as e:  # network/ratelimit — fall through with what we have
            print(f"[researcher] reddit fetch failed for r/{sub}: {e}")
    random.shuffle(out)
    return out[:30]


_PICK_PROMPT = """You are a viral short-form content researcher.

From the candidate items below, pick ONE that would make the best
{seconds}-second {genre} reel for a young Indian audience.

Rules:
- Pick something with a clear emotional hook or actionable insight.
- Avoid politics, religion, brand names, personal trauma posts.
- Synthesize a fresh angle — do not copy the source title verbatim.

Output ONLY this JSON (no prose, no fences):
{{
  "topic": "<one-sentence angle for the reel, in English>",
  "rationale": "<why this will resonate, 1 sentence>",
  "source_url": "<the picked item's url, or empty string if synthesized>"
}}

CANDIDATES:
{candidates}
"""


def _coerce_json(raw: str) -> dict:
    raw = raw.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError(f"Researcher LLM did not return JSON: {raw[:300]}")
    return json.loads(m.group(0))


def run(cfg: dict) -> dict:
    """Return {topic, rationale, source_url, candidates}."""
    genre_cfg = cfg["_genre"]
    subs: list[str] = genre_cfg.get("research", {}).get("subreddits", [])
    candidates = gather_candidates(subs) if subs else []

    if not candidates:
        # Cold-start fallback: brainstorm topics from genre alone.
        candidates = [{"title": f"general {genre_cfg['genre']} insight", "selftext": "", "score": 0, "subreddit": "n/a", "url": ""}]

    rendered = "\n".join(
        f"- [{c['subreddit']} score={c['score']}] {c['title']} :: {c['selftext'][:160]}"
        for c in candidates
    )
    prompt = _PICK_PROMPT.format(
        genre=genre_cfg["genre"],
        seconds=genre_cfg["script"]["total_seconds"],
        candidates=rendered,
    )
    client = ollama.Client(host=env("OLLAMA_HOST", "http://127.0.0.1:11434"))
    resp = client.generate(
        model=env("OLLAMA_MODEL", cfg["llm"]["model"]),
        prompt=prompt,
        options={"temperature": 0.7},
        format="json",
    )
    pick = _coerce_json(resp["response"])
    pick["candidates_seen"] = len(candidates)
    return pick
