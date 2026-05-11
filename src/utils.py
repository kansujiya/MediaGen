from __future__ import annotations

import os
import re
import json
import time
import logging
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return logging.getLogger("mediagen")


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(genre: str) -> dict:
    base = load_yaml(REPO_ROOT / "config" / "config.yaml")
    genre_path = REPO_ROOT / "config" / "genres" / f"{genre}.yaml"
    if not genre_path.exists():
        raise FileNotFoundError(f"No genre config: {genre_path}")
    base["_genre"] = load_yaml(genre_path)
    return base


def env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def slugify(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s).strip("-").lower()
    return s[:60] or f"reel-{int(time.time())}"


def workspace_dir(reel_id: str) -> Path:
    d = REPO_ROOT / env("WORKSPACE_DIR", "workspace") / reel_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def output_dir() -> Path:
    d = REPO_ROOT / env("OUTPUT_DIR", "output")
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_json(obj, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
