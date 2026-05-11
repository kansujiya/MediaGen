"""SQLite state for runs and steps. Single source of truth shared between
the runner worker process and the Streamlit UI.

Schema:
  runs(id, genre, topic, status, current_agent, final_video_path,
       youtube_video_id, created_at, updated_at, error)
  steps(id, run_id, agent, attempt, status, input_json, output_json,
        model, started_at, ended_at, error)

Statuses on runs:
  queued -> running -> awaiting_approval -> approved -> uploading -> done
                    \\-> failed
                    \\-> rejected
                    \\-> cancelled
"""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ..utils import REPO_ROOT

DB_PATH = REPO_ROOT / "workspace" / "mediagen.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    genre TEXT NOT NULL,
    topic TEXT,
    status TEXT NOT NULL,
    current_agent TEXT,
    final_video_path TEXT,
    youtube_video_id TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL,
    agent TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 1,
    status TEXT NOT NULL,
    input_json TEXT,
    output_json TEXT,
    model TEXT,
    error TEXT,
    started_at REAL,
    ended_at REAL,
    FOREIGN KEY(run_id) REFERENCES runs(id)
);

CREATE INDEX IF NOT EXISTS ix_steps_run ON steps(run_id);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);
"""


def _now() -> float:
    return time.time()


@contextmanager
def conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
    finally:
        c.close()


def init_db() -> None:
    with conn() as c:
        c.executescript(SCHEMA)


# --- runs ----------------------------------------------------------------

def create_run(genre: str, topic: str | None) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO runs(genre, topic, status, created_at, updated_at) "
            "VALUES (?, ?, 'queued', ?, ?)",
            (genre, topic, _now(), _now()),
        )
        return cur.lastrowid


def update_run(run_id: int, **fields) -> None:
    if not fields:
        return
    fields["updated_at"] = _now()
    keys = ", ".join(f"{k}=?" for k in fields)
    with conn() as c:
        c.execute(f"UPDATE runs SET {keys} WHERE id=?", (*fields.values(), run_id))


def get_run(run_id: int) -> dict | None:
    with conn() as c:
        row = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(row) if row else None


def list_runs(limit: int = 50) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def pick_runnable_run() -> dict | None:
    """Pick one run the worker should advance now.

    A run is runnable if it's queued, approved (=> upload), or already running
    with a stale heartbeat (worker crashed). The 'approved' transition is
    triggered by the UI calling approve_run().
    """
    with conn() as c:
        row = c.execute(
            "SELECT * FROM runs "
            "WHERE status IN ('queued', 'approved') "
            "ORDER BY id ASC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None


def approve_run(run_id: int) -> None:
    update_run(run_id, status="approved", current_agent=None)


def reject_run(run_id: int, reason: str = "") -> None:
    update_run(run_id, status="rejected", current_agent=None, error=reason or None)


def cancel_run(run_id: int) -> None:
    update_run(run_id, status="cancelled", current_agent=None)


# --- steps ---------------------------------------------------------------

def start_step(
    run_id: int,
    agent: str,
    attempt: int = 1,
    input_obj: Any = None,
    model: str | None = None,
) -> int:
    with conn() as c:
        cur = c.execute(
            "INSERT INTO steps(run_id, agent, attempt, status, input_json, model, started_at) "
            "VALUES (?, ?, ?, 'running', ?, ?, ?)",
            (run_id, agent, attempt, json.dumps(input_obj, ensure_ascii=False) if input_obj is not None else None, model, _now()),
        )
        update_run(run_id, status="running", current_agent=agent)
        return cur.lastrowid


def finish_step(step_id: int, output_obj: Any = None, status: str = "done") -> None:
    with conn() as c:
        c.execute(
            "UPDATE steps SET status=?, output_json=?, ended_at=? WHERE id=?",
            (status, json.dumps(output_obj, ensure_ascii=False) if output_obj is not None else None, _now(), step_id),
        )


def fail_step(step_id: int, error: str) -> None:
    with conn() as c:
        c.execute(
            "UPDATE steps SET status='failed', error=?, ended_at=? WHERE id=?",
            (error, _now(), step_id),
        )


def list_steps(run_id: int) -> list[dict]:
    with conn() as c:
        rows = c.execute(
            "SELECT * FROM steps WHERE run_id=? ORDER BY id ASC", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]
