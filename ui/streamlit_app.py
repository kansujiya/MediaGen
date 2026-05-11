"""MediaGen control panel.

Run alongside the worker:
    streamlit run ui/streamlit_app.py
    python -m src.orchestrator.runner   # in another terminal

The worker advances runs; this UI reads the same SQLite DB and provides:
  - Start a new run for a genre (with optional topic override)
  - Live agent cards (researcher / writer / critic loop / producer)
  - Approve / Reject gate showing the final mp4
  - History + per-step input/output inspection
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.orchestrator import state  # noqa: E402

AGENT_ORDER = ["researcher", "writer", "critic", "producer", "uploader"]
AGENT_EMOJI = {  # noqa  (UI affordance only — not added to code/docs)
    "researcher": "[R]",
    "writer": "[W]",
    "critic": "[C]",
    "producer": "[P]",
    "uploader": "[U]",
}

STATUS_COLOR = {
    "queued": "gray",
    "running": "blue",
    "awaiting_approval": "orange",
    "approved": "blue",
    "done": "green",
    "failed": "red",
    "rejected": "red",
    "cancelled": "gray",
}


def _fmt_time(ts: float | None) -> str:
    if not ts:
        return "—"
    return datetime.fromtimestamp(ts).strftime("%H:%M:%S")


def _list_genres() -> list[str]:
    return sorted(p.stem for p in (ROOT / "config" / "genres").glob("*.yaml"))


def sidebar_new_run():
    st.sidebar.header("Start a new run")
    genres = _list_genres()
    if not genres:
        st.sidebar.warning("No genre configs found in config/genres/")
        return
    genre = st.sidebar.selectbox("Genre", genres, index=0)
    topic = st.sidebar.text_input(
        "Topic (optional)", placeholder="Leave blank to let the Researcher pick"
    ).strip()
    if st.sidebar.button("Start run", type="primary", use_container_width=True):
        run_id = state.create_run(genre, topic or None)
        st.sidebar.success(f"Queued run #{run_id}")
        st.rerun()

    st.sidebar.divider()
    st.sidebar.caption(
        f"Worker: `python -m src.orchestrator.runner`\nDB: `{state.DB_PATH.relative_to(ROOT)}`"
    )


def render_step_card(step: dict):
    label = AGENT_EMOJI.get(step["agent"], "[?]") + " " + step["agent"].title()
    attempt = step["attempt"]
    if attempt > 1:
        label += f" (v{attempt})"
    color = STATUS_COLOR.get(step["status"], "gray")
    with st.container(border=True):
        cols = st.columns([3, 1, 1])
        cols[0].markdown(f"**{label}** — :{color}[{step['status']}]")
        cols[1].caption(f"start {_fmt_time(step['started_at'])}")
        cols[2].caption(f"end   {_fmt_time(step['ended_at'])}")
        if step.get("error"):
            st.error(step["error"])
        with st.expander("input / output"):
            if step["input_json"]:
                st.caption("INPUT")
                st.json(json.loads(step["input_json"]))
            if step["output_json"]:
                st.caption("OUTPUT")
                st.json(json.loads(step["output_json"]))


def render_approval(run: dict):
    st.subheader("Awaiting human approval")
    video_path = run["final_video_path"]
    if video_path and Path(video_path).exists():
        st.video(video_path)
    else:
        st.warning(f"Expected video at {video_path}, not found.")
    c1, c2, c3 = st.columns(3)
    if c1.button("Approve & upload", type="primary", use_container_width=True):
        state.approve_run(run["id"])
        st.rerun()
    if c2.button("Reject", use_container_width=True):
        state.reject_run(run["id"], reason="human rejected at QC gate")
        st.rerun()
    if c3.button("Cancel", use_container_width=True):
        state.cancel_run(run["id"])
        st.rerun()


def render_run_detail(run: dict):
    color = STATUS_COLOR.get(run["status"], "gray")
    st.markdown(
        f"### Run #{run['id']} — :{color}[{run['status']}] "
        f"({run['genre']})"
    )
    if run["topic"]:
        st.caption(f"Topic: *{run['topic']}*")
    if run["error"]:
        st.error(run["error"])

    steps = state.list_steps(run["id"])
    if not steps:
        st.info("Worker hasn't picked this up yet. Make sure the runner is running.")
    else:
        for s in steps:
            render_step_card(s)

    if run["status"] == "awaiting_approval":
        st.divider()
        render_approval(run)
    elif run["status"] == "done" and run.get("youtube_video_id"):
        st.success(
            f"Uploaded -> https://youtube.com/shorts/{run['youtube_video_id']}"
        )


def render_history(active_run_id: int | None):
    st.subheader("History")
    rows = state.list_runs(limit=30)
    if not rows:
        st.caption("No runs yet.")
        return
    for r in rows:
        if r["id"] == active_run_id:
            continue
        color = STATUS_COLOR.get(r["status"], "gray")
        topic = (r["topic"] or "(no topic)")[:60]
        c1, c2 = st.columns([4, 1])
        c1.markdown(
            f"**#{r['id']}** :{color}[{r['status']}] · {r['genre']} · {topic} · "
            f"{_fmt_time(r['updated_at'])}"
        )
        if c2.button("Open", key=f"open-{r['id']}"):
            st.session_state["active_run"] = r["id"]
            st.rerun()


def main():
    st.set_page_config(page_title="MediaGen", layout="wide")
    state.init_db()

    st.title("MediaGen — agentic reel pipeline")

    sidebar_new_run()

    # Pick the active run: explicit selection wins, else newest non-terminal,
    # else newest of all.
    runs = state.list_runs(limit=50)
    active_id = st.session_state.get("active_run")
    if active_id is None:
        non_terminal = [r for r in runs if r["status"] not in ("done", "failed", "rejected", "cancelled")]
        active_id = (non_terminal[0]["id"] if non_terminal else (runs[0]["id"] if runs else None))

    left, right = st.columns([2, 1])
    with left:
        if active_id is None:
            st.info("Start a run from the sidebar.")
        else:
            run = state.get_run(active_id)
            if run:
                render_run_detail(run)
    with right:
        render_history(active_id)

    # Lightweight live refresh while something is in flight.
    if active_id:
        run = state.get_run(active_id)
        if run and run["status"] in ("queued", "running", "approved"):
            time.sleep(2)
            st.rerun()


if __name__ == "__main__":
    main()
