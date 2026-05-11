"""Worker loop: polls the runs table and advances one run at a time.

Run as:  python -m src.orchestrator.runner

State machine per run:
  queued -> [Researcher] -> [Writer v1] -> [Critic] --(score>=7 or attempts==MAX)--> [Producer]
                              ^------------|  not approved & attempts<MAX
                                                                        |
                                                                        v
                                                            awaiting_approval
                                                            (UI: Approve / Reject)
                                                                        |
                                                                        v
                                                                  [Uploader] -> done

If the run was started with a topic provided by the user, the Researcher
step is skipped and that topic flows straight to the Writer.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

from ..utils import load_config, setup_logging
from . import state
from .agents import critic as critic_agent
from .agents import producer as producer_agent
from .agents import researcher as researcher_agent
from .agents import uploader as uploader_agent
from .agents import writer as writer_agent

MAX_WRITE_ATTEMPTS = 3
POLL_SEC = 2.0


def _load_last_output(run_id: int, agent: str) -> dict | None:
    """Return the output_json of the most recent successful step for that agent."""
    for s in reversed(state.list_steps(run_id)):
        if s["agent"] == agent and s["status"] == "done" and s["output_json"]:
            return json.loads(s["output_json"])
    return None


def _run_one(run: dict, log) -> None:
    run_id = run["id"]
    cfg = load_config(run["genre"])

    # 1. Research (skip if user supplied a topic)
    topic = run["topic"]
    if not topic:
        step = state.start_step(run_id, "researcher", model=cfg["llm"]["model"])
        try:
            pick = researcher_agent.run(cfg)
        except Exception as e:
            state.fail_step(step, f"{type(e).__name__}: {e}")
            raise
        state.finish_step(step, pick)
        topic = pick["topic"]
        state.update_run(run_id, topic=topic)
        log.info("[%s] researcher picked: %s", run_id, topic)

    # 2. Writer <-> Critic loop
    draft: dict | None = None
    verdict: dict | None = None
    for attempt in range(1, MAX_WRITE_ATTEMPTS + 1):
        step = state.start_step(
            run_id, "writer", attempt=attempt, model=cfg["llm"]["model"],
            input_obj={"topic": topic, "prior_score": verdict.get("score") if verdict else None,
                       "prior_feedback": verdict.get("feedback") if verdict else None},
        )
        try:
            draft = writer_agent.run(
                cfg, topic,
                prior=draft if attempt > 1 else None,
                feedback=verdict.get("feedback") if verdict else None,
                score=verdict.get("score") if verdict else None,
            )
        except Exception as e:
            state.fail_step(step, f"{type(e).__name__}: {e}")
            raise
        state.finish_step(step, draft)

        step = state.start_step(run_id, "critic", attempt=attempt, model=cfg["llm"]["model"], input_obj=draft)
        try:
            verdict = critic_agent.run(cfg, topic, draft)
        except Exception as e:
            state.fail_step(step, f"{type(e).__name__}: {e}")
            raise
        state.finish_step(step, verdict)
        log.info("[%s] critic attempt %d: score=%d approved=%s",
                 run_id, attempt, verdict["score"], verdict["approved"])
        if verdict["approved"]:
            break

    # 3. Producer (one big step — slow, no sub-progress for now)
    step = state.start_step(run_id, "producer", input_obj={"script_title": draft["title"]})
    try:
        prod = producer_agent.run(cfg, draft)
    except Exception as e:
        state.fail_step(step, f"{type(e).__name__}: {e}")
        raise
    state.finish_step(step, prod)
    state.update_run(run_id, final_video_path=prod["final_video_path"])

    # 4. Hand off to human approval gate.
    state.update_run(run_id, status="awaiting_approval", current_agent="awaiting_human")
    log.info("[%s] awaiting human approval (video: %s)", run_id, prod["final_video_path"])


def _upload(run: dict, log) -> None:
    run_id = run["id"]
    cfg = load_config(run["genre"])
    draft = _load_last_output(run_id, "writer")
    if draft is None:
        raise RuntimeError("no writer output found for run")
    step = state.start_step(run_id, "uploader", input_obj={"video": run["final_video_path"]})
    try:
        out = uploader_agent.run(cfg, draft, run["final_video_path"])
    except Exception as e:
        state.fail_step(step, f"{type(e).__name__}: {e}")
        raise
    state.finish_step(step, out)
    state.update_run(
        run_id,
        status="done",
        current_agent=None,
        youtube_video_id=out.get("youtube_video_id"),
    )
    log.info("[%s] uploaded: %s", run_id, out.get("url"))


def loop_forever():
    log = setup_logging()
    state.init_db()
    log.info("Runner up. Polling every %.1fs. DB=%s", POLL_SEC, state.DB_PATH)

    while True:
        run = state.pick_runnable_run()
        if run is None:
            time.sleep(POLL_SEC)
            continue

        try:
            if run["status"] == "queued":
                _run_one(run, log)
            elif run["status"] == "approved":
                _upload(run, log)
        except Exception as e:
            tb = traceback.format_exc(limit=4)
            log.error("[%s] failed: %s\n%s", run["id"], e, tb)
            state.update_run(run["id"], status="failed", current_agent=None, error=f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        loop_forever()
    except KeyboardInterrupt:
        print("\nrunner stopped")
        sys.exit(0)
