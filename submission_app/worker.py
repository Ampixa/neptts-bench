"""Single worker process; no scoring runs in the web request or submitted code."""

import argparse
import fcntl
import hashlib
import json
import logging
import time
from pathlib import Path

from .contract import SENTENCES
from .evaluation import evaluate, profile, validate_report
from .storage import db, event, initialize


def process_one(root, model="small", scorer=evaluate, spec=None):
    spec = spec or profile(model)
    with db(root) as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute(
            "SELECT * FROM submissions WHERE state='queued' ORDER BY created LIMIT 1"
        ).fetchone()
        if not row:
            return False
        sid = row["id"]
        c.execute(
            "UPDATE submissions SET state='evaluating',profile=?,progress=0,report=NULL,message=?,updated=? WHERE id=?",
            (spec["id"], "Evaluation is running.", time.time(), sid),
        )
        event(c, sid, "evaluating", "Evaluation started with " + spec["id"])
    try:
        files = root / "uploads" / sid
        expected = json.loads(row["validation"])["per_file"]
        for sent_id in SENTENCES:
            path = files / f"{sent_id}.wav"
            if (
                hashlib.sha256(path.read_bytes()).hexdigest()
                != expected[sent_id]["sha256"]
            ):
                raise ValueError("Audio integrity verification failed.")

        def progress(n):
            with db(root) as c:
                c.execute(
                    "UPDATE submissions SET progress=?,updated=? WHERE id=?",
                    (n, time.time(), sid),
                )

        report = scorer(files, model, progress)
        validate_report(report, spec["id"])
        with db(root) as c:
            c.execute(
                "UPDATE submissions SET state='awaiting_review',report=?,progress=?,message=?,updated=? WHERE id=? AND state='evaluating'",
                (
                    json.dumps(report, allow_nan=False),
                    len(SENTENCES),
                    "Evaluation complete. Awaiting admin review.",
                    time.time(),
                    sid,
                ),
            )
            event(
                c,
                sid,
                "awaiting_review",
                "All prompts evaluated; scores are not public until approved.",
            )
    except Exception:
        logging.getLogger(__name__).exception("Evaluation failed for %s", sid)
        with db(root) as c:
            c.execute(
                "UPDATE submissions SET state='evaluation_failed',message=?,updated=? WHERE id=?",
                (
                    "Evaluation failed. An administrator must inspect worker logs before retrying.",
                    time.time(),
                    sid,
                ),
            )
            event(
                c,
                sid,
                "evaluation_failed",
                "Evaluation did not complete. No scores published.",
            )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/submissions"))
    parser.add_argument("--model", choices=["tiny", "small"], default="small")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    initialize(args.data)
    lock = (args.data / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with db(args.data) as c:
        interrupted = c.execute(
            "SELECT id FROM submissions WHERE state='evaluating'"
        ).fetchall()
        for row in interrupted:
            c.execute(
                "UPDATE submissions SET state='evaluation_failed',message=?,updated=? WHERE id=?",
                ("Worker interrupted. Admin retry required.", time.time(), row["id"]),
            )
            event(
                c,
                row["id"],
                "evaluation_failed",
                "Previous worker stopped before completion.",
            )
    spec = profile(args.model)
    while True:
        worked = process_one(args.data, args.model, spec=spec)
        if args.once:
            break
        if not worked:
            time.sleep(3)


if __name__ == "__main__":
    main()
