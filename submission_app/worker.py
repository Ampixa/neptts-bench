"""Single worker process; no scoring runs in the web request or submitted code."""

import argparse
import fcntl
import hashlib
import json
import logging
import time
from pathlib import Path

from . import autoreview, donation_store
from .contract import BENCHMARK, SENTENCES
from .donations import apply_asr, load_prompt_sets, next_asr_job
from .evaluation import evaluate, profile, validate_report
from .offline import validate_count_report
from .storage import db, event, initialize


def verify_audio(files, validation):
    expected = json.loads(validation)["per_file"]
    for sent_id in SENTENCES:
        path = files / f"{sent_id}.wav"
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected[sent_id]["sha256"]:
            raise ValueError("Audio integrity verification failed.")


def process_additional(root, spec, scorer, label, skip=None, check=None):
    """Score one approved submission that lacks this judge.

    With automatic review on, a result that passes every check is published straight
    away; otherwise it awaits separate approval.

    Never touches the primary report, state or leaderboard. A failure is logged and the
    submission skipped for this process lifetime; a worker restart retries it.
    """
    skip = set() if skip is None else skip
    with db(root) as c:
        rows = c.execute(
            "SELECT * FROM submissions s WHERE s.state='published' AND s.profile IS NOT ? "
            "AND NOT EXISTS (SELECT 1 FROM profile_evaluations p WHERE p.submission_id=s.id AND p.profile=?) "
            "ORDER BY s.created",
            (spec["id"], spec["id"]),
        ).fetchall()
    row = next((r for r in rows if r["id"] not in skip), None)
    if row is None:
        return False
    sid = row["id"]
    try:
        files = root / "uploads" / sid
        verify_audio(files, row["validation"])
        report = scorer(files, lambda n: None)
        validate_report(report, spec["id"])
        (check or (lambda r: validate_count_report(r, label)))(report)
        if report["profile"] != spec:
            raise ValueError("Evaluator profile mismatch.")
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            current = c.execute("SELECT state FROM submissions WHERE id=?", (sid,)).fetchone()
            if current is None or current["state"] != "published":
                return True
            now = time.time()
            auto = autoreview.enabled()
            c.execute(
                "INSERT OR IGNORE INTO profile_evaluations(submission_id,profile,created,updated,state,report,reviewer_note) VALUES(?,?,?,?,?,?,?)",
                (sid, spec["id"], now, now, "published" if auto else "awaiting_review",
                 json.dumps(report, allow_nan=False),
                 f"Approved automatically by {autoreview.VERSION}." if auto else ""),
            )
            event(c, sid, "published",
                  f"Additional judge {spec['id']} scored by worker; "
                  + ("published automatically." if auto else "awaiting separate approval."))
    except Exception:
        logging.getLogger(__name__).exception("Additional judge failed for %s", sid)
        skip.add(sid)
        with db(root) as c:
            event(c, sid, "published",
                  f"Additional judge {spec['id']} did not complete; no scores recorded.")
    return True


def process_one(root, model="small", scorer=evaluate, spec=None, check=None):
    """Evaluate the oldest queued submission with the primary judge. False when idle."""
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
            "UPDATE submissions SET state='evaluating',profile=?,progress=0,report=NULL,message=?,attempts=attempts+1,updated=? WHERE id=?",
            (spec["id"], "Evaluation is running.", time.time(), sid),
        )
        event(c, sid, "evaluating", "Evaluation started with " + spec["id"])
    try:
        files = root / "uploads" / sid
        verify_audio(files, row["validation"])

        def progress(n):
            with db(root) as c:
                c.execute(
                    "UPDATE submissions SET progress=?,updated=? WHERE id=?",
                    (n, time.time(), sid),
                )

        report = scorer(files, model, progress)
        validate_report(report, spec["id"])
        if check:
            check(report)
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
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
            autoreview.retry_or_fail(c, row, "Evaluation did not complete.")
        return True
    if autoreview.enabled():
        autoreview.review(root, sid, spec["id"])
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(".local/submissions"))
    parser.add_argument(
        "--engine", choices=["rani", "whisper"], default="rani",
        help="Rani is the primary judge. Whisper is kept as a second opinion or fallback.",
    )
    parser.add_argument(
        "--role", choices=["primary", "additional"], default="primary",
        help="primary: evaluate the queue (Rani also checks donated takes and backfills "
             "older submissions). additional: score every published submission as a second opinion.",
    )
    parser.add_argument("--checkpoint", type=Path, help="Pinned Rani checkpoint (rani.nemo).")
    parser.add_argument("--model", choices=["tiny", "small"], default="small",
                        help="Whisper model size for --engine whisper.")
    parser.add_argument("--once", action="store_true", help="Stop when there is no work left.")
    args = parser.parse_args()
    initialize(args.data)
    spec, scorer, check = engine(args)
    # One lock per engine: Rani and Whisper run in different Python environments.
    lock = (args.data / f".worker-{args.engine}.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if args.role == "additional":
        label, skip = args.engine.capitalize(), set()
        while True:
            worked = process_additional(args.data, spec, lambda f, p: scorer(f, args.model, p),
                                        label, skip, check)
            if args.once and not worked:
                return
            if not worked:
                time.sleep(30)
    recover_interrupted(args.data)
    run_primary(args, spec, scorer, check)


def engine(args):
    """(profile, scorer(files, model, progress), extra report check) for one judge."""
    if args.engine == "whisper":
        # Whisper reports carry rates only; validate_report already checked them.
        return profile(args.model), evaluate, lambda report: None
    from neptts_eval import rani

    if not args.checkpoint:
        raise SystemExit("--engine rani requires --checkpoint (see rani --pin).")
    spec = rani.profile(BENCHMARK)
    decoder = rani.build_decoder({"checkpoint": args.checkpoint})
    args.decoder = decoder

    def scorer(files, model, progress):
        return rani.evaluate(files, SENTENCES, BENCHMARK, {}, progress, decoder=decoder)

    def check(report):
        validate_count_report(report, "Rani")
        if report["profile"] != spec:
            raise ValueError("Evaluator profile mismatch.")

    return spec, scorer, check


def recover_interrupted(root):
    with db(root) as c:
        c.execute("BEGIN IMMEDIATE")
        for row in c.execute("SELECT * FROM submissions WHERE state='evaluating'").fetchall():
            autoreview.retry_or_fail(c, row, "Previous worker stopped before completion.")


def run_primary(args, spec, scorer, check):
    donations = args.engine == "rani"
    if donations:
        from neptts_eval import rani

        sets = load_prompt_sets()
        store = donation_store.from_env(args.data, open_sets=len(sets) > 1)

        def donation_cer(path, text):
            return rani.score_file(args.decoder, path, text)["cer"]

    donation_skip, backfill_skip = set(), set()
    while True:
        # Donor takes first: donors are waiting, and each takes about a second.
        worked = donations and process_donation(args.data, sets, donation_cer, spec["id"], donation_skip, store)
        worked = worked or process_one(args.data, args.model, scorer, spec, check)
        # Published submissions scored by an earlier primary judge get this one too.
        worked = worked or process_additional(
            args.data, spec, lambda f, p: scorer(f, args.model, p),
            args.engine.capitalize(), backfill_skip, check)
        if args.once and not worked:
            break
        if not worked:
            time.sleep(10)


def process_donation(root, sets, score, profile_id, skip=None, store=None):
    """ASR-check one donated take: was the prompt actually read? Returns False when idle."""
    skip = set() if skip is None else skip
    job = next_asr_job(root, sets, skip)
    if job is None:
        return False
    try:
        if job["text"] is None:
            raise ValueError("Prompt no longer exists.")
        store = store or donation_store.from_env(root)
        with store.local_copy(job["rec"]) as path:
            cer = score(path, job["text"])
        if isinstance(cer, bool) or not isinstance(cer, (int, float)) or not cer >= 0:
            raise ValueError("Invalid CER.")
        apply_asr(root, job["id"], float(cer), profile_id)
    except Exception:
        logging.getLogger(__name__).exception("Donation ASR failed for %s", job["id"])
        skip.add(job["id"])
    return True


if __name__ == "__main__":
    main()
