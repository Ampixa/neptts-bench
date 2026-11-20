"""Automatic admin: queue uploads, publish clean evaluations, hold the rest for a person.

The policy never rejects. It either publishes (every check passed) or holds the
submission in `awaiting_review` with machine-readable reasons, so a reviewer only
sees what needs judgement. Reviewers can still take down anything it published.

NEPTTS_AUTO_REVIEW selects the mode:
- on (default): queue, retry, publish clean results, hold the rest.
- shadow: queue and retry as with `on`, and record what the policy would decide,
  but publish nothing. A reviewer approves every submission, and the admin
  readiness report shows how often they agreed with the policy. Run this for a
  few weeks before switching to `on`.
- off: fully manual review.
"""

import json
import os
import time

from .contract import SENTENCES, template
from .evaluation import validate_report
from .profiles import selection_group
from .storage import db, event

VERSION = "auto-review-v1"
MAX_ATTEMPTS = 2  # evaluation runs per submission before a person must retry it
FAILED_PROMPT_CER = 0.9  # a prompt this bad was effectively not understood
MAX_FAILED_SHARE = 0.2  # more than this many such prompts suggests wrong files or language
MAX_SHARED_AUDIO = 0.05  # share of WAVs identical to another submission's
# Lowest believable CER for synthetic speech under the primary judge. Below it, the
# audio may be human recordings. Unset until Rani's real-speech calibration exists.
PLAUSIBLE_FLOOR = None
PLACEHOLDER_FIELDS = ("system_name", "model_version", "organization", "voice",
                      "generation_settings", "training_overlap")

REASONS = {
    "no_primary_judge": "The primary judge is not configured, so nothing can be published automatically.",
    "not_primary_judge": "Scored by a judge other than the primary one.",
    "development_judge": "Scored by a development-only judge.",
    "invalid_report": "The evaluation report failed consistency checks.",
    "selection_unknown": "The generation policy (single-take or best-of-N) is not fully declared.",
    "judge_family": "Declares use of a judge-family model for selection, filtering or training.",
    "placeholder": "Some manifest fields still contain the template's example text.",
    "duplicate_name": "Another submission already uses this system name and version.",
    "shared_audio": "Some audio files are identical to another submission's.",
    "failed_prompts": "Many prompts were not understood at all: wrong files, order or language?",
    "implausibly_good": "Scores better than real human speech: are these recordings synthetic?",
}


def mode():
    value = os.environ.get("NEPTTS_AUTO_REVIEW", "on").strip().lower()
    if value in ("off", "0", "false"):
        return "off"
    return "shadow" if value == "shadow" else "on"


def enabled():
    """Uploads queue themselves and failed runs retry (modes `on` and `shadow`)."""
    return mode() != "off"


def publishes():
    """The policy may publish without a person (mode `on` only)."""
    return mode() == "on"


def norm(text):
    return " ".join(str(text).lower().split())


def assess(c, row, report, primary_id):
    """Reasons to hold this evaluated submission; empty means it can be published."""
    if primary_id is None:
        return ["no_primary_judge"]
    reasons = []
    spec = report.get("profile", {})
    if spec.get("id") != primary_id:
        reasons.append("not_primary_judge")
    if spec.get("development_only"):
        reasons.append("development_judge")
    try:
        validate_report(report, row["profile"])
    except (ValueError, TypeError, AttributeError, KeyError):
        reasons.append("invalid_report")
        return reasons
    manifest = json.loads(row["manifest"])
    if selection_group(manifest) == "unknown":
        reasons.append("selection_unknown")
    if manifest.get("judge_family_used") is True:
        reasons.append("judge_family")
    example = template()
    if any(norm(manifest.get(k, "")) == norm(example[k]) for k in PLACEHOLDER_FIELDS):
        reasons.append("placeholder")
    others = c.execute(
        "SELECT manifest,validation FROM submissions WHERE id!=? AND state!='rejected'",
        (row["id"],),
    ).fetchall()
    name = (norm(manifest["system_name"]), norm(manifest["model_version"]))
    mine = {f["sha256"] for f in json.loads(row["validation"])["per_file"].values()}
    shared = set()
    for other in others:
        m = json.loads(other["manifest"])
        if (norm(m["system_name"]), norm(m["model_version"])) == name:
            reasons.append("duplicate_name")
        shared |= mine & {f["sha256"] for f in json.loads(other["validation"])["per_file"].values()}
    if len(shared) > MAX_SHARED_AUDIO * len(SENTENCES):
        reasons.append("shared_audio")
    rows = report["per_sentence"].values()
    if sum(r["cer"] >= FAILED_PROMPT_CER for r in rows) > MAX_FAILED_SHARE * len(rows):
        reasons.append("failed_prompts")
    if PLAUSIBLE_FLOOR is not None and report["cer"] < PLAUSIBLE_FLOOR:
        reasons.append("implausibly_good")
    return sorted(set(reasons))


def review(root, sid, primary_id):
    """Publish or hold one submission waiting for review. Returns the resulting state."""
    with db(root) as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
        if row is None or row["state"] != "awaiting_review" or not row["report"]:
            return row["state"] if row else None
        reasons = assess(c, row, json.loads(row["report"]), primary_id)
        shadow = not publishes()
        record = json.dumps({"policy": VERSION, "time": time.time(), "reasons": reasons,
                             "details": [REASONS[r] for r in reasons], "shadow": shadow})
        if shadow:
            verdict = ("would hold it: " + ", ".join(reasons)) if reasons else "would publish it"
            c.execute("UPDATE submissions SET message=?,auto_review=?,updated=? WHERE id=?",
                      ("Evaluation complete. Awaiting reviewer approval.", record, time.time(), sid))
            event(c, sid, "awaiting_review", f"{VERSION} (shadow) {verdict}. A reviewer decides.")
            return "awaiting_review"
        if reasons:
            message = "Held for a reviewer: " + " ".join(REASONS[r] for r in reasons)
            c.execute("UPDATE submissions SET message=?,auto_review=?,updated=? WHERE id=?",
                      (message, record, time.time(), sid))
            event(c, sid, "awaiting_review", f"{VERSION} held it: {', '.join(reasons)}.")
            return "awaiting_review"
        note = f"Approved automatically by {VERSION}: every check passed."
        c.execute(
            "UPDATE submissions SET state='published',message=?,reviewer_note=?,auto_review=?,updated=? WHERE id=?",
            ("Published on the leaderboard after automatic review.", note, record, time.time(), sid),
        )
        event(c, sid, "published", note)
        return "published"


def retry_or_fail(c, row, message):
    """After a failed or interrupted run: queue again while attempts remain."""
    if enabled() and row["attempts"] < MAX_ATTEMPTS:
        c.execute("UPDATE submissions SET state='queued',progress=0,message=?,updated=? WHERE id=?",
                  ("Evaluation will be retried automatically.", time.time(), row["id"]))
        event(c, row["id"], "queued", f"{message} Retrying automatically "
              f"(attempt {row['attempts'] + 1} of {MAX_ATTEMPTS}).")
        return "queued"
    c.execute("UPDATE submissions SET state='evaluation_failed',message=?,updated=? WHERE id=?",
              ("Evaluation failed. An administrator must inspect worker logs before retrying.",
               time.time(), row["id"]))
    event(c, row["id"], "evaluation_failed", message + " No scores published.")
    return "evaluation_failed"


def shadow_agreement(c):
    """How reviewers decided on submissions the policy judged in shadow mode."""
    out = {"would_publish": {"published": 0, "rejected": 0, "pending": 0},
           "would_hold": {"published": 0, "rejected": 0, "pending": 0}}
    for row in c.execute("SELECT state, auto_review FROM submissions WHERE auto_review IS NOT NULL"):
        record = json.loads(row["auto_review"])
        if not record.get("shadow"):
            continue
        verdict = "would_hold" if record["reasons"] else "would_publish"
        outcome = row["state"] if row["state"] in ("published", "rejected") else "pending"
        out[verdict][outcome] += 1
    decided = sum(out[v][o] for v in out for o in ("published", "rejected"))
    agreed = out["would_publish"]["published"] + out["would_hold"]["rejected"]
    out["decided"] = decided
    out["agreement"] = agreed / decided if decided else None
    return out
