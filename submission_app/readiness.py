"""Production readiness: what Ampixa still has to set up, and how.

Every check returns a status:
- "ok": done.
- "todo": required before the named feature can launch; `fix` says what to do.
- "warn": works, but not as intended for production.
- "info": a setting worth knowing, not a problem.

Used by `python -m submission_app.doctor` (run on the VM) and the reviewer-only
`GET /api/admin/readiness`. No check makes a network call unless `online=True`.
"""

import json
import os
import shutil
import time
from pathlib import Path

from . import autoreview
from .storage import db

WORKER_STALE = 15 * 60  # a primary worker silent this long is treated as down
BACKUP_STALE = 2 * 86400
MARKER = ".last-backup"  # touched by deploy/arena/backup.sh after a verified backup
CONSENT = Path(__file__).parent / "consent" / "voice_donation_v1.json"


def _check(cid, feature, status, title, detail="", fix=""):
    return {"id": cid, "feature": feature, "status": status, "title": title,
            "detail": detail, "fix": fix}


def checks(root, env=None, online=False):
    env = os.environ if env is None else env
    root = Path(root)
    out = []

    origin = env.get("NEPTTS_ORIGIN", "")
    out.append(_check(
        "origin", "site", "ok" if origin.startswith("https://") else "todo",
        "Public HTTPS origin", origin or "NEPTTS_ORIGIN is unset.",
        "Set NEPTTS_ORIGIN=https://ampixa.benchmarks.cloud in /etc/ampixa-arena/environment."))

    password = env.get("NEPTTS_ADMIN_PASSWORD", "")
    out.append(_check(
        "admin_password", "review", "ok" if len(password) >= 24 else "todo",
        "Strong reviewer password", f"{len(password)} characters.",
        "bootstrap.sh generates one; keep it in a password manager shared with reviewers."))

    oauth = bool(env.get("NEPTTS_GITHUB_CLIENT_ID")) and bool(env.get("NEPTTS_GITHUB_CLIENT_SECRET"))
    out.append(_check(
        "github_oauth", "submissions", "ok" if oauth else "todo",
        "GitHub sign-in for submitters", "" if oauth else "Uploads stay closed until this is set.",
        "Create a GitHub OAuth App owned by the Ampixa org (callback "
        f"{origin or 'https://<origin>'}/api/auth/github/callback) and set NEPTTS_GITHUB_CLIENT_ID/SECRET."))

    try:
        from .profiles import primary_profile_id

        primary = primary_profile_id()
    except Exception as error:  # noqa: BLE001 - doctor must report, not crash: missing toolkit or unreadable contract
        primary, detail = None, f"Could not load the Rani contract: {error}"
    else:
        detail = primary or "rani_contract.json does not pin a checkpoint yet."
    out.append(_check(
        "primary_judge", "scoring", "ok" if primary else "todo", "Rani contract pinned", detail,
        "Review neptts-eval/src/neptts_eval/rani_contract.json (checkpoint sha256 and package "
        "versions) and merge it; the web service only needs the pin, not the weights."))

    with db(root) as c:
        beats = [dict(r) for r in c.execute("SELECT * FROM worker_heartbeats")]
        queued = c.execute("SELECT COUNT(*) FROM submissions WHERE state='queued'").fetchone()[0]
        shadow = autoreview.shadow_agreement(c)
        from .arena import eligible, kinds

        systems = eligible(c)
        first_party = kinds(c)
    now = time.time()
    primary_beats = [b for b in beats if b["role"] == "primary"]
    alive = [b for b in primary_beats if now - b["last_seen"] < WORKER_STALE]
    if alive:
        status, detail = "ok", ", ".join(f"{b['engine']} seen {int(now - b['last_seen'])} s ago" for b in alive)
    elif primary_beats:
        status, detail = "todo", f"Last heartbeat {int(now - max(b['last_seen'] for b in primary_beats))} s ago."
    else:
        status, detail = "todo", "No worker has ever reported in."
    out.append(_check(
        "worker", "scoring", status, "Rani evaluation worker running",
        detail + (f" {queued} submission(s) queued." if queued else ""),
        "On a machine with ≥4 GiB RAM and access to the data directory, install "
        "requirements-worker-rani.txt and enable deploy/arena/ampixa-arena-worker.service."))

    mode = autoreview.mode()
    agreement = shadow["agreement"]
    detail = f"Mode `{mode}`."
    if shadow["decided"]:
        detail += f" Shadow decisions reviewed: {shadow['decided']}, agreement {agreement:.0%}."
    out.append(_check(
        "auto_review", "review", "info" if mode != "on" or shadow["decided"] >= 10 else "warn",
        "Automatic review", detail,
        "Start with NEPTTS_AUTO_REVIEW=shadow; switch to `on` once at least 10 shadow "
        "decisions agree with reviewers."))

    n_kinds = {k: sum(v == k for v in first_party.values()) for k in ("baseline", "human_reference")}
    status = "ok" if len(systems) >= 4 and n_kinds["human_reference"] else ("warn" if len(systems) >= 2 else "todo")
    out.append(_check(
        "arena_systems", "arena", status, "Systems in the blind arena",
        f"{len(systems)} eligible; {n_kinds['baseline']} Ampixa baseline(s), "
        f"{n_kinds['human_reference']} human reference(s).",
        "Seed baselines and a human reading with `python -m submission_app.seed --help`. "
        "The arena needs 2 systems to open; 4+ including a human reference makes the scale meaningful."))

    doc = json.loads(CONSENT.read_text())
    approved = doc.get("status") == "approved"
    out.append(_check(
        "donation_consent", "donations", "ok" if approved else "todo",
        "Voice-donation consent approved", f"{doc.get('version')}: status {doc.get('status')}.",
        "Legal and native-speaker review of consent/voice_donation_v1.json (name Hugging Face as "
        "storage provider, payout terms, withdrawal), then set status to \"approved\"."))

    repo = env.get("NEPTTS_HF_DONATIONS_BENCHMARK_REPO", "").strip()
    token = bool(env.get("HF_TOKEN"))
    if repo and token:
        status, detail = "ok", repo
    elif approved:
        status, detail = "todo", "Donations would be kept on this VM's disk."
    else:
        status, detail = "warn", "Not needed until donations open."
    out.append(_check(
        "hf_storage", "donations", status, "Private Hugging Face storage for donations", detail,
        "Run `python scripts/create_hf_donation_repos.py --org Ampixa` with an org admin token, "
        "then set NEPTTS_HF_DONATIONS_BENCHMARK_REPO, NEPTTS_HF_DONATIONS_OPEN_REPO and a "
        "fine-grained HF_TOKEN (write access to those two repos only)."))
    if online and repo and token:
        try:
            from huggingface_hub import HfApi

            info = HfApi(token=env["HF_TOKEN"]).repo_info(repo, repo_type="dataset")
            out.append(_check("hf_private", "donations", "ok" if info.private else "todo",
                              "Donation dataset is private", repo,
                              "Make the dataset private; the sync refuses public repos."))
        except Exception as error:  # noqa: BLE001 - any Hub/auth/network failure is reported as a todo
            out.append(_check("hf_private", "donations", "todo", "Donation dataset reachable",
                              str(error)[:300], "Check the token's repo permissions."))

    ffmpeg = shutil.which("ffmpeg")
    out.append(_check(
        "ffmpeg", "arena", "ok" if ffmpeg else "todo", "FFmpeg installed", ffmpeg or "Not on PATH.",
        "apt install ffmpeg (arena clips are loudness-matched and re-encoded with it)."))

    free = shutil.disk_usage(root).free
    out.append(_check(
        "disk", "site", "ok" if free > 5 * 1024**3 else "warn", "Free disk space",
        f"{free / 1024**3:.1f} GiB free.", "Keep at least 5 GiB free; uploads refuse below 2 GiB."))

    marker = root / MARKER
    age = now - marker.stat().st_mtime if marker.exists() else None
    out.append(_check(
        "backup", "site", "ok" if age is not None and age < BACKUP_STALE else "todo",
        "Recent off-host backup",
        f"Last verified backup {age / 3600:.0f} h ago." if age is not None else "No backup recorded.",
        "Install deploy/arena/backup.sh as a daily timer (see PRODUCTION_SETUP.md) with an "
        "encrypted off-host destination."))

    export = env.get("NEPTTS_ARENA_PUBLIC_EXPORT", "off").lower() in ("on", "1", "true")
    out.append(_check(
        "open_votes", "arena", "info", "Public vote export",
        "On: /api/arena/votes.csv is public." if export else "Off: reviewers only.",
        "Decide whether pseudonymised votes are published (recommended once the privacy notice says so)."))
    return out


def summary(results):
    blocking = [r for r in results if r["status"] == "todo"]
    features = sorted({r["feature"] for r in results})
    return {"ready": not blocking,
            "features": {f: all(r["status"] != "todo" for r in results if r["feature"] == f) for f in features},
            "checks": results}
