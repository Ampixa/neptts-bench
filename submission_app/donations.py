"""Voice donation and paid voice programme.

Donors enrol anonymously under a versioned consent document, record prompts in
the browser, and get a one-time recovery code instead of an account. Each upload
passes instant signal checks (donation_audio.py). The worker then checks with ASR
whether the prompt was read (CER thresholds below). Borderline takes go to a
reviewer.

Prompt sets:
- anchor-v1: the arena's benchmark prompts. Evaluation-only: these recordings are
  human reference speech and real-speech judge calibration, never training data.
- open sets: operator-supplied, licensed sentences in benchmark/donation_prompts.json.
  Training or licensing use needs the donor's explicit purpose consent.

Paid programme: donors apply with a payout handle (eSewa, Khalti or bank). A
reviewer approves them at a per-minute rate. Earnings are accepted minutes times
the rate, minus payouts recorded in a ledger.

Enrollment stays closed until the consent document's status is "approved".
"""

import hashlib
import json
import os
import random
import re
import secrets
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from . import donation_audio, donation_store
from .contract import ROOT, SENTENCES
from .storage import db

COOKIE = "donor_session"
CONSENT_PATH = Path(__file__).with_name("consent") / "voice_donation_v1.json"
OPEN_SETS_PATH = ROOT / "benchmark/donation_prompts.json"
ANCHOR = "anchor-v1"
ACCEPT_CER, REVIEW_CER = 0.15, 0.40
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

AGE_BANDS = Literal["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
GENDERS = Literal["female", "male", "other", "prefer_not"]
PROVINCES = Literal["koshi", "madhesh", "bagmati", "gandaki", "lumbini", "karnali",
                    "sudurpashchim", "outside_nepal", "prefer_not"]
LANGUAGES = Literal["nepali", "maithili", "bhojpuri", "tharu", "tamang", "newar", "magar",
                    "bajjika", "doteli", "other"]
DEVICES = Literal["phone", "laptop", "headset", "external_mic", "other"]


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def load_consent(path=CONSENT_PATH):
    raw = path.read_bytes()
    doc = json.loads(raw)
    return doc, hashlib.sha256(raw).hexdigest()


def load_prompt_sets(open_path=OPEN_SETS_PATH):
    """Anchor set from the benchmark, plus validated operator-supplied open sets."""
    sets = {ANCHOR: {
        "id": ANCHOR, "kind": "evaluation_only",
        "description": "Benchmark prompts. Recordings are human reference speech for evaluation only.",
        "prompts": {sid: s["text_devanagari"] for sid, s in SENTENCES.items()},
    }}
    if open_path.exists():
        data = json.loads(open_path.read_text(encoding="utf-8"))
        for entry in data.get("sets", []):
            sid = entry["id"]
            if sid in sets or not re.fullmatch(r"open-[a-z0-9-]{1,40}", sid):
                raise ValueError(f"Invalid or duplicate open prompt set id: {sid}")
            if not entry.get("license") or not entry.get("source"):
                raise ValueError(f"Open prompt set {sid} must declare its source and license.")
            prompts = {}
            for p in entry["prompts"]:
                text = p["text"].strip()
                if p["id"] in prompts or not DEVANAGARI.search(text) or len(text) > 200:
                    raise ValueError(f"Invalid prompt {p['id']} in {sid}.")
                if text in sets[ANCHOR]["prompts"].values():
                    raise ValueError(f"Prompt {p['id']} in {sid} duplicates a benchmark prompt.")
                prompts[p["id"]] = text
            sets[sid] = {"id": sid, "kind": "open", "description": entry.get("description", ""),
                         "license": entry["license"], "source": entry["source"], "prompts": prompts}
    return sets


def asr_state(cer):
    if cer <= ACCEPT_CER:
        return "accepted"
    return "needs_review" if cer <= REVIEW_CER else "rejected"


def apply_asr(root, rec_id, cer, profile_id):
    """Record a worker ASR result. Returns the new state, or None if the take is gone."""
    with db(root) as c:
        c.execute("BEGIN IMMEDIATE")
        row = c.execute("SELECT state FROM donor_recordings WHERE id=?", (rec_id,)).fetchone()
        if row is None or row["state"] != "asr_pending":
            return None
        state = asr_state(cer)
        c.execute("UPDATE donor_recordings SET state=?, cer=?, asr_profile=? WHERE id=?",
                  (state, cer, profile_id, rec_id))
    return state


def next_asr_job(root, sets, skip=()):
    """Oldest take awaiting ASR (ignoring ones that failed this run), with its row and text."""
    with db(root) as c:
        rows = c.execute(
            "SELECT * FROM donor_recordings WHERE state='asr_pending' ORDER BY created"
        ).fetchall()
    row = next((r for r in rows if r["id"] not in skip), None)
    if row is None:
        return None
    text = sets.get(row["prompt_set"], {}).get("prompts", {}).get(row["prompt_id"])
    return {"id": row["id"], "rec": row, "text": text}


def install_donations(app, root, rate, origin, admin):
    secure = origin.startswith("https://")
    app.state.consent_path = CONSENT_PATH
    app.state.prompt_sets = load_prompt_sets()
    rate_env = os.environ.get("NEPTTS_DONOR_RATE_NPR_PER_MIN")
    app.state.donor_rate = float(rate_env) if rate_env else None
    store = app.state.donation_store = donation_store.from_env(
        root, open_sets=any(s["kind"] == "open" for s in app.state.prompt_sets.values()))
    if store.remote:
        donation_store.start_sync_thread(app, root)

    def current_store():
        return app.state.donation_store  # tests swap in a store with a fake Hugging Face

    def consent():
        return load_consent(app.state.consent_path)

    def is_open():
        return consent()[0].get("status") == "approved"

    def donor(request, c, required=True):
        token = request.cookies.get(COOKIE, "")
        row = None
        if token:
            row = c.execute(
                "SELECT * FROM donors WHERE session_hash=? AND withdrawn IS NULL", (digest(token),)
            ).fetchone()
        if row is None and required:
            raise HTTPException(401, "Join the programme or recover your session first.")
        return row

    def issue_session(response, c, donor_id):
        token = secrets.token_urlsafe(32)
        c.execute("UPDATE donors SET session_hash=? WHERE id=?", (digest(token), donor_id))
        response.set_cookie(COOKIE, token, max_age=365 * 86400, httponly=True,
                            samesite="lax", secure=secure, path="/")

    async def read_body(request, limit):
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > limit:
                raise HTTPException(413, "Recording is too large.")
        return bytes(body)

    def summary(c, row):
        recs = c.execute(
            "SELECT * FROM donor_recordings WHERE donor_id=? ORDER BY created DESC", (row["id"],)
        ).fetchall()
        counts = {s: 0 for s in ("asr_pending", "accepted", "needs_review", "rejected")}
        for r in recs:
            counts[r["state"]] += 1
        minutes = sum(r["seconds"] for r in recs if r["state"] == "accepted") / 60
        paid = c.execute(
            "SELECT COALESCE(SUM(amount_npr),0) FROM donor_payouts WHERE donor_id=?", (row["id"],)
        ).fetchone()[0]
        earned = minutes * row["rate_npr_per_minute"] if row["paid_status"] == "approved" else 0.0
        sets = app.state.prompt_sets
        return {
            "id": row["id"],
            "purposes": json.loads(row["purposes"]),
            "profile": json.loads(row["profile"]),
            "mic_check": json.loads(row["mic_check"]) if row["mic_check"] else None,
            "paid_status": row["paid_status"],
            "rate_npr_per_minute": row["rate_npr_per_minute"],
            "counts": counts,
            "accepted_minutes": round(minutes, 2),
            "earned_npr": round(earned, 2),
            "paid_npr": round(paid, 2),
            "balance_npr": round(earned - paid, 2),
            "recordings": [{
                "id": r["id"], "prompt_set": r["prompt_set"], "prompt_id": r["prompt_id"],
                "text": sets.get(r["prompt_set"], {}).get("prompts", {}).get(r["prompt_id"], ""),
                "state": r["state"], "seconds": r["seconds"], "created": r["created"],
                "reviewer_note": r["reviewer_note"],
            } for r in recs],
        }

    # ------------------------------------------------------------------ public

    @app.get("/api/donate/info")
    def info(request: Request):
        doc, sha = consent()
        with db(root) as c:
            row = donor(request, c, required=False)
            me = summary(c, row) if row else None
        return {
            "open": doc.get("status") == "approved",
            "consent": {**doc, "sha256": sha},
            "prompt_sets": [{k: v for k, v in s.items() if k != "prompts"} | {"count": len(s["prompts"])}
                            for s in app.state.prompt_sets.values()],
            "paid_programme": {"open": app.state.donor_rate is not None,
                               "rate_npr_per_minute": app.state.donor_rate},
            "thresholds": {"min_snr_db": donation_audio.MIN_SNR_DB,
                           "max_seconds": donation_audio.MAX_SECONDS},
            "me": me,
        }

    class Enroll(BaseModel):
        model_config = ConfigDict(extra="forbid")
        consent_version: str
        consent_sha256: str
        adult: Literal[True]
        own_voice: Literal[True]
        purposes: list[str] = Field(max_length=10)
        age_band: AGE_BANDS
        gender: GENDERS
        province: PROVINCES
        first_language: LANGUAGES
        device: DEVICES

    @app.post("/api/donate/enroll")
    def enroll(body: Enroll, request: Request, response: Response):
        doc, sha = consent()
        if doc.get("status") != "approved":
            raise HTTPException(403, "Voice donation opens once the consent text is approved.")
        if body.consent_version != doc["version"] or body.consent_sha256 != sha:
            raise HTTPException(409, "The consent text changed. Reload the page and read it again.")
        known = {p["id"]: p for p in doc["purposes"]}
        purposes = sorted(set(body.purposes))
        if any(p not in known for p in purposes):
            raise HTTPException(422, "Unknown purpose.")
        missing = [p for p, spec in known.items() if spec.get("required") and p not in purposes]
        if missing:
            raise HTTPException(422, "Required consent missing: " + ", ".join(missing))
        rate("donor-enroll:" + (request.client.host if request.client else "?"), 20)
        recovery = "-".join(secrets.token_hex(3) for _ in range(4))
        donor_id = uuid.uuid4().hex
        profile = body.model_dump(include={"age_band", "gender", "province", "first_language", "device"})
        with db(root) as c:
            c.execute(
                "INSERT INTO donors(id,recovery_hash,created,consent_version,consent_sha256,purposes,profile) "
                "VALUES(?,?,?,?,?,?,?)",
                (donor_id, digest(recovery), time.time(), doc["version"], sha,
                 json.dumps(purposes), json.dumps(profile)),
            )
            issue_session(response, c, donor_id)
        return {"recovery_code": recovery,
                "note": "Save this code. It is the only way to reach your recordings from another browser."}

    class Recover(BaseModel):
        code: str = Field(min_length=10, max_length=40)

    @app.post("/api/donate/recover")
    def recover(body: Recover, request: Request, response: Response):
        rate("donor-recover:" + (request.client.host if request.client else "?"), 10)
        with db(root) as c:
            row = c.execute(
                "SELECT id FROM donors WHERE recovery_hash=? AND withdrawn IS NULL",
                (digest(body.code.strip().lower()),),
            ).fetchone()
            if not row:
                raise HTTPException(404, "Recovery code not recognised.")
            issue_session(response, c, row["id"])
        return {"recovered": True}

    @app.get("/api/donate/me")
    def me(request: Request):
        with db(root) as c:
            return summary(c, donor(request, c))

    @app.post("/api/donate/mic-check")
    async def mic_check(request: Request):
        body = await read_body(request, donation_audio.MAX_BYTES)
        metrics, problems = donation_audio.measure(body)
        result = {"metrics": metrics, "problems": problems,
                  "messages": [donation_audio.MESSAGES[p] for p in problems],
                  "passed": not problems, "time": time.time()}
        with db(root) as c:
            row = donor(request, c)
            c.execute("UPDATE donors SET mic_check=? WHERE id=?", (json.dumps(result), row["id"]))
        return result

    @app.get("/api/donate/next")
    def next_prompt(request: Request, prompt_set: str = ANCHOR):
        sets = app.state.prompt_sets
        if prompt_set not in sets:
            raise HTTPException(404, "Unknown prompt set.")
        prompts = sets[prompt_set]["prompts"]
        with db(root) as c:
            row = donor(request, c)
            mine = {r["prompt_id"] for r in c.execute(
                "SELECT prompt_id FROM donor_recordings WHERE donor_id=? AND prompt_set=? AND state!='rejected'",
                (row["id"], prompt_set),
            )}
            coverage = {}
            for r in c.execute(
                "SELECT prompt_id, COUNT(*) AS n FROM donor_recordings "
                "WHERE prompt_set=? AND state IN ('accepted','asr_pending','needs_review') GROUP BY prompt_id",
                (prompt_set,),
            ):
                coverage[r["prompt_id"]] = r["n"]
        todo = [p for p in prompts if p not in mine]
        if not todo:
            return {"done": True, "remaining": 0}
        fewest = min(coverage.get(p, 0) for p in todo)
        choice = random.choice([p for p in todo if coverage.get(p, 0) == fewest])
        return {"done": False, "prompt_set": prompt_set, "prompt_id": choice,
                "text": prompts[choice], "remaining": len(todo)}

    @app.post("/api/donate/recordings/{prompt_set}/{prompt_id}")
    async def upload(prompt_set: str, prompt_id: str, request: Request):
        if prompt_id not in app.state.prompt_sets.get(prompt_set, {}).get("prompts", {}):
            raise HTTPException(404, "Unknown prompt.")
        with db(root) as c:
            row = donor(request, c)
        rate("donor-upload:" + row["id"], 400)
        body = await read_body(request, donation_audio.MAX_BYTES)
        metrics, problems = donation_audio.measure(body)
        store = current_store()
        try:
            store.check_capacity()
        except donation_store.StorageUnavailable as error:
            raise HTTPException(503, str(error)) from None
        if problems:
            return JSONResponse(status_code=422, content={
                "detail": " ".join(donation_audio.MESSAGES[p] for p in problems),
                "problems": problems, "metrics": metrics})
        rec_id = uuid.uuid4().hex
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            old = c.execute(
                "SELECT * FROM donor_recordings WHERE donor_id=? AND prompt_set=? AND prompt_id=?",
                (row["id"], prompt_set, prompt_id),
            ).fetchone()
            if old and old["state"] == "accepted":
                raise HTTPException(409, "This sentence is already accepted. Delete it first to re-record.")
            stored = store.save(row["id"], rec_id, body)
            if old:
                c.execute("DELETE FROM donor_recordings WHERE id=?", (old["id"],))
                store.remove(c, old)
            c.execute(
                "INSERT INTO donor_recordings(id,donor_id,prompt_set,prompt_id,created,seconds,sha256,signal,state,stored) "
                "VALUES(?,?,?,?,?,?,?,?,'asr_pending',?)",
                (rec_id, row["id"], prompt_set, prompt_id, time.time(), metrics["seconds"],
                 hashlib.sha256(body).hexdigest(), json.dumps(metrics), stored),
            )
        return {"id": rec_id, "state": "asr_pending", "signal": metrics}

    def own_recording(rec_id, request, c):
        row = donor(request, c)
        rec = c.execute(
            "SELECT * FROM donor_recordings WHERE id=? AND donor_id=?", (rec_id, row["id"])
        ).fetchone()
        if not rec:
            raise HTTPException(404, "Recording not found.")
        return rec

    def wav(rec):
        try:
            data = current_store().read(rec)
        except FileNotFoundError:
            raise HTTPException(404, "Recording not found.") from None
        except Exception:  # noqa: BLE001 - any Hub/network failure is an outage to the listener
            donation_store.log.exception("Reading take %s failed", rec["id"])
            raise HTTPException(503, "Recording storage is temporarily unavailable.") from None
        return Response(data, media_type="audio/wav", headers={"Cache-Control": "no-store"})

    @app.get("/api/donate/recordings/{rec_id}/audio")
    def my_audio(rec_id: str, request: Request):
        with db(root) as c:
            rec = own_recording(rec_id, request, c)
        return wav(rec)

    @app.delete("/api/donate/recordings/{rec_id}")
    def delete_recording(rec_id: str, request: Request):
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            rec = own_recording(rec_id, request, c)
            c.execute("DELETE FROM donor_recordings WHERE id=?", (rec_id,))
            current_store().remove(c, rec)
        return {"deleted": rec_id}

    @app.post("/api/donate/withdraw")
    def withdraw(request: Request, response: Response):
        """Delete every recording and the profile. The payout ledger is a financial record and stays."""
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = donor(request, c)
            for rec in c.execute("SELECT * FROM donor_recordings WHERE donor_id=?", (row["id"],)).fetchall():
                current_store().remove(c, rec)
            c.execute("DELETE FROM donor_recordings WHERE donor_id=?", (row["id"],))
            c.execute(
                "UPDATE donors SET session_hash=NULL, purposes='[]', profile='{}', mic_check=NULL, "
                "payout=NULL, withdrawn=? WHERE id=?",
                (time.time(), row["id"]),
            )
        for name in ("donations", "donation_spool"):
            folder = root / name / row["id"]
            if folder.is_dir() and not any(folder.iterdir()):
                folder.rmdir()
        response.delete_cookie(COOKIE, path="/")
        return {"withdrawn": True}

    class Apply(BaseModel):
        model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
        method: Literal["esewa", "khalti", "bank"]
        handle: str = Field(min_length=5, max_length=120)
        name: str = Field(min_length=2, max_length=120)

    @app.post("/api/donate/paid/apply")
    def apply(body: Apply, request: Request):
        if app.state.donor_rate is None:
            raise HTTPException(403, "The paid programme is not open.")
        with db(root) as c:
            row = donor(request, c)
            if row["paid_status"] in ("applied", "approved"):
                raise HTTPException(409, "You have already applied.")
            c.execute("UPDATE donors SET paid_status='applied', payout=? WHERE id=?",
                      (json.dumps(body.model_dump()), row["id"]))
        return {"paid_status": "applied"}

    # ------------------------------------------------------------------ reviewer

    @app.get("/api/admin/donations", dependencies=[Depends(admin)])
    def admin_overview():
        sets = app.state.prompt_sets
        with db(root) as c:
            donors = c.execute(
                "SELECT COUNT(*) AS n, SUM(withdrawn IS NOT NULL) AS w FROM donors").fetchone()
            queue = c.execute(
                "SELECT * FROM donor_recordings WHERE state='needs_review' ORDER BY created LIMIT 100"
            ).fetchall()
            applications = c.execute(
                "SELECT * FROM donors WHERE paid_status IN ('applied','approved') AND withdrawn IS NULL"
            ).fetchall()
            coverage = {}
            for r in c.execute(
                "SELECT prompt_set, state, COUNT(DISTINCT prompt_id) AS prompts, SUM(seconds) AS s "
                "FROM donor_recordings GROUP BY prompt_set, state"
            ):
                coverage.setdefault(r["prompt_set"], {})[r["state"]] = {
                    "prompts": r["prompts"], "minutes": round(r["s"] / 60, 2)}
            apps = [{**summary(c, d), "payout": json.loads(d["payout"] or "null")} for d in applications]
        for a in apps:
            a.pop("recordings")
        return {
            "donors": donors["n"], "withdrawn": donors["w"] or 0,
            "open": is_open(), "donor_rate": app.state.donor_rate,
            "coverage": {k: {"prompts_total": len(sets[k]["prompts"]) if k in sets else None, **v}
                         for k, v in coverage.items()},
            "review_queue": [{
                "id": r["id"], "prompt_set": r["prompt_set"], "prompt_id": r["prompt_id"],
                "text": sets.get(r["prompt_set"], {}).get("prompts", {}).get(r["prompt_id"], ""),
                "cer": r["cer"], "signal": json.loads(r["signal"]), "seconds": r["seconds"],
            } for r in queue],
            "applications": apps,
            "storage": current_store().status(root),
        }

    @app.get("/api/admin/donations/recordings/{rec_id}/audio", dependencies=[Depends(admin)])
    def admin_audio(rec_id: str):
        with db(root) as c:
            rec = c.execute("SELECT * FROM donor_recordings WHERE id=?", (rec_id,)).fetchone()
        if not rec:
            raise HTTPException(404)
        return wav(rec)

    class Decision(BaseModel):
        action: Literal["accept", "reject"]
        note: str = Field(min_length=3, max_length=500)

    @app.post("/api/admin/donations/recordings/{rec_id}/decision", dependencies=[Depends(admin)])
    def decide(rec_id: str, body: Decision):
        state = "accepted" if body.action == "accept" else "rejected"
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            rec = c.execute("SELECT state FROM donor_recordings WHERE id=?", (rec_id,)).fetchone()
            if not rec:
                raise HTTPException(404)
            if rec["state"] == "asr_pending":
                raise HTTPException(409, "Wait for the ASR check before deciding.")
            c.execute("UPDATE donor_recordings SET state=?, reviewer_note=?, reviewed=? WHERE id=?",
                      (state, body.note, time.time(), rec_id))
        return {"id": rec_id, "state": state}

    class PaidDecision(BaseModel):
        action: Literal["approve", "reject"]
        rate_npr_per_minute: float | None = Field(default=None, gt=0, le=10000)

    @app.post("/api/admin/donations/donors/{donor_id}/paid", dependencies=[Depends(admin)])
    def decide_paid(donor_id: str, body: PaidDecision):
        with db(root) as c:
            row = c.execute("SELECT paid_status FROM donors WHERE id=? AND withdrawn IS NULL",
                            (donor_id,)).fetchone()
            if not row or row["paid_status"] != "applied":
                raise HTTPException(409, "No pending application.")
            if body.action == "approve":
                rate_value = body.rate_npr_per_minute or app.state.donor_rate
                if rate_value is None:
                    raise HTTPException(422, "Set a per-minute rate.")
                c.execute("UPDATE donors SET paid_status='approved', rate_npr_per_minute=? WHERE id=?",
                          (rate_value, donor_id))
            else:
                c.execute("UPDATE donors SET paid_status='rejected' WHERE id=?", (donor_id,))
        return {"id": donor_id, "paid_status": "approved" if body.action == "approve" else "rejected"}

    class Payout(BaseModel):
        amount_npr: float = Field(gt=0, le=1_000_000)
        reference: str = Field(min_length=3, max_length=120)
        note: str = Field(default="", max_length=300)

    @app.post("/api/admin/donations/donors/{donor_id}/payouts", dependencies=[Depends(admin)])
    def record_payout(donor_id: str, body: Payout):
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM donors WHERE id=?", (donor_id,)).fetchone()
            if not row or row["paid_status"] != "approved":
                raise HTTPException(409, "Donor is not in the paid programme.")
            if body.amount_npr > summary(c, row)["balance_npr"] + 1e-6:
                raise HTTPException(409, "Payout exceeds the unpaid balance.")
            c.execute(
                "INSERT INTO donor_payouts(donor_id,amount_npr,reference,note,created) VALUES(?,?,?,?,?)",
                (donor_id, body.amount_npr, body.reference, body.note, time.time()),
            )
            return {k: v for k, v in summary(c, row).items() if k != "recordings"}
