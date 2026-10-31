"""Local submission portal. A separate serial worker evaluates validated audio."""

import asyncio
import hashlib
import json
import os
import secrets
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from .auth import install_auth
from .contract import (
    BENCHMARK,
    MAX_ARCHIVE,
    SENTENCES,
    sentence_pack,
    validate_archive,
)
from .evaluation import validate_report
from .offline import validate_import
from .storage import db, event, initialize, unpack

STATIC = Path(__file__).parent / "static"


def create_app(root: Path, password: str, origin="http://127.0.0.1:5193"):
    if len(password) < 12:
        raise ValueError("Set NEPTTS_ADMIN_PASSWORD to at least 12 characters.")
    initialize(root)
    app = FastAPI(
        title="Ampixa Benchmarks Arena submissions", docs_url=None, redoc_url=None
    )
    app.state.root = root
    sessions = {}
    upload_gate = asyncio.Semaphore(2)

    def rate(key, limit, window=3600):
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM rate_limits WHERE key=?", (key,)).fetchone()
            now = time.time()
            if row and now - row["start"] < window:
                if row["count"] >= limit:
                    raise HTTPException(
                        429, "Too many requests. Please try again later."
                    )
                c.execute("UPDATE rate_limits SET count=count+1 WHERE key=?", (key,))
            else:
                c.execute(
                    "INSERT OR REPLACE INTO rate_limits VALUES(?,?,1)", (key, now)
                )
            c.execute("DELETE FROM rate_limits WHERE start<?", (now - 86400,))

    current_user, require_user = install_auth(app, root, origin, rate)

    @app.middleware("http")
    async def guards(request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            if request.headers.get("origin") not in (None, origin):
                return JSONResponse(
                    {"detail": "Cross-origin requests are not allowed."},
                    status_code=403,
                )
            if request.headers.get("x-neptts-request") != "submission":
                return JSONResponse(
                    {"detail": "Missing request header."}, status_code=403
                )
        response = await call_next(request)
        response.headers.update(
            {
                "X-Content-Type-Options": "nosniff",
                "Referrer-Policy": "no-referrer",
                "Cache-Control": "no-store",
                "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
            }
        )
        return response

    def admin(request: Request):
        token = request.cookies.get("neptts_admin", "")
        if sessions.get(hashlib.sha256(token.encode()).hexdigest(), 0) < time.time():
            raise HTTPException(401, "Sign in to review submissions.")
        return True

    def row_for(sid):
        with db(root) as c:
            row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
        if not row:
            raise HTTPException(404, "Submission not found.")
        return row

    def authorized_row(sid, request):
        row = row_for(sid)
        identity = current_user(request)
        if identity and row["owner_id"] == identity["github_id"]:
            return row
        token = request.headers.get("x-submission-token", "")
        if not secrets.compare_digest(
            hashlib.sha256(token.encode()).hexdigest(), row["token_hash"]
        ):
            try:
                admin(request)
            except HTTPException:
                raise HTTPException(404, "Submission not found.") from None
        return row

    @app.get("/api/benchmark")
    def benchmark():
        return {
            "id": BENCHMARK,
            "count": len(SENTENCES),
            "max_upload_mib": MAX_ARCHIVE // 1024**2,
            "sentences": list(SENTENCES.values()),
            "profile_note": "Automated CER/WER and human MOS are separate tracks.",
        }

    @app.get("/api/benchmark/pack")
    def pack():
        return Response(
            sentence_pack(),
            media_type="application/zip",
            headers={
                "Content-Disposition": 'attachment; filename="neptts-sentence-pack.zip"'
            },
        )

    @app.post("/api/submissions", status_code=201)
    async def submit(request: Request):
        identity = require_user(request)
        rate("upload-user:" + identity["github_id"], 10)
        rate("upload:" + request.client.host, 10)
        if request.headers.get("content-type", "").split(";")[0] != "application/zip":
            raise HTTPException(415, "Upload a ZIP archive.")
        async with upload_gate:
            if shutil.disk_usage(root).free < 2 * 1024**3:
                raise HTTPException(
                    503, "Submission storage is temporarily unavailable."
                )
            sid = uuid.uuid4().hex
            work = Path(tempfile.mkdtemp(prefix="incoming-", dir=root))
            target = root / "uploads" / sid
            try:
                archive = work / "submission.zip"
                size = 0
                with archive.open("wb") as f:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > MAX_ARCHIVE:
                            raise HTTPException(413, "Archive exceeds 150 MiB.")
                        f.write(chunk)
                try:
                    manifest, validation = await asyncio.to_thread(
                        validate_archive, archive, work / "audio"
                    )
                except (ValueError, zipfile.BadZipFile, RuntimeError, OSError) as error:
                    raise HTTPException(422, str(error)[:1200]) from None
                token = secrets.token_urlsafe(32)
                now = time.time()
                with db(root) as c:
                    c.execute("BEGIN IMMEDIATE")
                    if (
                        c.execute("SELECT COUNT(*) FROM submissions").fetchone()[0]
                        >= 100
                    ):
                        raise HTTPException(
                            503, "Submission capacity reached. Contact Ampixa."
                        )
                    if c.execute(
                        "SELECT 1 FROM submissions WHERE audio_sha256=?",
                        (validation["audio_sha256"],),
                    ).fetchone():
                        raise HTTPException(
                            409,
                            "This audio package was already submitted. Use its saved receipt.",
                        )
                    shutil.move(str(work / "audio"), target)
                    c.execute(
                        "INSERT INTO submissions(id,token_hash,created,updated,state,manifest,validation,audio_sha256,message,owner_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                        (
                            sid,
                            hashlib.sha256(token.encode()).hexdigest(),
                            now,
                            now,
                            "validated",
                            json.dumps(manifest),
                            json.dumps(validation),
                            validation["audio_sha256"],
                            "Audio checks passed. Waiting for an evaluator to be assigned.",
                            identity["github_id"],
                        ),
                    )
                    event(
                        c,
                        sid,
                        "validated",
                        f"All {len(SENTENCES)} prompts passed audio checks.",
                    )
                return {
                    "id": sid,
                    "token": token,
                    "state": "validated",
                    "validation": {
                        k: v for k, v in validation.items() if k != "per_file"
                    },
                }
            except Exception:
                if target.exists():
                    shutil.rmtree(target)
                raise
            finally:
                shutil.rmtree(work, ignore_errors=True)

    @app.get("/api/submissions/{sid}")
    def status(sid: str, request: Request):
        d = unpack(authorized_row(sid, request))
        with db(root) as c:
            d["events"] = [
                dict(x)
                for x in c.execute(
                    "SELECT time,state,message FROM events WHERE submission_id=? ORDER BY id",
                    (sid,),
                )
            ]
        d["validation"].pop("per_file", None)
        return d

    @app.get("/api/submissions/{sid}/audio/{sent_id}")
    def audio(sid: str, sent_id: str, request: Request):
        authorized_row(sid, request)
        if sent_id not in SENTENCES:
            raise HTTPException(404)
        return FileResponse(
            root / "uploads" / sid / f"{sent_id}.wav", media_type="audio/wav"
        )

    @app.get("/api/leaderboard")
    def leaderboard():
        with db(root) as c:
            rows = c.execute(
                "SELECT * FROM submissions WHERE state='published' ORDER BY created DESC"
            ).fetchall()
        entries = []
        for row in rows:
            d = unpack(row)
            entries.append(
                {
                    k: d[k]
                    for k in (
                        "id",
                        "created",
                        "updated",
                        "manifest",
                        "report",
                        "profile",
                    )
                }
            )
        return {"benchmark": BENCHMARK, "entries": entries}

    @app.get("/api/leaderboard/{sid}/report")
    def report(sid: str):
        row = row_for(sid)
        if row["state"] != "published":
            raise HTTPException(404)
        d = unpack(row)
        return {
            "id": sid,
            "manifest": d["manifest"],
            "report": d["report"],
            "audio_sha256": d["audio_sha256"],
        }

    class Login(BaseModel):
        password: str = Field(max_length=200)

    @app.post("/api/admin/login")
    def login(body: Login, request: Request):
        rate("login:" + request.client.host, 10, 900)
        if not secrets.compare_digest(body.password, password):
            raise HTTPException(401, "Incorrect password.")
        now = time.time()
        for key in list(sessions):
            if sessions[key] < now:
                sessions.pop(key)
        token = secrets.token_urlsafe(32)
        sessions[hashlib.sha256(token.encode()).hexdigest()] = now + 8 * 3600
        response = JSONResponse({"ok": True})
        response.set_cookie(
            "neptts_admin",
            token,
            httponly=True,
            samesite="strict",
            secure=origin.startswith("https://"),
            max_age=8 * 3600,
        )
        return response

    @app.post("/api/admin/logout")
    def logout(request: Request):
        sessions.pop(
            hashlib.sha256(
                request.cookies.get("neptts_admin", "").encode()
            ).hexdigest(),
            None,
        )
        response = JSONResponse({"ok": True})
        response.delete_cookie("neptts_admin")
        return response

    @app.get("/api/admin/submissions", dependencies=[Depends(admin)])
    def review_list():
        with db(root) as c:
            rows = c.execute(
                "SELECT * FROM submissions ORDER BY created DESC"
            ).fetchall()
        return [unpack(x) for x in rows]

    @app.get("/api/admin/submissions/{sid}/package", dependencies=[Depends(admin)])
    def evaluation_package(sid: str):
        with db(root) as c:
            row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
        if not row:
            raise HTTPException(404)
        d = unpack(row)
        work = Path(tempfile.mkdtemp(prefix="reviewer-export-"))
        path = work / "evaluation.zip"
        try:
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
                z.writestr("submission.json", json.dumps(d["manifest"]))
                z.writestr(
                    "evaluation-job.json",
                    json.dumps(
                        {
                            "submission_id": sid,
                            "benchmark": BENCHMARK,
                            "audio_sha256": row["audio_sha256"],
                        }
                    ),
                )
                for sent_id, info in d["validation"]["per_file"].items():
                    audio = root / "uploads" / sid / f"{sent_id}.wav"
                    if hashlib.sha256(audio.read_bytes()).hexdigest() != info["sha256"]:
                        raise HTTPException(409, "Stored audio integrity check failed.")
                    z.write(audio, f"audio/{sent_id}.wav")
            return FileResponse(
                path,
                filename=f"evaluation-{sid}.zip",
                media_type="application/zip",
                background=BackgroundTask(shutil.rmtree, work),
                headers={"Cache-Control": "no-store"},
            )
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise

    @app.post("/api/admin/submissions/{sid}/results", dependencies=[Depends(admin)])
    async def import_results(sid: str, request: Request):
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > 2 * 1024**2:
                raise HTTPException(413, "Result JSON must be at most 2 MiB.")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise HTTPException(422, "Upload a valid result JSON file.") from None
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
            if not row:
                raise HTTPException(404)
            if row["state"] not in ("validated", "evaluation_failed", "queued"):
                raise HTTPException(
                    409,
                    "Results can only be imported before evaluation/review completes.",
                )
            try:
                report = validate_import(payload, sid, row["audio_sha256"])
            except (ValueError, TypeError, AttributeError, KeyError) as error:
                raise HTTPException(422, str(error)) from None
            message = "Reviewer-uploaded evaluation passed consistency checks. Awaiting approval."
            c.execute(
                "UPDATE submissions SET state='awaiting_review',report=?,profile=?,progress=?,message=?,updated=? WHERE id=?",
                (
                    json.dumps(report, allow_nan=False),
                    report["profile"]["id"],
                    len(SENTENCES),
                    message,
                    time.time(),
                    sid,
                ),
            )
            event(c, sid, "awaiting_review", message)
        return {"id": sid, "state": "awaiting_review"}

    class Decision(BaseModel):
        action: str = Field(pattern="^(queue|approve|reject)$")
        note: str = Field(min_length=3, max_length=1000)

    @app.post("/api/admin/submissions/{sid}/decision", dependencies=[Depends(admin)])
    def decide(sid: str, body: Decision):
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
            if not row:
                raise HTTPException(404)
            if body.action == "queue":
                if row["state"] not in ("validated", "evaluation_failed"):
                    raise HTTPException(
                        409, "Only validated or failed evaluations can be queued."
                    )
                state, message = "queued", "Queued for the serial evaluation worker."
            elif body.action == "approve":
                if row["state"] != "awaiting_review" or not row["report"]:
                    raise HTTPException(
                        409, "A complete evaluation is required before approval."
                    )
                try:
                    validate_report(json.loads(row["report"]), row["profile"])
                except ValueError as error:
                    raise HTTPException(409, str(error)) from None
                if json.loads(row["report"])["profile"].get("development_only"):
                    raise HTTPException(
                        409,
                        "Development evaluator results cannot be published. Use the small evaluation profile.",
                    )
                state, message = (
                    "published",
                    "Reviewed and published on the leaderboard.",
                )
            else:
                if row["state"] in ("evaluating", "published", "rejected"):
                    raise HTTPException(
                        409, "This submission cannot be rejected in its current state."
                    )
                state, message = (
                    "rejected",
                    "Not accepted for publication. See the review note.",
                )
            c.execute(
                "UPDATE submissions SET state=?,message=?,reviewer_note=?,updated=? WHERE id=?",
                (state, message, body.note, time.time(), sid),
            )
            event(c, sid, state, body.note)
        return {"id": sid, "state": state}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def main():
    import uvicorn

    root = Path(os.environ.get("NEPTTS_DATA", ".local/submissions"))
    origin = os.environ.get("NEPTTS_ORIGIN", "http://127.0.0.1:5193")
    app = create_app(root, os.environ.get("NEPTTS_ADMIN_PASSWORD", ""), origin)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=int(os.environ.get("NEPTTS_PORT", "5193")),
        access_log=False,
    )


if __name__ == "__main__":
    main()
