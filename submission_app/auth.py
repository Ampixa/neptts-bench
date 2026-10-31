"""GitHub identity only: no repository permissions or retained provider tokens."""

import base64
import hashlib
import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from .storage import db, unpack


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def install_auth(app, root, origin, rate):
    client_id = os.environ.get("NEPTTS_GITHUB_CLIENT_ID", "")
    client_secret = os.environ.get("NEPTTS_GITHUB_CLIENT_SECRET", "")
    configured = bool(client_id and client_secret)
    callback = origin.rstrip("/") + "/api/auth/github/callback"
    secure = origin.startswith("https://")

    def user(request):
        with db(root) as c:
            row = c.execute(
                "SELECT github_id,login FROM user_sessions WHERE token_hash=? AND expires>?",
                (digest(request.cookies.get("arena_user", "")), time.time()),
            ).fetchone()
        return dict(row) if row else None

    def require_user(request):
        identity = user(request)
        if not identity:
            raise HTTPException(401, "Sign in with GitHub before submitting audio.")
        return identity

    @app.get("/api/auth/me")
    def me(request: Request):
        return {"configured": configured, "user": user(request)}

    @app.get("/api/auth/github")
    def start(request: Request):
        if not configured:
            raise HTTPException(
                503, "GitHub sign-in has not been configured by the operator."
            )
        rate("oauth:" + request.client.host, 20, 900)
        state, binding, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        with db(root) as c:
            c.execute("DELETE FROM oauth_states WHERE expires<?", (time.time(),))
            c.execute(
                "INSERT INTO oauth_states VALUES(?,?,?,?,?)",
                (digest(state), digest(binding), verifier, time.time() + 600, callback),
            )
        challenge = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .decode()
            .rstrip("=")
        )
        response = RedirectResponse(
            "https://github.com/login/oauth/authorize?"
            + urlencode(
                {
                    "client_id": client_id,
                    "redirect_uri": callback,
                    "scope": "",
                    "state": state,
                    "code_challenge": challenge,
                    "code_challenge_method": "S256",
                }
            ),
            status_code=303,
        )
        response.set_cookie(
            "arena_oauth",
            binding,
            httponly=True,
            secure=secure,
            samesite="lax",
            max_age=600,
        )
        return response

    @app.get("/api/auth/github/callback")
    async def finish(request: Request, state: str = "", code: str = ""):
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            pending = c.execute(
                "SELECT * FROM oauth_states WHERE state_hash=?", (digest(state),)
            ).fetchone()
            if (
                not pending
                or pending["expires"] < time.time()
                or not secrets.compare_digest(
                    pending["binding_hash"],
                    digest(request.cookies.get("arena_oauth", "")),
                )
            ):
                raise HTTPException(400, "Sign-in expired or invalid. Start again.")
            c.execute("DELETE FROM oauth_states WHERE state_hash=?", (digest(state),))
        if not code or not configured:
            raise HTTPException(400, "GitHub sign-in was not completed. Start again.")
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                token_response = await client.post(
                    "https://github.com/login/oauth/access_token",
                    headers={"Accept": "application/json"},
                    data={
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "code": code,
                        "redirect_uri": callback,
                        "code_verifier": pending["verifier"],
                    },
                )
                token_response.raise_for_status()
                token = token_response.json()["access_token"]
                identity_response = await client.get(
                    "https://api.github.com/user",
                    headers={
                        "Authorization": "Bearer " + token,
                        "Accept": "application/vnd.github+json",
                    },
                )
                identity_response.raise_for_status()
                identity = identity_response.json()
                if type(identity["id"]) is not int or not isinstance(
                    identity["login"], str
                ):
                    raise ValueError("Invalid identity")
        except (httpx.HTTPError, KeyError, ValueError, TypeError):
            raise HTTPException(
                502, "GitHub sign-in failed. Please start again."
            ) from None
        session = secrets.token_urlsafe(32)
        with db(root) as c:
            c.execute(
                "DELETE FROM user_sessions WHERE expires<? OR token_hash=?",
                (time.time(), digest(request.cookies.get("arena_user", ""))),
            )
            c.execute(
                "INSERT INTO user_sessions VALUES(?,?,?,?)",
                (
                    digest(session),
                    str(identity["id"]),
                    identity["login"],
                    time.time() + 7 * 86400,
                ),
            )
        response = RedirectResponse("/#receipts", status_code=303)
        response.delete_cookie("arena_oauth")
        response.set_cookie(
            "arena_user",
            session,
            httponly=True,
            secure=secure,
            samesite="lax",
            max_age=7 * 86400,
        )
        return response

    @app.post("/api/auth/logout")
    def logout(request: Request):
        with db(root) as c:
            c.execute(
                "DELETE FROM user_sessions WHERE token_hash=?",
                (digest(request.cookies.get("arena_user", "")),),
            )
        response = JSONResponse({"ok": True})
        response.delete_cookie("arena_user")
        return response

    @app.get("/api/my/submissions")
    def mine(request: Request):
        identity = require_user(request)
        with db(root) as c:
            rows = c.execute(
                "SELECT * FROM submissions WHERE owner_id=? ORDER BY created DESC",
                (identity["github_id"],),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "state": r["state"],
                "created": r["created"],
                "manifest": unpack(r)["manifest"],
            }
            for r in rows
        ]

    return user, require_user
