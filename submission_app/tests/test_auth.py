import time
from urllib.parse import parse_qs, urlparse

import httpx
from fastapi.testclient import TestClient

from submission_app.auth import digest
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import archive


def test_oauth_owner_recovery_and_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("NEPTTS_GITHUB_CLIENT_ID", "test-client")
    monkeypatch.setenv("NEPTTS_GITHUB_CLIENT_SECRET", "test-secret")
    app = create_app(tmp_path, "test-admin-password")
    c = TestClient(app, follow_redirects=False)
    c.headers["X-Neptts-Request"] = "submission"
    assert c.post("/api/submissions", content=b"bad").status_code == 401
    start = c.get("/api/auth/github")
    query = parse_qs(urlparse(start.headers["location"]).query)
    assert query["code_challenge_method"] == ["S256"]
    assert "repo" not in query.get("scope", [])
    state = query["state"][0]
    assert (
        TestClient(app)
        .get("/api/auth/github/callback", params={"state": state, "code": "code"})
        .status_code
        == 400
    )
    original_client = httpx.AsyncClient

    def provider(request):
        if request.url.path.endswith("access_token"):
            assert b"code_verifier=" in request.content
            return httpx.Response(200, json={"access_token": "provider-secret"})
        return httpx.Response(200, json={"id": 777, "login": "contributor"})

    monkeypatch.setattr(
        "submission_app.auth.httpx.AsyncClient",
        lambda **kw: original_client(transport=httpx.MockTransport(provider), **kw),
    )
    result = c.get("/api/auth/github/callback", params={"state": state, "code": "code"})
    assert result.status_code == 303
    assert "HttpOnly" in result.headers["set-cookie"]
    assert (
        c.get(
            "/api/auth/github/callback", params={"state": state, "code": "code"}
        ).status_code
        == 400
    )
    assert c.get("/api/auth/me").json()["user"]["github_id"] == "777"
    receipt = c.post(
        "/api/submissions",
        content=archive(),
        headers={"Content-Type": "application/zip"},
    ).json()
    sid = receipt["id"]
    assert c.get("/api/my/submissions").json()[0]["id"] == sid
    # Session and ownership survive web process restart.
    restarted = TestClient(create_app(tmp_path, "test-admin-password"))
    restarted.cookies.update(c.cookies)
    assert restarted.get(f"/api/submissions/{sid}").status_code == 200
    with db(tmp_path) as conn:
        conn.execute(
            "INSERT INTO user_sessions VALUES(?,?,?,?)",
            (digest("other"), "888", "other", time.time() + 60),
        )
    other = TestClient(app)
    other.cookies.set("arena_user", "other")
    assert other.get("/api/my/submissions").json() == []
    assert other.get(f"/api/submissions/{sid}").status_code == 404
    assert other.get(f"/api/submissions/{sid}/audio/sent_001").status_code == 404
    assert c.post("/api/auth/logout").status_code == 200
    assert c.get("/api/my/submissions").status_code == 401
    assert (
        c.get(
            f"/api/submissions/{sid}", headers={"X-Submission-Token": receipt["token"]}
        ).status_code
        == 200
    )
    with db(tmp_path) as conn:
        assert "provider-secret" not in str(
            conn.execute("SELECT * FROM user_sessions").fetchall()
        )


def test_unconfigured_and_expired_oauth(tmp_path, monkeypatch):
    monkeypatch.delenv("NEPTTS_GITHUB_CLIENT_ID", raising=False)
    monkeypatch.delenv("NEPTTS_GITHUB_CLIENT_SECRET", raising=False)
    c = TestClient(create_app(tmp_path, "test-admin-password"))
    assert c.get("/api/auth/me").json() == {"configured": False, "user": None}
    assert c.get("/api/auth/github").status_code == 503
    with db(tmp_path) as conn:
        conn.execute(
            "INSERT INTO oauth_states VALUES(?,?,?,?,?)",
            (digest("state"), digest("cookie"), "verifier", time.time() - 1, "unused"),
        )
    c.cookies.set("arena_oauth", "cookie")
    assert c.get("/api/auth/github/callback?state=state&code=code").status_code == 400
