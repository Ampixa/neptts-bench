import io
import json
import stat
import time
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from submission_app.auth import digest
from submission_app.contract import BENCHMARK, SENTENCES, template
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.worker import process_one


def archive(*, omit=None, extra=None, silent=False, name="Test system", stereo=False):
    audio = io.BytesIO()
    samples = np.zeros(4000) if silent else np.sin(np.arange(4000) * 0.06) * 0.2
    if stereo:
        samples = np.column_stack([samples, samples])
    sf.write(audio, samples, 16000, format="WAV", subtype="PCM_16")
    result = io.BytesIO()
    manifest = template()
    manifest.update(system_name=name, permission_to_evaluate_and_publish=True)
    with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("submission.json", json.dumps(manifest))
        for sid in SENTENCES:
            if sid != omit:
                z.writestr(f"audio/{sid}.wav", audio.getvalue())
        if extra:
            z.writestr(*extra)
    return result.getvalue()


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path, "testing-reviewer-password")


def client(app):
    c = TestClient(app)
    c.headers["X-Neptts-Request"] = "submission"
    with db(app.state.root) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO user_sessions VALUES(?,?,?,?)",
            (digest("test-user-session"), "123", "tester", time.time() + 3600),
        )
    c.cookies.set("arena_user", "test-user-session")
    return c


def submit(c, **kwargs):
    return c.post(
        "/api/submissions",
        content=archive(**kwargs),
        headers={"Content-Type": "application/zip"},
    )


def login(c):
    assert (
        c.post(
            "/api/admin/login", json={"password": "testing-reviewer-password"}
        ).status_code
        == 200
    )


def decision(c, sid, action):
    return c.post(
        f"/api/admin/submissions/{sid}/decision",
        json={"action": action, "note": "Fixture reviewed in automated test."},
    )


def fake_scorer(development=False, incomplete=False, fail=False):
    spec = {"id": "unit-test-evaluator", "development_only": development}

    def run(files, model, progress):
        if fail:
            raise RuntimeError("Secret diagnostic must not reach public API")
        scores = {s: {"cer": 0.2, "wer": 0.3} for s in SENTENCES}
        if incomplete:
            scores.pop(next(iter(scores)))
        progress(len(scores))
        return {
            "profile": spec,
            "per_sentence": scores,
            "cer": 0.2,
            "wer": 0.3,
            "human_mos": None,
            "predicted_mos": None,
        }

    return spec, run


def test_upload_evaluate_approve_and_public_report(app):
    c = client(app)
    receipt = submit(c).json()
    sid, token = receipt["id"], receipt["token"]
    assert receipt["validation"]["files"] == len(SENTENCES) == 205
    c.post("/api/auth/logout")
    assert c.get(f"/api/submissions/{sid}").status_code == 404
    assert (
        c.get(
            f"/api/submissions/{sid}", headers={"X-Submission-Token": token}
        ).status_code
        == 200
    )
    assert c.get("/api/leaderboard").json()["entries"] == []
    assert c.get(f"/api/leaderboard/{sid}/report").status_code == 404
    assert decision(c, sid, "approve").status_code == 401
    login(c)
    assert decision(c, sid, "approve").status_code == 409
    assert decision(c, sid, "queue").status_code == 200
    assert decision(c, sid, "queue").status_code == 409
    spec, scorer = fake_scorer()
    assert process_one(app.state.root, scorer=scorer, spec=spec)
    assert c.get("/api/leaderboard").json()["entries"] == []
    assert decision(c, sid, "approve").json()["state"] == "published"
    data = c.get("/api/leaderboard").json()["entries"]
    assert len(data) == 1
    assert data[0]["report"]["human_mos"] is None
    assert "token_hash" not in json.dumps(data)
    assert c.get(f"/api/leaderboard/{sid}/report").status_code == 200
    assert decision(c, sid, "approve").status_code == 409
    c.post("/api/admin/logout")
    assert c.get(f"/api/submissions/{sid}/audio/sent_001").status_code == 404


@pytest.mark.parametrize(
    "kwargs",
    [
        {"omit": "sent_001"},
        {"silent": True},
        {"stereo": True},
        {"extra": ("../escape.txt", b"bad")},
        {"extra": ("audio/sent_999.wav", b"bad")},
        {"extra": ("audio/sent_001.wav", b"duplicate")},
    ],
)
def test_invalid_upload_is_rejected_and_cleaned(app, kwargs):
    c = client(app)
    assert submit(c, **kwargs).status_code == 422
    assert list((app.state.root / "uploads").iterdir()) == []
    assert list(app.state.root.glob("incoming-*")) == []


def test_symlink_and_wrong_benchmark(app):
    c = client(app)
    z = io.BytesIO(archive())
    with zipfile.ZipFile(z, "a") as f:
        item = zipfile.ZipInfo("audio/link.wav")
        item.external_attr = (stat.S_IFLNK | 0o777) << 16
        f.writestr(item, "../../secret")
    assert (
        c.post(
            "/api/submissions",
            content=z.getvalue(),
            headers={"Content-Type": "application/zip"},
        ).status_code
        == 422
    )
    from submission_app.contract import Manifest

    bad = template()
    bad.update(benchmark="stale", permission_to_evaluate_and_publish=True)
    with pytest.raises(ValueError):
        Manifest.model_validate(bad)


def test_duplicate_upload_and_csrf(app):
    c = client(app)
    assert submit(c).status_code == 201
    assert submit(c).status_code == 409
    login(c)
    assert (
        c.post(
            "/api/admin/logout", headers={"Origin": "https://evil.example"}
        ).status_code
        == 403
    )
    c.headers.pop("X-Neptts-Request")
    assert c.post("/api/admin/logout").status_code == 403


@pytest.mark.parametrize("mode", ["incomplete", "fail", "development"])
def test_bad_or_development_evaluation_cannot_publish(app, mode):
    c = client(app)
    sid = submit(c).json()["id"]
    login(c)
    decision(c, sid, "queue")
    spec, scorer = fake_scorer(**{mode: True})
    process_one(app.state.root, scorer=scorer, spec=spec)
    assert decision(c, sid, "approve").status_code == 409
    state = c.get(f"/api/submissions/{sid}").json()
    assert "Secret diagnostic" not in json.dumps(state)
    assert c.get("/api/leaderboard").json()["entries"] == []


def test_audio_integrity_and_reject(app):
    c = client(app)
    sid = submit(c).json()["id"]
    login(c)
    decision(c, sid, "queue")
    (app.state.root / "uploads" / sid / "sent_001.wav").write_bytes(b"changed")
    spec, scorer = fake_scorer()
    process_one(app.state.root, scorer=scorer, spec=spec)
    assert c.get(f"/api/submissions/{sid}").json()["state"] == "evaluation_failed"
    assert decision(c, sid, "reject").status_code == 200
    assert decision(c, sid, "queue").status_code == 409


def test_pack_ids_and_no_credentials(app):
    c = client(app)
    z = zipfile.ZipFile(io.BytesIO(c.get("/api/benchmark/pack").content))
    prompts = json.loads(z.read("sentences.json"))
    assert {s["sent_id"] for s in prompts} == set(SENTENCES)
    assert json.loads(z.read("submission.json"))["benchmark"] == BENCHMARK
    assert c.get("/api/admin/submissions").status_code == 401
    assert c.post("/api/admin/login", json={"password": "wrong"}).status_code == 401
