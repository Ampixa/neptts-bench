import hashlib
import io
import json
import zipfile

import pytest

from submission_app.contract import BENCHMARK, SENTENCES
from submission_app.server import create_app
from submission_app.tests.test_flow import client, login, submit


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path, "testing-reviewer-password")


def payload(job):
    spec = {
        "benchmark": BENCHMARK,
        "model": "small",
        "development_only": False,
        "weights_sha256": "a" * 64,
        "code_sha256": "b" * 64,
        "packages": {"test": "fixture"},
    }
    spec["id"] = (
        "whisper-small-ne-"
        + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    )
    return {
        **job,
        "report": {
            "profile": spec,
            "per_sentence": {s: {"cer": 0.2, "wer": 0.3} for s in SENTENCES},
            "cer": 0.2,
            "wer": 0.3,
            "human_mos": None,
            "predicted_mos": None,
        },
    }


def test_download_import_approve(app):
    c = client(app)
    sid = submit(c).json()["id"]
    base = f"/api/admin/submissions/{sid}"
    assert c.get(base + "/package").status_code == 401
    assert c.post(base + "/results", json={}).status_code == 401
    login(c)
    response = c.get(base + "/package")
    assert response.status_code == 200
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        assert len(z.namelist()) == 207
        job = json.loads(z.read("evaluation-job.json"))
    body = payload(job)
    for field in ("submission_id", "benchmark", "audio_sha256"):
        bad = {**body, field: "wrong"}
        assert c.post(base + "/results", json=bad).status_code == 422
    bad = payload(job)
    bad["report"]["per_sentence"].pop(next(iter(SENTENCES)))
    assert c.post(base + "/results", json=bad).status_code == 422
    bad = payload(job)
    bad["report"]["cer"] = float("nan")
    assert c.post(base + "/results", content=json.dumps(bad)).status_code == 422
    assert (
        c.post(base + "/results", content=b"x" * (2 * 1024**2 + 1)).status_code == 413
    )
    bad = payload(job)
    bad["report"]["profile"]["development_only"] = True
    assert c.post(base + "/results", json=bad).status_code == 422
    bad = payload(job)
    bad["report"]["profile"]["code_sha256"] = "c" * 64
    assert c.post(base + "/results", json=bad).status_code == 422
    assert c.post(base + "/results", json=body).status_code == 200
    assert c.get("/api/leaderboard").json()["entries"] == []
    assert c.post(base + "/results", json=body).status_code == 409
    assert (
        c.post(
            base + "/decision",
            json={"action": "approve", "note": "Fixture review complete"},
        ).status_code
        == 200
    )
    assert len(c.get("/api/leaderboard").json()["entries"]) == 1
    assert c.post(base + "/results", json=body).status_code == 409
