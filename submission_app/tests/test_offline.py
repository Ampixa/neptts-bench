import hashlib
import io
import json
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from submission_app import offline
from submission_app.contract import BENCHMARK, SENTENCES
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import archive, client, login, submit

GENERATION_FIELDS = (
    "generation_mode",
    "candidate_count",
    "selector_model",
    "judge_family_used",
    "judge_family_details",
)
LEGACY_MAIN = "210dd973ca5ea80149fcccbe68c73ca8f203c624"


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


@pytest.mark.parametrize(
    "missing", [GENERATION_FIELDS, *[(f,) for f in GENERATION_FIELDS]]
)
def test_legacy_submission_export_offline_import(app, tmp_path, monkeypatch, missing):
    c = client(app)
    sid = submit(c).json()["id"]
    # Reproduce persisted main-era metadata, including misleading old free text.
    with db(app.state.root) as conn:
        row = conn.execute(
            "SELECT manifest FROM submissions WHERE id=?", (sid,)
        ).fetchone()
        declaration = json.loads(row["manifest"])
        for field in missing:
            declaration.pop(field)
        declaration["selection_policy"] = (
            "One generation per sentence; no cherry-picking"
        )
        conn.execute(
            "UPDATE submissions SET manifest=? WHERE id=?",
            (json.dumps(declaration), sid),
        )
    check_export_offline_import(c, sid, declaration, tmp_path, monkeypatch)


def test_main_created_submission_export_offline_import(tmp_path, monkeypatch):
    """Exercise an actual old-server upload, then reopen its database with the new server."""
    repo = Path(__file__).resolve().parents[2]
    snapshot = subprocess.run(
        [
            "git",
            "archive",
            LEGACY_MAIN,
            "submission_app",
            "benchmark/sentences.json",
            "neptts-eval/src",
        ],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    source = tmp_path / "legacy-main"
    source.mkdir()
    with tarfile.open(fileobj=io.BytesIO(snapshot)) as bundle:
        bundle.extractall(source, filter="data")
    root = tmp_path / "state"
    creator = """
import json
import sys
from pathlib import Path
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import client, submit
app = create_app(Path(sys.argv[1]), "testing-reviewer-password")
response = submit(client(app))
assert response.status_code == 201, response.text
sid = response.json()["id"]
with db(app.state.root) as conn:
    row = conn.execute("SELECT manifest FROM submissions WHERE id=?", (sid,)).fetchone()
print(json.dumps({"id": sid, "manifest": json.loads(row["manifest"])}))
"""
    result = subprocess.run(
        [sys.executable, "-c", creator, str(root)],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    )
    old = json.loads(result.stdout)
    assert not set(GENERATION_FIELDS) & old["manifest"].keys()
    app = create_app(root, "testing-reviewer-password")
    check_export_offline_import(
        client(app), old["id"], old["manifest"], tmp_path, monkeypatch
    )


def check_export_offline_import(c, sid, declaration, tmp_path, monkeypatch):
    login(c)
    base = f"/api/admin/submissions/{sid}"
    response = c.get(base + "/package")
    assert response.status_code == 200
    package = tmp_path / "evaluation.zip"
    package.write_bytes(response.content)
    with zipfile.ZipFile(package) as bundle:
        assert json.loads(bundle.read("submission.json")) == declaration
        job = json.loads(bundle.read("evaluation-job.json"))
    output = tmp_path / "results.json"
    evaluated = []

    def scorer(audio_dir, model_name, progress):
        assert {p.stem for p in audio_dir.glob("*.wav")} == set(SENTENCES)
        assert model_name == "small"
        evaluated.append(True)
        return payload(job)["report"]

    monkeypatch.setattr(offline, "evaluate", scorer)
    monkeypatch.setattr(
        sys, "argv", ["offline", "--package", str(package), "--output", str(output)]
    )
    offline.main()
    assert evaluated == [True]
    result = c.post(base + "/results", content=output.read_bytes())
    assert result.status_code == 200
    assert result.json()["state"] == "awaiting_review"
    assert (
        c.post(
            base + "/decision",
            json={"action": "approve", "note": "Legacy fixture reviewed"},
        ).status_code
        == 200
    )
    assert c.get("/api/leaderboard").json()["entries"] == []
    assert c.get("/api/leaderboard?selection=best_of_n").json()["entries"] == []
    entries = c.get("/api/leaderboard?selection=unknown").json()["entries"]
    assert [entry["id"] for entry in entries] == [sid]
    assert entries[0]["manifest"] == declaration


@pytest.mark.parametrize(
    "missing", [GENERATION_FIELDS, *[(f,) for f in GENERATION_FIELDS]]
)
def test_new_upload_rejects_missing_generation_fields(app, missing):
    c = client(app)
    source = archive()
    result = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(source)) as original:
        with zipfile.ZipFile(result, "w", zipfile.ZIP_DEFLATED) as target:
            for member in original.infolist():
                data = original.read(member)
                if member.filename == "submission.json":
                    declaration = json.loads(data)
                    for field in missing:
                        declaration.pop(field)
                    data = json.dumps(declaration).encode()
                target.writestr(member, data)
    response = c.post(
        "/api/submissions",
        content=result.getvalue(),
        headers={"Content-Type": "application/zip"},
    )
    assert response.status_code == 422
    assert "Field required" in response.json()["detail"]
    with db(app.state.root) as conn:
        assert conn.execute("SELECT COUNT(*) FROM submissions").fetchone()[0] == 0
