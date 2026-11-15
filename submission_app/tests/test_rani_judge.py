"""Rani runs over every approved submission as a separately approved judge."""

import copy
import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from submission_app.contract import BENCHMARK, SENTENCES, template
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import client, decision, fake_scorer, login
from submission_app.worker import process_additional, process_one


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path, "testing-reviewer-password")


@pytest.fixture
def rani(monkeypatch):
    # Imported after submission_app, which puts the evaluator toolkit on sys.path.
    from neptts_eval import rani

    pins = {
        **rani.PINS,
        "checkpoint_sha256": "b" * 64,
        "packages": {n: "0.0.0" for n in rani.RUNTIME_PACKAGES},
    }
    monkeypatch.setattr(rani, "PINS", copy.deepcopy(pins))
    return rani


def rani_scorer(spec, fail=False, calls=None):
    def run(files, progress):
        if calls is not None:
            calls.append(files.name)
        if fail:
            raise RuntimeError("decoder crashed")
        rows = {s: {"cer": 0.1, "wer": 0.2, "char_errors": 1, "chars": 10,
                    "word_errors": 1, "words": 5} for s in SENTENCES}
        return {"profile": spec, "per_sentence": rows, "cer": 0.1, "wer": 0.2,
                "corpus_cer": 0.1, "corpus_wer": 0.2, "human_mos": None, "predicted_mos": None}
    return run


def submit(c, i):
    """Distinct audio per system: audio hashes are unique across submissions."""
    wav = io.BytesIO()
    sf.write(wav, np.sin(np.arange(4000) * (0.05 + 0.01 * i)) * 0.2, 16000, format="WAV", subtype="PCM_16")
    manifest = template()
    manifest.update(system_name=f"System {i}", permission_to_evaluate_and_publish=True)
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as z:
        z.writestr("submission.json", json.dumps(manifest))
        for sid in SENTENCES:
            z.writestr(f"audio/{sid}.wav", wav.getvalue())
    response = c.post("/api/submissions", content=body.getvalue(),
                      headers={"Content-Type": "application/zip"})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def published(arena, c, n=1):
    login(c)
    ids = []
    for i in range(n):
        sid = submit(c, i)
        assert decision(c, sid, "queue").status_code == 200
        spec, scorer = fake_scorer()
        assert process_one(arena.state.root, scorer=scorer, spec=spec)
        assert decision(c, sid, "approve").json()["state"] == "published"
        ids.append(sid)
    return ids


def rows(root):
    with db(root) as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM profile_evaluations")]


def test_backfills_every_approved_submission_once(app, rani):
    c = client(app)
    ids = published(app, c, 2)
    pending = submit(c, 99)
    spec = rani.profile(BENCHMARK)
    legacy = c.get("/api/leaderboard?selection=all").json()
    calls = []
    scorer = rani_scorer(spec, calls=calls)
    while process_additional(app.state.root, spec, scorer, "Rani"):
        pass
    assert sorted(calls) == sorted(ids) and pending not in calls
    assert {r["submission_id"] for r in rows(app.state.root)} == set(ids)
    assert all(r["state"] == "awaiting_review" for r in rows(app.state.root))
    assert not process_additional(app.state.root, spec, scorer, "Rani")  # idempotent
    # Nothing public until a reviewer approves the Rani profile separately.
    pid = spec["id"]
    assert c.get(f"/api/leaderboard?selection=all&profile={pid}").json()["entries"] == []
    assert c.get("/api/leaderboard?selection=all").json() == legacy
    base = f"/api/admin/submissions/{ids[0]}/profiles/{pid}/decision"
    assert c.post(base, json={"action": "approve", "note": "Rani reviewed"}).status_code == 200
    entries = c.get(f"/api/leaderboard?selection=all&profile={pid}").json()["entries"]
    assert [e["id"] for e in entries] == [ids[0]]


def test_tampered_audio_is_not_scored(app, rani):
    c = client(app)
    (sid,) = published(app, c)
    (app.state.root / "uploads" / sid / "sent_001.wav").write_bytes(b"swapped")
    spec = rani.profile(BENCHMARK)
    calls, skip = [], set()
    assert process_additional(app.state.root, spec, rani_scorer(spec, calls=calls), "Rani", skip)
    assert calls == [] and skip == {sid} and rows(app.state.root) == []
    assert not process_additional(app.state.root, spec, rani_scorer(spec), "Rani", skip)


@pytest.mark.parametrize("bad", ["crash", "profile", "count"])
def test_failed_or_invalid_reports_record_nothing(app, rani, bad):
    c = client(app)
    published(app, c)
    spec = rani.profile(BENCHMARK)
    base = rani_scorer(spec, fail=bad == "crash")

    def scorer(files, progress):
        report = base(files, progress)
        if bad == "profile":
            report["profile"] = {**spec, "threads": 8}
        if bad == "count":
            report["per_sentence"][next(iter(SENTENCES))]["char_errors"] = 7
        return report

    assert process_additional(app.state.root, spec, scorer, "Rani", set())
    assert rows(app.state.root) == []


def test_offline_import_accepts_pinned_rani_and_rejects_drift(app, rani):
    c = client(app)
    (sid,) = published(app, c)
    with db(app.state.root) as conn:
        audio = conn.execute("SELECT audio_sha256 FROM submissions WHERE id=?", (sid,)).fetchone()[0]
    spec = rani.profile(BENCHMARK)
    report = rani_scorer(spec)(app.state.root / "uploads" / sid, None)
    body = {"submission_id": sid, "benchmark": BENCHMARK, "audio_sha256": audio, "report": report}
    url = f"/api/admin/submissions/{sid}/profiles/{spec['id']}/results"
    drift = json.loads(json.dumps(body))
    drift["report"]["profile"]["decoding"] = "beam search with LM"
    assert c.post(url, json=drift).status_code == 422
    assert c.post(url, json=body).status_code == 200
    assert c.post(url, json=body).status_code == 409
    assert rows(app.state.root)[0]["state"] == "awaiting_review"


def test_catalog_lists_rani_only_when_pinned(app, rani, monkeypatch):
    c = client(app)
    listed = c.get("/api/judge-profiles").json()["profiles"]
    entry = next(p for p in listed if p["stage"] == "rani")
    assert entry["id"] == rani.profile(BENCHMARK)["id"]
    assert "Rani (Ampixa Labs)" in entry["availability"]
    assert entry["provisional"] is True
    monkeypatch.setattr(rani, "PINS", {**rani.PINS, "checkpoint_sha256": None})
    assert all(p["stage"] != "rani" for p in c.get("/api/judge-profiles").json()["profiles"])
