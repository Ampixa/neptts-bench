"""Automatic admin: queueing, publish-or-hold decisions, retries and takedown."""

import io
import json
import zipfile

import numpy as np
import pytest
import soundfile as sf

from submission_app import autoreview
from submission_app.contract import SENTENCES, template
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import client, decision, login
from submission_app.worker import process_additional, process_one, recover_interrupted

SPEC = {"id": "primary-judge-test", "development_only": False}


@pytest.fixture(autouse=True)
def automatic(monkeypatch):
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "on")


@pytest.fixture
def app(tmp_path):
    return create_app(tmp_path, "testing-reviewer-password")


def tone(i, j=0):
    data = io.BytesIO()
    sf.write(data, np.sin(np.arange(4000) * (0.03 + 0.004 * i + 0.0001 * j)) * 0.2, 16000,
             format="WAV", subtype="PCM_16")
    return data.getvalue()


def manifest(i, **change):
    m = template()
    m.update(system_name=f"Kathmandu TTS {i}", model_version="v1.0", organization="Lab",
             voice="female-1", generation_settings="seed 1, temperature 0.6",
             training_overlap="none", permission_to_evaluate_and_publish=True)
    m.update(change)
    return m


def upload(c, i, audio=None, **change):
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as z:
        z.writestr("submission.json", json.dumps(manifest(i, **change)))
        for j, sid in enumerate(SENTENCES):
            z.writestr(f"audio/{sid}.wav", (audio or {}).get(sid) or tone(i, j))
    r = c.post("/api/submissions", content=body.getvalue(),
               headers={"Content-Type": "application/zip"})
    assert r.status_code == 201, r.text
    return r.json()


def scorer(cer=0.1, failed=0, crash=False):
    def run(files, model, progress):
        if crash:
            raise RuntimeError("decoder crashed")
        rows = {s: {"cer": 1.0 if k < failed else cer, "wer": 0.2}
                for k, s in enumerate(SENTENCES)}
        return {"profile": SPEC, "per_sentence": rows,
                "cer": sum(r["cer"] for r in rows.values()) / len(rows),
                "wer": 0.2, "human_mos": None, "predicted_mos": None}
    return run


def row(app, sid):
    with db(app.state.root) as c:
        return dict(c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone())


def evaluate(app, run=None):
    assert process_one(app.state.root, scorer=run or scorer(), spec=SPEC)


def reasons(app, sid):
    return json.loads(row(app, sid)["auto_review"])["reasons"]


def test_clean_submission_is_queued_scored_and_published_without_a_person(app):
    c = client(app)
    sub = upload(c, 0)
    assert sub["state"] == "queued"
    evaluate(app)
    r = row(app, sub["id"])
    assert r["state"] == "published" and reasons(app, sub["id"]) == []
    assert autoreview.VERSION in r["reviewer_note"]
    board = c.get("/api/leaderboard?selection=all").json()
    assert [e["id"] for e in board["entries"]] == [sub["id"]]
    status = c.get(f"/api/submissions/{sub['id']}", headers={"X-Submission-Token": sub["token"]}).json()
    assert [e["state"] for e in status["events"]] == [
        "validated", "queued", "evaluating", "awaiting_review", "published"]


@pytest.mark.parametrize("change,audio,run,expected", [
    ({"system_name": "Your TTS"}, None, None, ["placeholder"]),
    ({"judge_family_used": True, "judge_family_details": "Rani reranked 8 takes"}, None, None,
     ["judge_family"]),
    ({}, None, scorer(failed=60), ["failed_prompts"]),
])
def test_doubtful_submissions_are_held_with_reasons(app, change, audio, run, expected):
    c = client(app)
    sub = upload(c, 0, audio, **change)
    evaluate(app, run)
    r = row(app, sub["id"])
    assert r["state"] == "awaiting_review" and reasons(app, sub["id"]) == expected
    assert r["message"].startswith("Held for a reviewer:")
    assert c.get("/api/leaderboard?selection=all").json()["entries"] == []
    login(c)  # a person can still approve it
    assert decision(c, sub["id"], "approve").json()["state"] == "published"


def test_reused_audio_and_reused_name_are_held(app):
    c = client(app)
    first = upload(c, 0)
    evaluate(app)
    copied = {sid: tone(0, j) for j, sid in enumerate(list(SENTENCES)[:20])}
    second = upload(c, 1, copied, system_name="Kathmandu TTS 0")
    evaluate(app)
    assert row(app, first["id"])["state"] == "published"
    assert reasons(app, second["id"]) == ["duplicate_name", "shared_audio"]


def test_failed_runs_retry_once_then_wait_for_a_person(app):
    c = client(app)
    sub = upload(c, 0)
    evaluate(app, scorer(crash=True))
    assert row(app, sub["id"])["state"] == "queued"
    evaluate(app, scorer(crash=True))
    r = row(app, sub["id"])
    assert r["state"] == "evaluation_failed" and r["attempts"] == autoreview.MAX_ATTEMPTS
    assert not process_one(app.state.root, scorer=scorer(), spec=SPEC)


def test_interrupted_run_is_requeued(app):
    c = client(app)
    sub = upload(c, 0)
    with db(app.state.root) as conn:
        conn.execute("UPDATE submissions SET state='evaluating',attempts=1 WHERE id=?", (sub["id"],))
    recover_interrupted(app.state.root)
    assert row(app, sub["id"])["state"] == "queued"
    evaluate(app)
    assert row(app, sub["id"])["state"] == "published"


def test_reviewer_can_take_down_an_automatic_approval(app):
    c = client(app)
    sub = upload(c, 0)
    evaluate(app)
    login(c)
    r = c.post(f"/api/admin/submissions/{sub['id']}/decision",
               json={"action": "reject", "note": "Audio is a human recording."})
    assert r.json()["state"] == "rejected"
    assert c.get("/api/leaderboard?selection=all").json()["entries"] == []
    assert "Removed from the leaderboard" in row(app, sub["id"])["message"]


def test_other_judges_and_missing_primary_never_publish(app, monkeypatch):
    c = client(app)
    sub = upload(c, 0)
    other = {"id": "some-other-judge", "development_only": False}
    run = scorer()
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "off")  # score now, decide below
    assert process_one(app.state.root, scorer=lambda f, m, p: {**run(f, m, p), "profile": other},
                       spec=other)
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "on")
    assert autoreview.review(app.state.root, sub["id"], SPEC["id"]) == "awaiting_review"
    assert reasons(app, sub["id"]) == ["not_primary_judge"]
    assert autoreview.review(app.state.root, sub["id"], None) == "awaiting_review"
    assert reasons(app, sub["id"]) == ["no_primary_judge"]


def test_second_opinion_judge_is_published_automatically(app):
    c = client(app)
    sub = upload(c, 0)
    evaluate(app)
    second = {"id": "whisper-second-opinion", "development_only": False}
    run = scorer()
    assert process_additional(app.state.root, second,
                              lambda f, p: {**run(f, None, p), "profile": second},
                              "Whisper", set(), check=lambda r: None)
    board = c.get("/api/leaderboard?selection=all&profile=whisper-second-opinion").json()
    assert [e["id"] for e in board["entries"]] == [sub["id"]]


def test_switching_automation_off_restores_manual_review(app, monkeypatch):
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "off")
    c = client(app)
    sub = upload(c, 0)
    assert sub["state"] == "validated"
    assert not process_one(app.state.root, scorer=scorer(), spec=SPEC)


def test_catalog_names_the_primary_judge(app, monkeypatch):
    from submission_app import server

    monkeypatch.setattr(server, "primary_profile_id", lambda: "rani-test-id")
    assert client(app).get("/api/judge-profiles").json()["primary"] == "rani-test-id"


def test_rani_worker_is_the_primary_judge_end_to_end(app, monkeypatch, tmp_path):
    import copy
    import sys

    from neptts_eval import rani

    from submission_app import worker
    from submission_app.contract import BENCHMARK
    from submission_app.profiles import primary_profile_id

    monkeypatch.setattr(rani, "PINS", copy.deepcopy({
        **rani.PINS, "checkpoint_sha256": "b" * 64,
        "packages": {n: "0.0.0" for n in rani.RUNTIME_PACKAGES}}))
    spec = rani.profile(BENCHMARK)
    assert primary_profile_id() == spec["id"]

    def fake_evaluate(audio_dir, sentences, benchmark, config, progress, decoder=None):
        assert decoder == "decoder"
        rows = {s: {"cer": 0.1, "wer": 0.2, "char_errors": 1, "chars": 10,
                    "word_errors": 1, "words": 5} for s in sentences}
        return {"profile": spec, "per_sentence": rows, "cer": 0.1, "wer": 0.2,
                "corpus_cer": 0.1, "corpus_wer": 0.2, "human_mos": None, "predicted_mos": None}

    monkeypatch.setattr(rani, "build_decoder", lambda config: "decoder")
    monkeypatch.setattr(rani, "evaluate", fake_evaluate)
    sub = upload(client(app), 0)
    monkeypatch.setattr(sys, "argv", ["worker", "--data", str(app.state.root), "--engine", "rani",
                                      "--checkpoint", str(tmp_path / "rani.nemo"), "--once"])
    worker.main()
    r = row(app, sub["id"])
    assert r["state"] == "published" and r["profile"] == spec["id"]
    board = client(app).get("/api/judge-profiles").json()
    assert board["primary"] == spec["id"]
