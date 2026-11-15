"""Voice donation: consent gate, signal checks, ASR states, privacy and payouts."""

import io
import json

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from submission_app import donations
from submission_app.donations import ANCHOR, CONSENT_PATH, load_prompt_sets
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import login
from submission_app.worker import process_donation

PASSWORD = "testing-reviewer-password"


def take(rate=48000, seconds=1.6, gain=0.1, noise=1e-4, channels=1, clip=False):
    """Speech-like: quiet lead-in, a voiced burst, quiet tail."""
    n = int(rate * seconds)
    t = np.arange(n) / rate
    voiced = (t > 0.3) & (t < seconds - 0.3)
    x = np.where(voiced, np.sin(2 * np.pi * 220 * t) * gain, 0.0)
    x = x + np.random.default_rng(0).normal(0, noise, n)
    if clip:
        x = np.clip(x * 20, -1, 1)
    if channels == 2:
        x = np.column_stack([x, x])
    data = io.BytesIO()
    sf.write(data, x, rate, format="WAV", subtype="PCM_16")
    return data.getvalue()


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path)


def make_app(tmp_path):
    """App with an approved consent copy and the paid programme open."""
    app = create_app(tmp_path / "data", PASSWORD)
    doc = json.loads(CONSENT_PATH.read_text())
    doc["status"] = "approved"
    approved = tmp_path / "consent.json"
    approved.write_text(json.dumps(doc))
    app.state.consent_path = approved
    app.state.donor_rate = 5.0
    return app


def browser(app):
    c = TestClient(app)
    c.headers["X-Neptts-Request"] = "submission"
    return c


def enroll(c, **change):
    info = c.get("/api/donate/info").json()
    body = {"consent_version": info["consent"]["version"], "consent_sha256": info["consent"]["sha256"],
            "adult": True, "own_voice": True, "purposes": ["benchmark_reference", "asr_evaluation"],
            "age_band": "25-34", "gender": "prefer_not", "province": "bagmati",
            "first_language": "nepali", "device": "phone", **change}
    return c.post("/api/donate/enroll", json=body)


def record(c, data=None):
    nxt = c.get("/api/donate/next").json()
    r = c.post(f"/api/donate/recordings/{nxt['prompt_set']}/{nxt['prompt_id']}",
               content=data or take(), headers={"Content-Type": "audio/wav"})
    return nxt, r


def wavs(app):
    return sorted(p.name for p in (app.state.root / "donations").rglob("*.wav"))


def test_enrollment_closed_while_consent_is_draft(tmp_path):
    app = create_app(tmp_path, PASSWORD)
    c = browser(app)
    info = c.get("/api/donate/info").json()
    assert info["open"] is False and info["consent"]["status"] == "draft"
    assert enroll(c).status_code == 403


def test_enrollment_requires_current_consent_and_required_purposes(app):
    c = browser(app)
    assert enroll(c, adult=False).status_code == 422
    assert enroll(c, purposes=["asr_evaluation"]).status_code == 422
    assert enroll(c, purposes=["benchmark_reference", "made_up"]).status_code == 422
    assert enroll(c, consent_sha256="0" * 64).status_code == 409
    r = enroll(c)
    assert r.status_code == 200 and len(r.json()["recovery_code"]) == 27
    me = c.get("/api/donate/me").json()
    assert me["purposes"] == ["asr_evaluation", "benchmark_reference"]
    assert me["profile"]["province"] == "bagmati" and me["paid_status"] == "volunteer"


def test_recovery_code_restores_session_elsewhere(app):
    c = browser(app)
    code = enroll(c).json()["recovery_code"]
    other = browser(app)
    assert other.get("/api/donate/me").status_code == 401
    assert other.post("/api/donate/recover", json={"code": "wrong-code-12"}).status_code == 404
    donor_id = c.get("/api/donate/me").json()["id"]
    assert other.post("/api/donate/recover", json={"code": code.upper()}).status_code == 200
    assert other.get("/api/donate/me").json()["id"] == donor_id
    # One active session per donor: recovering signs out the old browser, so a
    # donor whose code leaked can take their account back the same way.
    assert c.get("/api/donate/me").status_code == 401


@pytest.mark.parametrize("bad,problem", [
    ({"gain": 0.0, "noise": 0.0}, "silent"),
    ({"clip": True}, "clipped"),
    ({"noise": 0.05}, "noisy"),
    ({"gain": 0.001, "noise": 1e-6}, "too_quiet"),
    ({"seconds": 0.5}, "too_short"),
    ({"seconds": 21}, "too_long"),
    ({"channels": 2}, "format"),
    ({"rate": 8000}, "rate"),
])
def test_signal_checks_reject_before_storing(app, bad, problem):
    c = browser(app)
    enroll(c)
    assert c.post("/api/donate/mic-check", content=take(**bad)).json()["problems"][0] == problem
    _, r = record(c, take(**bad))
    assert r.status_code == 422 and problem in r.json()["problems"]
    assert wavs(app) == [] and c.get("/api/donate/me").json()["recordings"] == []


def test_fluent_reading_with_one_short_pause_is_not_called_noisy():
    from submission_app.donation_audio import measure

    rate = 48000
    t = np.arange(rate * 12) / rate
    x = np.sin(2 * np.pi * 220 * t) * 0.1 * ((t < 6) | (t > 6.15))  # 150 ms pause in 12 s
    x = x + np.random.default_rng(1).normal(0, 1e-4, len(t))
    data = io.BytesIO()
    sf.write(data, x, rate, format="WAV", subtype="PCM_16")
    metrics, problems = measure(data.getvalue())
    assert problems == [] and metrics["snr_db"] > 40


def test_good_take_is_stored_privately_and_prompts_advance(app):
    c = browser(app)
    enroll(c)
    mic = c.post("/api/donate/mic-check", content=take()).json()
    assert mic["passed"] and mic["metrics"]["snr_db"] > 40
    first, r = record(c)
    assert r.status_code == 200 and r.json()["state"] == "asr_pending"
    assert first["prompt_set"] == ANCHOR and first["remaining"] == 205
    second = c.get("/api/donate/next").json()
    assert second["prompt_id"] != first["prompt_id"] and second["remaining"] == 204
    rec = c.get("/api/donate/me").json()["recordings"][0]
    assert c.get(f"/api/donate/recordings/{rec['id']}/audio").content == take()
    stranger = browser(app)
    enroll(stranger)
    assert stranger.get(f"/api/donate/recordings/{rec['id']}/audio").status_code == 404
    assert c.get(f"/api/admin/donations/recordings/{rec['id']}/audio").status_code == 401
    login(c)
    assert c.get(f"/api/admin/donations/recordings/{rec['id']}/audio").status_code == 200


def test_rerecording_replaces_pending_take(app):
    c = browser(app)
    enroll(c)
    nxt, _ = record(c)
    url = f"/api/donate/recordings/{nxt['prompt_set']}/{nxt['prompt_id']}"
    assert c.post(url, content=take(gain=0.2)).status_code == 200
    assert len(c.get("/api/donate/me").json()["recordings"]) == 1 and len(wavs(app)) == 1


@pytest.mark.parametrize("cer,state", [(0.05, "accepted"), (0.3, "needs_review"), (0.8, "rejected")])
def test_worker_asr_check_sets_state(app, cer, state):
    c = browser(app)
    enroll(c)
    nxt, _ = record(c)
    seen = []

    def score(path, text):
        seen.append(text)
        return cer

    sets = load_prompt_sets()
    assert process_donation(app.state.root, sets, score, "rani-test")
    assert not process_donation(app.state.root, sets, score, "rani-test")
    assert seen == [nxt["text"]]
    assert c.get("/api/donate/me").json()["recordings"][0]["state"] == state


def test_failed_asr_job_does_not_block_the_queue(app):
    c = browser(app)
    enroll(c)
    record(c)
    record(c)
    calls = []

    def score(path, text):
        calls.append(path.name)
        if len(calls) == 1:
            raise RuntimeError("decoder crashed")
        return 0.0

    sets, skip = load_prompt_sets(), set()
    while process_donation(app.state.root, sets, score, "rani-test", skip):
        pass
    states = sorted(r["state"] for r in c.get("/api/donate/me").json()["recordings"])
    assert states == ["accepted", "asr_pending"] and len(calls) == 2


def test_reviewer_decides_borderline_takes(app):
    c = browser(app)
    enroll(c)
    record(c)
    process_donation(app.state.root, load_prompt_sets(), lambda p, t: 0.3, "rani-test")
    admin = browser(app)
    login(admin)
    queue = admin.get("/api/admin/donations").json()["review_queue"]
    assert len(queue) == 1 and queue[0]["cer"] == 0.3
    url = f"/api/admin/donations/recordings/{queue[0]['id']}/decision"
    assert admin.post(url, json={"action": "accept", "note": "Clear read, minor slip"}).status_code == 200
    assert c.get("/api/donate/me").json()["counts"]["accepted"] == 1


def test_delete_and_withdraw_remove_audio(app):
    c = browser(app)
    code = enroll(c).json()["recovery_code"]
    record(c)
    record(c)
    first = c.get("/api/donate/me").json()["recordings"][0]["id"]
    assert c.delete(f"/api/donate/recordings/{first}").status_code == 200
    assert len(wavs(app)) == 1
    assert c.post("/api/donate/withdraw").json() == {"withdrawn": True}
    assert wavs(app) == []
    assert c.get("/api/donate/me").status_code == 401
    assert browser(app).post("/api/donate/recover", json={"code": code}).status_code == 404
    with db(app.state.root) as conn:
        row = conn.execute("SELECT profile, purposes FROM donors").fetchone()
        assert (row["profile"], row["purposes"]) == ("{}", "[]")
        assert conn.execute("SELECT COUNT(*) FROM donor_recordings").fetchone()[0] == 0


def test_paid_programme_application_earnings_and_payouts(app):
    c = browser(app)
    enroll(c)
    donor_id = c.get("/api/donate/me").json()["id"]
    app.state.donor_rate = None
    assert c.post("/api/donate/paid/apply", json={"method": "esewa", "handle": "9800000000",
                                                  "name": "Test Donor"}).status_code == 403
    app.state.donor_rate = 5.0
    assert c.post("/api/donate/paid/apply", json={"method": "esewa", "handle": "9800000000",
                                                  "name": "Test Donor"}).status_code == 200
    for _ in range(3):
        record(c, take(seconds=10))
    while process_donation(app.state.root, load_prompt_sets(), lambda p, t: 0.0, "rani-test"):
        pass
    assert c.get("/api/donate/me").json()["earned_npr"] == 0  # not yet approved
    admin = browser(app)
    login(admin)
    apps = admin.get("/api/admin/donations").json()["applications"]
    assert apps[0]["payout"]["handle"] == "9800000000"
    base = f"/api/admin/donations/donors/{donor_id}"
    assert admin.post(base + "/payouts", json={"amount_npr": 1, "reference": "x1y"}).status_code == 409
    assert admin.post(base + "/paid", json={"action": "approve", "rate_npr_per_minute": 6}).status_code == 200
    me = c.get("/api/donate/me").json()
    assert me["accepted_minutes"] == 0.5 and me["earned_npr"] == 3.0
    assert admin.post(base + "/payouts", json={"amount_npr": 4, "reference": "esewa-123"}).status_code == 409
    paid = admin.post(base + "/payouts", json={"amount_npr": 2, "reference": "esewa-123"}).json()
    assert paid["paid_npr"] == 2 and paid["balance_npr"] == 1
    assert "payout" not in json.dumps(c.get("/api/donate/info").json()["me"])


def test_open_prompt_sets_are_validated(tmp_path):
    good = {"sets": [{"id": "open-news", "source": "Example corpus", "license": "CC-BY-4.0",
                      "prompts": [{"id": "n1", "text": "आज मौसम सफा छ।"}]}]}
    path = tmp_path / "sets.json"
    path.write_text(json.dumps(good, ensure_ascii=False))
    sets = load_prompt_sets(path)
    assert sets["open-news"]["kind"] == "open" and sets[ANCHOR]["kind"] == "evaluation_only"
    benchmark_text = next(iter(sets[ANCHOR]["prompts"].values()))
    for change in ({"license": ""}, {"id": "anchor-v1"},
                   {"prompts": [{"id": "n1", "text": benchmark_text}]},
                   {"prompts": [{"id": "n1", "text": "English only"}]}):
        path.write_text(json.dumps({"sets": [{**good["sets"][0], **change}]}, ensure_ascii=False))
        with pytest.raises(ValueError):
            load_prompt_sets(path)
    assert donations.asr_state(0.15) == "accepted" and donations.asr_state(0.41) == "rejected"
