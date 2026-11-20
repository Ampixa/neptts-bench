"""Blind pairwise arena: consent, screening, blinding, vote validity and ranking."""

import io
import json
import random
import zipfile

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from submission_app import arena, arena_audio
from submission_app.arena import digest
from submission_app.contract import SENTENCES, template
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import client, decision, login
from submission_app.worker import process_one

PASSWORD = "testing-reviewer-password"


@pytest.fixture
def app(tmp_path):
    app = create_app(tmp_path, PASSWORD)
    app.state.arena_rng = random.Random(7)
    clock = {"t": 1_000_000.0}
    app.state.arena_clock = lambda: clock["t"]
    app.state.test_clock = clock
    return app


@pytest.fixture(autouse=True)
def no_hidden_checks(monkeypatch):
    monkeypatch.setattr(arena, "CHECK_RATE", 0.0)


def wav(i, rate=16000, gain=0.2):
    data = io.BytesIO()
    t = np.arange(int(rate * 0.5)) / rate
    sf.write(data, np.sin(2 * np.pi * (180 + 20 * i) * t) * gain, rate, format="WAV", subtype="PCM_16")
    return data.getvalue()


def publish(app, c, i, consent=True, rate=16000, gain=0.2):
    manifest = template()
    manifest.update(system_name=f"System {i}", permission_to_evaluate_and_publish=True,
                    permission_to_play_audio_to_listeners=consent)
    body = io.BytesIO()
    with zipfile.ZipFile(body, "w") as z:
        z.writestr("submission.json", json.dumps(manifest))
        for sid in SENTENCES:
            z.writestr(f"audio/{sid}.wav", wav(i, rate, gain))
    r = c.post("/api/submissions", content=body.getvalue(), headers={"Content-Type": "application/zip"})
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    login(c)
    assert decision(c, sid, "queue").status_code == 200
    spec = {"id": "unit-test-evaluator", "development_only": False}

    def scorer(files, model, progress):
        rows = {s: {"cer": 0.05, "wer": 0.1} for s in SENTENCES}
        return {"profile": spec, "per_sentence": rows, "cer": 0.05, "wer": 0.1,
                "human_mos": None, "predicted_mos": None}

    assert process_one(app.state.root, scorer=scorer, spec=spec)
    assert decision(c, sid, "approve").json()["state"] == "published"
    return sid, r.json()["token"]


def listener(app):
    c = TestClient(app)
    c.headers["X-Neptts-Request"] = "submission"
    return c


def start(c, app, pass_check=True):
    assert c.post("/api/arena/session", json={"native_speaker": True, "headphones": True}).status_code == 200
    items = c.post("/api/arena/screening").json()["items"]
    assert len(items) == 4
    with db(app.state.root) as conn:
        truth = {r["token_hash"]: r["answer"] for r in conn.execute("SELECT * FROM arena_screen_items")}
    answers = {i["token"]: truth[digest(i["token"])] if pass_check else (truth[digest(i["token"])] + 1) % 4
               for i in items}
    return c.post("/api/arena/screening/answers", json={"answers": answers}).json()


def battle(c, app, listen=True, wait=True):
    b = c.post("/api/arena/battles").json()
    if listen:
        for side in ("a", "b"):
            assert c.get(b["audio"][side]).status_code == 200
    if wait:
        app.state.test_clock["t"] += 5
    return b


def test_blind_flow_reveals_only_after_vote(app):
    owner = client(app)
    a, _ = publish(app, owner, 0)
    b, _ = publish(app, owner, 1)
    hidden, _ = publish(app, owner, 2, consent=False)
    c = listener(app)
    assert c.get("/api/arena/state").json() == {"listener": None, "systems": 2, "ready": True,
                                                 "tags": arena.TAGS,
                                                 "regions": [{"id": k, "ne": ne, "en": en}
                                                             for k, (ne, en) in arena.REGIONS.items()]}
    assert c.post("/api/arena/battles").status_code == 401
    assert c.post("/api/arena/session", json={"native_speaker": False, "headphones": True}).status_code == 422
    assert c.post("/api/arena/session", json={"native_speaker": True, "headphones": True}).status_code == 200
    assert c.post("/api/arena/battles").status_code == 403  # listening check first
    assert start(c, app)["passed"] is True
    payload = battle(c, app, listen=False, wait=False)
    text = json.dumps(payload)
    for secret in (a, b, hidden, "System 0", "System 1"):
        assert secret not in text
    assert payload["text"] in {s["text_devanagari"] for s in SENTENCES.values()}
    vote_url = f"/api/arena/battles/{payload['token']}/vote"
    assert c.post(vote_url, json={"choice": "a"}).status_code == 409  # must listen first
    for side in ("a", "b"):
        r = c.get(payload["audio"][side])
        assert "etag" not in r.headers and "last-modified" not in r.headers
    app.state.test_clock["t"] += 5
    stranger = listener(app)
    stranger.post("/api/arena/session", json={"native_speaker": True, "headphones": False})
    assert stranger.post(vote_url, json={"choice": "a"}).status_code == 404
    assert c.post(vote_url, json={"choice": "a", "tags": {"b": ["robotic"]}}).status_code == 200
    assert c.post(vote_url, json={"choice": "b"}).status_code == 409
    for _ in range(5):
        b2 = battle(c, app)
        r = c.post(f"/api/arena/battles/{b2['token']}/vote", json={"choice": "both_good"}).json()
        names = {r["reveal"]["a"]["system"], r["reveal"]["b"]["system"]}
        assert names == {"System 0", "System 1"} and r["kind"] == "compare"
    board = c.get("/api/arena/leaderboard").json()
    assert {e["id"] for e in board["entries"]} == {a, b}
    assert board["votes"] == 6 and board["listeners"] == 1
    assert all(e["provisional"] for e in board["entries"])
    assert "excluded_votes" not in board


def test_unknown_tags_and_cross_origin_rejected(app):
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    start(c, app)
    b = battle(c, app)
    url = f"/api/arena/battles/{b['token']}/vote"
    assert c.post(url, json={"choice": "a", "tags": {"a": ["made_up"]}}).status_code == 422
    raw = TestClient(app)
    assert raw.post("/api/arena/session", json={"native_speaker": True, "headphones": True}).status_code == 403


def test_clips_are_format_and_loudness_matched(app):
    owner = client(app)
    publish(app, owner, 0, rate=16000, gain=0.05)
    publish(app, owner, 1, rate=48000, gain=0.6)
    c = listener(app)
    start(c, app)
    b = battle(c, app, listen=False)
    clips = [sf.read(io.BytesIO(c.get(b["audio"][s]).content), always_2d=True) for s in ("a", "b")]
    assert {rate for _, rate in clips} == {arena_audio.RATE}
    assert all(x.shape[1] == 1 for x, _ in clips)
    levels = [20 * np.log10(arena_audio.active_rms(x[:, 0])) for x, _ in clips]
    assert all(abs(level - arena_audio.TARGET_DBFS) < 0.5 for level in levels)


def test_too_fast_votes_are_stored_but_not_counted(app):
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    start(c, app)
    b = battle(c, app, wait=False)
    assert c.post(f"/api/arena/battles/{b['token']}/vote", json={"choice": "a"}).status_code == 200
    with db(app.state.root) as conn:
        assert conn.execute("SELECT valid, reason FROM arena_votes").fetchone()[:] == (0, "too_fast")
    assert c.get("/api/arena/leaderboard").json()["votes"] == 0


def test_failed_hidden_checks_exclude_listener(app, monkeypatch):
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    start(c, app)
    monkeypatch.setattr(arena, "CHECK_RATE", 1.0)
    monkeypatch.setattr(arena, "CHECK_WARMUP", 0)
    for _ in range(3):
        b = battle(c, app)
        with db(app.state.root) as conn:
            row = conn.execute("SELECT kind, degraded_side FROM arena_battles WHERE token_hash=?",
                               (digest(b["token"]),)).fetchone()
        wrong = "a" if row["kind"] == "null" else row["degraded_side"]
        r = c.post(f"/api/arena/battles/{b['token']}/vote", json={"choice": wrong}).json()
        assert r["kind"] == "check"
    monkeypatch.setattr(arena, "CHECK_RATE", 0.0)
    for _ in range(3):
        b = battle(c, app)
        c.post(f"/api/arena/battles/{b['token']}/vote", json={"choice": "a"})
    board = c.get("/api/arena/leaderboard").json()
    assert board["votes"] == 0 and board["excluded_listeners"] == 1
    login(owner)
    assert owner.get("/api/admin/arena").json()["excluded_votes"] == {"excluded_listener": 3}


def test_degraded_clip_is_clearly_worse(app):
    owner = client(app)
    sid, _ = publish(app, owner, 0)
    clean, _ = arena_audio.playback(app.state.root, sid, "sent_001")
    bad, _ = arena_audio.playback(app.state.root, sid, "sent_001", degraded=True)
    x, y = sf.read(clean)[0], sf.read(bad)[0]
    noise = y - x * (np.dot(x, y) / np.dot(x, x))
    assert 10 * np.log10(np.mean(x**2) / np.mean(noise**2)) < 3  # SNR near 0 dB


def test_screening_attempts_are_limited(app):
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    for left in (2, 1, 0):
        result = start(c, app, pass_check=False) if left == 2 else None
        if result is None:
            items = c.post("/api/arena/screening").json()["items"]
            with db(app.state.root) as conn:
                truth = {r["token_hash"]: r["answer"] for r in conn.execute("SELECT * FROM arena_screen_items")}
            result = c.post("/api/arena/screening/answers", json={"answers": {
                i["token"]: (truth[digest(i["token"])] + 1) % 4 for i in items}}).json()
        assert result == {"passed": False, "correct": 0, "attempts_left": left}
    assert c.post("/api/arena/screening").status_code == 403
    assert c.post("/api/arena/battles").status_code == 403


def test_owner_controls_listener_consent(app):
    owner = client(app)
    publish(app, owner, 0)
    sid, token = publish(app, owner, 1, consent=False)
    c = listener(app)
    assert c.get("/api/arena/state").json()["systems"] == 1
    url = f"/api/submissions/{sid}/listening-consent"
    assert c.post(url, json={"granted": True}).status_code == 404  # strangers cannot decide
    login(c)
    assert c.post(url, json={"granted": True}).status_code == 404  # nor can reviewers
    assert owner.post(url, json={"granted": True}).json()["listening_consent"] is True
    assert c.get("/api/arena/state").json()["systems"] == 2
    arena_audio.playback(app.state.root, sid, "sent_001")
    receipt = listener(app)
    r = receipt.post(url, json={"granted": False}, headers={"X-Submission-Token": token})
    assert r.status_code == 200
    assert not (app.state.root / "arena_cache" / sid).exists()
    assert owner.get(f"/api/submissions/{sid}").json()["listening_consent"] is False
    assert c.get("/api/arena/state").json()["systems"] == 1


def test_daily_and_lifetime_comparison_caps(app, monkeypatch):
    owner = client(app)
    for i in range(3):
        publish(app, owner, i)
    monkeypatch.setattr(arena, "DAILY_CAP", 3)
    monkeypatch.setattr(arena, "LIFETIME_CAP", 5)
    c = listener(app)
    start(c, app)
    lefts = [c.post("/api/arena/battles").json()["left_today"] for _ in range(3)]
    assert lefts == [2, 1, 0]
    r = c.post("/api/arena/battles")
    assert r.status_code == 429 and "tomorrow" in r.json()["detail"]
    assert c.get("/api/arena/state").json()["listener"]["left_today"] == 0
    app.state.test_clock["t"] += 86400 + 1
    assert c.post("/api/arena/battles").status_code == 200
    assert c.post("/api/arena/battles").json()["left_today"] == 0  # lifetime cap is nearer
    r = c.post("/api/arena/battles")
    assert r.status_code == 409 and "limit of 5" in r.json()["detail"]


def test_network_address_cap_stops_fresh_sessions(app, monkeypatch):
    owner = client(app)
    for i in range(2):
        publish(app, owner, i)
    monkeypatch.setattr(arena, "IP_DAILY_CAP", 2)
    first = listener(app)
    start(first, app)
    assert first.post("/api/arena/battles").status_code == 200
    second = listener(app)  # same machine, new browser session
    start(second, app)
    assert second.post("/api/arena/battles").status_code == 200
    assert second.post("/api/arena/battles").status_code == 429


def test_prolific_listener_is_capped_in_the_ranking(app, monkeypatch):
    owner = client(app)
    a, _ = publish(app, owner, 0)
    b, _ = publish(app, owner, 1)
    monkeypatch.setattr(arena, "VOTE_WEIGHT_CAP", 3)

    def vote_for(c, winner, n):
        for _ in range(n):
            bat = battle(c, app)
            with db(app.state.root) as conn:
                row = conn.execute("SELECT sub_a FROM arena_battles WHERE token_hash=?",
                                   (digest(bat["token"]),)).fetchone()
            side = "a" if row["sub_a"] == winner else "b"
            assert c.post(f"/api/arena/battles/{bat['token']}/vote", json={"choice": side}).status_code == 200

    heavy = listener(app)
    start(heavy, app)
    vote_for(heavy, a, 12)
    for _ in range(3):
        c = listener(app)
        start(c, app)
        vote_for(c, b, 3)
    board = c.get("/api/arena/leaderboard").json()
    assert board["capped_listeners"] == 1
    assert board["entries"][0]["id"] == b  # 3 listeners outweigh 1 prolific one
