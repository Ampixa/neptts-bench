"""Production readiness: shadow review, first-party seeding, health, readiness and arena extras."""

import csv
import io
import json
import random

import pytest

from submission_app import arena, autoreview, doctor, readiness, seed
from submission_app.contract import SENTENCES
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_arena import battle, listener, publish, start, wav
from submission_app.tests.test_autoreview import evaluate, row, upload
from submission_app.tests.test_flow import client, decision, login
from submission_app.worker import heartbeat


@pytest.fixture
def app(tmp_path):
    app = create_app(tmp_path, "testing-reviewer-password")
    app.state.arena_rng = random.Random(7)
    clock = {"t": 1_000_000.0}
    app.state.arena_clock = lambda: clock["t"]
    app.state.test_clock = clock
    return app


@pytest.fixture(autouse=True)
def no_hidden_checks(monkeypatch):
    monkeypatch.setattr(arena, "CHECK_RATE", 0.0)


# ------------------------------------------------------------------ shadow review

def test_shadow_mode_records_the_decision_but_publishes_nothing(app, monkeypatch):
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "shadow")
    c = client(app)
    clean = upload(c, 0)
    held = upload(c, 9, system_name="Your TTS")  # tone(i, j) bands must not overlap
    assert clean["state"] == held["state"] == "queued"  # still queued automatically
    evaluate(app)
    evaluate(app)
    for sub, expected in ((clean, []), (held, ["placeholder"])):
        r = row(app, sub["id"])
        record = json.loads(r["auto_review"])
        assert r["state"] == "awaiting_review"
        assert record["shadow"] is True and record["reasons"] == expected
    login(c)
    assert decision(c, clean["id"], "approve").json()["state"] == "published"
    assert decision(c, held["id"], "approve").json()["state"] == "published"
    with db(app.state.root) as conn:
        agreement = autoreview.shadow_agreement(conn)
    assert agreement["decided"] == 2 and agreement["agreement"] == 0.5
    assert agreement["would_hold"]["published"] == 1


def test_mode_parsing(monkeypatch):
    for value, mode in (("off", "off"), ("0", "off"), ("shadow", "shadow"), ("on", "on"), ("", "on")):
        monkeypatch.setenv("NEPTTS_AUTO_REVIEW", value)
        assert autoreview.mode() == mode
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "shadow")
    assert autoreview.enabled() and not autoreview.publishes()


# ------------------------------------------------------------------ first-party seeding

def wav_dir(tmp_path, i):
    d = tmp_path / f"wavs-{i}"
    d.mkdir()
    for sid in SENTENCES:
        (d / f"{sid}.wav").write_bytes(wav(i))
    return d


FIELDS = {"system_name": "TingTing Asmita", "model_version": "api-2026-10", "organization": "TingTing",
          "voice": "asmita", "generation_settings": "default API settings, 22.05 kHz",
          "training_overlap": "unknown"}


def test_seeded_baseline_is_scored_like_any_submission_and_labelled(app, monkeypatch, tmp_path):
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "on")
    root = app.state.root
    baseline_wavs = wav_dir(tmp_path, 3)
    sid, token = seed.seed(root, baseline_wavs, FIELDS, "baseline", "generated 2026-11-20")
    r = row(app, sid)
    assert r["state"] == "queued" and r["owner_id"] == seed.OWNER
    evaluate(app)
    assert row(app, sid)["state"] == "published"
    human, _ = seed.seed(root, wav_dir(tmp_path, 4), {**FIELDS, "system_name": "Human reference",
                                                      "organization": "Ampixa", "voice": "speaker-01"},
                         "human_reference")
    evaluate(app)
    with db(root) as c:
        assert set(arena.eligible(c)) == {sid, human}  # playback consent is implied
        assert arena.kinds(c) == {sid: "baseline", human: "human_reference"}
    with pytest.raises(ValueError, match="already"):
        seed.seed(root, baseline_wavs, FIELDS, "baseline")


def test_seed_refuses_incomplete_audio(app, tmp_path):
    d = wav_dir(tmp_path, 5)
    next(d.iterdir()).unlink()
    with pytest.raises(ValueError, match="no WAV"):
        seed.seed(app.state.root, d, FIELDS, "baseline")


def test_placeholder_baseline_is_held(app, monkeypatch, tmp_path):
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "on")
    sid, _ = seed.seed(app.state.root, wav_dir(tmp_path, 6), {}, "baseline")
    evaluate(app)
    assert row(app, sid)["state"] == "awaiting_review"


# ------------------------------------------------------------------ health and readiness

def test_health_paths_and_readiness(app):
    c = client(app)
    assert c.get("/api/health").json()["ok"] is True
    for path in ("/arena", "/leaderboard", "/methodology", "/donate"):
        r = c.get(path)
        assert r.status_code == 200 and "<!doctype html>" in r.text.lower()
    assert c.get("/api/admin/readiness").status_code == 401
    login(c)
    report = c.get("/api/admin/readiness").json()
    statuses = {r["id"]: r["status"] for r in report["checks"]}
    assert report["ready"] is False
    assert statuses["github_oauth"] == "todo" and statuses["worker"] == "todo"
    assert statuses["donation_consent"] == "todo"  # still a draft
    assert all(r["fix"] for r in report["checks"])


def test_worker_heartbeat_and_backup_marker_clear_checks(app, tmp_path):
    root = app.state.root
    heartbeat(root, "rani", "primary", "rani-test", True, 0.0)
    (root / readiness.MARKER).touch()
    env = {"NEPTTS_ORIGIN": "https://ampixa.benchmarks.cloud", "NEPTTS_ADMIN_PASSWORD": "x" * 40,
           "NEPTTS_GITHUB_CLIENT_ID": "id", "NEPTTS_GITHUB_CLIENT_SECRET": "secret"}
    statuses = {r["id"]: r["status"] for r in readiness.checks(root, env)}
    assert statuses["worker"] == statuses["backup"] == statuses["origin"] == statuses["github_oauth"] == "ok"


def test_doctor_cli_exit_status(tmp_path, capsys):
    assert doctor.main(["--data", str(tmp_path / "data")]) == 1
    assert "NOT READY" in capsys.readouterr().out
    assert doctor.main(["--data", str(tmp_path / "data"), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ready"] is False


# ------------------------------------------------------------------ arena extras

def test_sample_sentence_never_battles_and_samples_need_eligibility(app):
    owner = client(app)
    a, _ = publish(app, owner, 0)
    b, _ = publish(app, owner, 1)
    hidden, _ = publish(app, owner, 2, consent=False)
    c = listener(app)
    start(c, app)
    for _ in range(12):
        bat = battle(c, app)
        assert c.post(f"/api/arena/battles/{bat['token']}/vote", json={"choice": "a"}).status_code == 200
    with db(app.state.root) as conn:
        used = {r["sent_id"] for r in conn.execute("SELECT sent_id FROM arena_battles")}
    assert arena.SAMPLE_SENTENCE not in used
    assert c.get(f"/api/arena/samples/{a}").status_code == 200
    assert c.get(f"/api/arena/samples/{hidden}").status_code == 404
    board = c.get("/api/arena/leaderboard").json()
    assert "_counted" not in board and board["sample_sentence"]["id"] == arena.SAMPLE_SENTENCE
    assert {e["kind"] for e in board["entries"]} == {None}


def test_region_is_optional_and_validated(app):
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    assert c.post("/api/arena/session", json={"native_speaker": True, "headphones": False,
                                              "region": "mars"}).status_code == 422
    assert c.post("/api/arena/session", json={"native_speaker": True, "headphones": False,
                                              "region": "madhesh_terai"}).status_code == 200
    with db(app.state.root) as conn:
        assert conn.execute("SELECT region FROM arena_listeners").fetchone()["region"] == "madhesh_terai"


def test_vote_export_is_reviewer_only_unless_published(app, monkeypatch):
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    start(c, app)
    for _ in range(3):
        bat = battle(c, app)
        c.post(f"/api/arena/battles/{bat['token']}/vote", json={"choice": "both_good"})
    anonymous = listener(app)
    assert anonymous.get("/api/arena/votes.csv").status_code == 401
    monkeypatch.setenv("NEPTTS_ARENA_PUBLIC_EXPORT", "on")
    rows = list(csv.DictReader(io.StringIO(anonymous.get("/api/arena/votes.csv").text)))
    assert len(rows) == 3 and {r["listener"] for r in rows} == {"1"}
    assert all(r["score_a"] == "0.5" and r["category"] for r in rows)


def test_board_refreshes_on_a_timer_not_on_every_vote(app, monkeypatch):
    monkeypatch.setattr(arena, "RATINGS_REFRESH", 60)
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    start(c, app)
    assert c.get("/api/arena/leaderboard").json()["votes"] == 0
    bat = battle(c, app)
    c.post(f"/api/arena/battles/{bat['token']}/vote", json={"choice": "a"})
    assert c.get("/api/arena/leaderboard").json()["votes"] == 0  # cached
    app.state.test_clock["t"] += 61
    assert c.get("/api/arena/leaderboard").json()["votes"] == 1


def test_per_category_ratings_appear_with_enough_votes(app, monkeypatch):
    monkeypatch.setattr(arena, "MIN_CATEGORY_VOTES", 2)
    owner = client(app)
    publish(app, owner, 0)
    publish(app, owner, 1)
    c = listener(app)
    start(c, app)
    for _ in range(10):
        bat = battle(c, app)
        c.post(f"/api/arena/battles/{bat['token']}/vote", json={"choice": "a"})
    board = c.get("/api/arena/leaderboard").json()
    assert board["categories"]
    assert all(set(v["elo"]) <= {e["id"] for e in board["entries"]} for v in board["categories"].values())
