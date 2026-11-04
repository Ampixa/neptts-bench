"""Additional judge results have their own approval and preserve legacy scores."""

import copy
import json
import time
import zipfile

import pytest

from submission_app import offline
from submission_app.contract import BENCHMARK, SENTENCES, template
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import client, login, submit
from submission_app.tests.test_offline import payload as whisper_payload


def d32_payload(sid="submission-0"):
    from neptts_eval import d32

    return {
        "submission_id": sid,
        "benchmark": BENCHMARK,
        "audio_sha256": sid,
        "report": {
            "profile": d32.profile(BENCHMARK),
            "per_sentence": {
                key: {
                    "cer": 0.1,
                    "wer": 0.2,
                    "char_errors": 1,
                    "chars": 10,
                    "word_errors": 1,
                    "words": 5,
                }
                for key in SENTENCES
            },
            "cer": 0.1,
            "wer": 0.2,
            "corpus_cer": 0.1,
            "corpus_wer": 0.2,
            "human_mos": None,
            "predicted_mos": None,
        },
    }


@pytest.fixture
def arena(tmp_path):
    app = create_app(tmp_path, "testing-reviewer-password")
    with db(tmp_path) as conn:
        for i in range(5):
            sid = f"submission-{i}"
            report = whisper_payload({})["report"]
            conn.execute(
                "INSERT INTO submissions(id,token_hash,created,updated,state,manifest,validation,audio_sha256,profile,report) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    "unused",
                    time.time(),
                    time.time(),
                    "published",
                    json.dumps(template()),
                    "{}",
                    sid,
                    report["profile"]["id"],
                    json.dumps(report),
                ),
            )
    return app, client(app)


def snapshot(root):
    with db(root) as conn:
        return [
            tuple(row) for row in conn.execute("SELECT * FROM submissions ORDER BY id")
        ]


def test_five_approved_rescores_are_separate_and_need_approval(arena):
    app, c = arena
    original = snapshot(app.state.root)
    legacy = c.get("/api/leaderboard?selection=all").json()
    body = d32_payload()
    profile = body["report"]["profile"]["id"]
    base = "/api/admin/submissions/submission-0/profiles/" + profile
    assert c.post(base + "/results", json=body).status_code == 401
    login(c)
    for i in range(5):
        sid = f"submission-{i}"
        base = f"/api/admin/submissions/{sid}/profiles/{profile}"
        body = d32_payload(sid)
        assert c.post(base + "/results", json=body).status_code == 200
        assert c.post(base + "/results", json=body).status_code == 409
        assert (
            c.get(f"/api/leaderboard/{sid}/report?profile={profile}").status_code == 404
        )
    assert (
        c.get(f"/api/leaderboard?selection=all&profile={profile}").json()["entries"]
        == []
    )
    for i in range(5):
        base = f"/api/admin/submissions/submission-{i}/profiles/{profile}"
        assert (
            c.post(
                base + "/decision",
                json={
                    "action": "approve",
                    "note": "Source hashes and full coverage reviewed",
                },
            ).status_code
            == 200
        )
        assert (
            c.post(
                base + "/decision", json={"action": "approve", "note": "Repeat"}
            ).status_code
            == 409
        )
    results = c.get(f"/api/leaderboard?selection=all&profile={profile}").json()
    assert len(results["entries"]) == 5
    assert {row["profile"] for row in results["entries"]} == {profile}
    assert results["total_approved"] == 5
    assert c.get("/api/leaderboard?selection=all").json() == legacy
    assert snapshot(app.state.root) == original
    for i in range(5):
        sid = f"submission-{i}"
        assert (
            c.get(f"/api/leaderboard/{sid}/report").json()["report"]
            == whisper_payload({})["report"]
        )
        assert (
            c.get(f"/api/leaderboard/{sid}/report?profile={profile}").json()["report"]
            == d32_payload(sid)["report"]
        )


@pytest.mark.parametrize(
    "change",
    [
        "identity",
        "audio",
        "coverage",
        "count",
        "corpus",
        "transcript",
        "mos",
        "profile",
    ],
)
def test_additional_import_rejects_unbound_or_private_evidence(arena, change):
    app, c = arena
    login(c)
    original = snapshot(app.state.root)
    body = d32_payload()
    profile = body["report"]["profile"]["id"]
    first = next(iter(SENTENCES))
    if change == "identity":
        body["submission_id"] = "wrong"
    elif change == "audio":
        body["audio_sha256"] = "wrong"
    elif change == "coverage":
        body["report"]["per_sentence"].pop(first)
    elif change == "count":
        body["report"]["per_sentence"][first]["char_errors"] = True
    elif change == "corpus":
        body["report"]["corpus_cer"] = 0.0
    elif change == "transcript":
        body["report"]["per_sentence"][first]["hyp"] = "private"
    elif change == "mos":
        body["report"]["human_mos"] = 5
    elif change == "profile":
        body["report"]["profile"]["weights_sha256"] = "c" * 64
    response = c.post(
        f"/api/admin/submissions/submission-0/profiles/{profile}/results", json=body
    )
    assert response.status_code == 422, response.text
    assert snapshot(app.state.root) == original
    assert c.get("/api/admin/submissions/submission-0/profiles").json() == []


def test_reject_and_wrong_profile_do_not_publish(arena):
    app, c = arena
    login(c)
    body = d32_payload()
    profile = body["report"]["profile"]["id"]
    base = f"/api/admin/submissions/submission-0/profiles/{profile}"
    assert (
        c.post(base.replace(profile, "wrong") + "/results", json=body).status_code
        == 422
    )
    assert c.post(base + "/results", json=body).status_code == 200
    assert (
        c.post(
            base + "/decision", json={"action": "reject", "note": "Fixture rejected"}
        ).status_code
        == 200
    )
    assert (
        c.get(f"/api/leaderboard?selection=all&profile={profile}").json()["entries"]
        == []
    )
    assert (
        c.get("/api/admin/submissions/submission-0/profiles").json()[0]["state"]
        == "rejected"
    )
    with db(app.state.root) as conn:
        conn.execute(
            "UPDATE submissions SET state='awaiting_review' WHERE id='submission-1'"
        )
    assert (
        c.post(
            base.replace("submission-0", "submission-1") + "/results",
            json=d32_payload("submission-1"),
        ).status_code
        == 409
    )


def test_report_validation_binds_counts_and_has_no_private_fields():
    body = d32_payload()
    offline.validate_import(body, "submission-0", "submission-0")
    bad = copy.deepcopy(body)
    bad["report"]["per_sentence"][next(iter(SENTENCES))]["cer"] += 0.1
    with pytest.raises(ValueError):
        offline.validate_import(bad, "submission-0", "submission-0")


def test_offline_cli_dispatches_complete_package_to_private_profile(
    tmp_path, monkeypatch
):
    import sys

    from neptts_eval import d32

    app = create_app(tmp_path / "state", "testing-reviewer-password")
    c = client(app)
    sid = submit(c).json()["id"]
    login(c)
    package = tmp_path / "evaluation.zip"
    package.write_bytes(c.get(f"/api/admin/submissions/{sid}/package").content)
    with zipfile.ZipFile(package) as archive:
        job = json.loads(archive.read("evaluation-job.json"))
    output = tmp_path / "results.json"
    seen = []

    def scorer(audio_dir, sentences, benchmark, config, progress):
        assert audio_dir.parent.parent == package.resolve().parent
        assert {path.stem for path in audio_dir.glob("*.wav")} == set(SENTENCES)
        assert sentences == SENTENCES and benchmark == BENCHMARK
        assert config == {
            **{
                key: tmp_path / key
                for key in ("checkpoint", "bpe", "recipe", "icefall", "normalizer_root")
            },
            "threads": 1,
        }
        seen.append(True)
        progress(len(sentences))
        return d32_payload()["report"]

    monkeypatch.setattr(d32, "evaluate", scorer)
    command = [
        "offline",
        "--package",
        str(package),
        "--output",
        str(output),
        "--profile",
        "d32v2-s60000-full-translit",
    ]
    for key in ("checkpoint", "bpe", "recipe", "icefall", "normalizer_root"):
        command += ["--" + key.replace("_", "-"), str(tmp_path / key)]
    monkeypatch.setattr(sys, "argv", command)
    offline.main()
    assert seen == [True]
    result = json.loads(output.read_text())
    assert {key: result[key] for key in job} == job
    assert offline.validate_import(result, sid, job["audio_sha256"])[
        "profile"
    ] == d32.profile(BENCHMARK)
