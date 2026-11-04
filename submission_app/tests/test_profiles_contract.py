"""Submission disclosure and judge separation are comparison boundaries."""

import json
import time

import pytest

from submission_app.contract import Manifest, template
from submission_app.profiles import judge_profiles
from submission_app.server import create_app
from submission_app.storage import db
from submission_app.tests.test_flow import client


def manifest(**changes):
    return {**template(), "permission_to_evaluate_and_publish": True, **changes}


@pytest.mark.parametrize(
    "field",
    [
        "generation_mode",
        "candidate_count",
        "selector_model",
        "judge_family_used",
        "judge_family_details",
    ],
)
def test_structured_selection_disclosure_is_required(field):
    data = manifest()
    data.pop(field)
    with pytest.raises(ValueError):
        Manifest.model_validate(data)


@pytest.mark.parametrize(
    "changes",
    [
        {"candidate_count": 2},
        {"selector_model": "Whisper"},
        {"generation_mode": "best_of_n", "candidate_count": 1},
        {
            "generation_mode": "best_of_n",
            "candidate_count": 8,
            "selector_model": "none",
        },
        {"candidate_count": True},
        {"candidate_count": "1"},
        {"judge_family_used": "no"},
        {"judge_family_details": ""},
    ],
)
def test_inconsistent_or_coerced_declarations_are_rejected(changes):
    with pytest.raises(ValueError):
        Manifest.model_validate(manifest(**changes))


def test_best_of_n_discloses_selector_and_judge_family():
    data = Manifest.model_validate(
        manifest(
            generation_mode="best_of_n",
            candidate_count=8,
            selector_model="MMS-1b-all npi",
            judge_family_used=True,
            judge_family_details="MMS selected minimum CER; no RL",
        )
    )
    assert data.candidate_count == 8
    assert data.judge_family_used is True


def test_default_single_take_and_explicit_profile_filter(tmp_path):
    app = create_app(tmp_path, "testing-reviewer-password")
    c = client(app)
    declarations = [
        ("single-a", "judge-a", manifest()),
        ("single-b", "judge-b", manifest()),
        (
            "best",
            "judge-a",
            manifest(
                generation_mode="best_of_n", candidate_count=8, selector_model="manual"
            ),
        ),
        (
            "legacy",
            "judge-a",
            {"system_name": "Legacy", "selection_policy": "single take"},
        ),
    ]
    with db(tmp_path) as conn:
        for sid, profile, declaration in declarations:
            conn.execute(
                "INSERT INTO submissions(id,token_hash,created,updated,state,manifest,validation,audio_sha256,profile,report) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    sid,
                    "unused",
                    time.time(),
                    time.time(),
                    "published",
                    json.dumps(declaration),
                    "{}",
                    sid,
                    profile,
                    "{}",
                ),
            )
    data = c.get("/api/leaderboard").json()
    assert {e["id"] for e in data["entries"]} == {"single-a", "single-b"}
    assert data["total_approved"] == 4
    assert [
        e["id"] for e in c.get("/api/leaderboard?profile=judge-b").json()["entries"]
    ] == ["single-b"]
    assert [
        e["id"] for e in c.get("/api/leaderboard?selection=best_of_n").json()["entries"]
    ] == ["best"]
    assert [
        e["id"] for e in c.get("/api/leaderboard?selection=unknown").json()["entries"]
    ] == ["legacy"]
    assert len(c.get("/api/leaderboard?selection=all").json()["entries"]) == 4
    assert c.get("/api/leaderboard?selection=unfiltered").status_code == 422


def test_profile_catalog_is_aggregate_and_does_not_change_report_identity(
    tmp_path, monkeypatch
):
    from submission_app import profiles

    catalog = tmp_path / "calibration.json"
    profile = {
        "id": "exact-profile",
        "status": "pending",
        "calibration": {"pv": None, "fresh": None},
    }
    catalog.write_text(json.dumps({"schema_version": 1, "profiles": [profile]}))
    monkeypatch.setattr(profiles, "CALIBRATION", catalog)
    assert judge_profiles() == [profile]
    app = create_app(tmp_path / "app", "testing-reviewer-password")
    assert client(app).get("/api/judge-profiles").json()["profiles"] == [profile]
    catalog.write_text(
        json.dumps({"schema_version": 1, "profiles": [profile, profile]})
    )
    with pytest.raises(ValueError, match="Duplicate"):
        judge_profiles()


def test_checked_in_profiles_match_complete_calibration_and_owner_gate():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    numbers = json.loads(
        (root / "benchmark/judge_calibration_v1/calibration/numbers.json").read_text()
    )
    profiles = judge_profiles()
    assert len(profiles) == 7
    assert {p["stage"] for p in profiles} == {
        "asr_small",
        "asr_long1c",
        "asr_mms",
        "asr_whisper",
        "d32_streaming",
        "d32_full",
    }
    for profile in profiles:
        for cohort, n in [("pv", 100), ("fresh", 96)]:
            measured = numbers["asr"][profile["primary_policy"]][profile["stage"]][
                cohort
            ]["real"]
            assert profile["calibration"][cohort] == measured
            assert measured["n"] == n
            assert len(measured["ci95_corpus_speaker_bootstrap"]) == 2
            assert len(measured["ci95_mean_speaker_bootstrap"]) == 2

    small = next(p for p in profiles if p["stage"] == "asr_small")
    assert small["id"] == "whisper-small-ne-048909d9dafd"
    assert small["status"] == "uncalibrated for spontaneous Nepali"
    assert small["public_enabled"] is True
    assert all(not p["public_enabled"] for p in profiles if p is not small)
    assert all(
        "owner decision" in p["availability"]
        for p in profiles
        if p["stage"].startswith("d32_")
    )


def test_private_executable_d32_card_matches_profile_identity_and_calibration():
    from neptts_eval import d32

    from submission_app.contract import BENCHMARK
    from submission_app.profiles import CALIBRATION

    original = CALIBRATION.read_bytes()
    catalog = json.loads(original)["profiles"]
    profiles = judge_profiles()
    assert profiles[:6] == catalog
    card = profiles[-1]
    source = next(profile for profile in catalog if profile["stage"] == "d32_full")
    assert card["id"] == d32.profile(BENCHMARK)["id"]
    assert card["instrument"] == d32.profile(BENCHMARK)
    assert card["source_calibration_id"] == source["id"]
    for key in ("calibration", "controls", "retest", "primary_policy", "stage"):
        assert card[key] == source[key]
    assert card["calibration"]["pv"]["corpus_cer_percent"] == pytest.approx(
        7.875536480686695
    )
    assert card["calibration"]["fresh"]["corpus_cer_percent"] == pytest.approx(
        13.68944386634293
    )
    assert card["public_enabled"] is False
    assert card["offline_enabled"] is True
    assert "Call-trained weights are not released" in card["availability"]
    assert "reproducible only by Ampixa" in card["availability"]
    assert "owner decision on weight release" in card["availability"]
    assert "no public worker" in card["availability"]
    assert CALIBRATION.read_bytes() == original


@pytest.mark.parametrize(
    "field", ["profile", "calibration", "synthetic_score", "report"]
)
def test_upload_metadata_cannot_choose_judge_or_inject_scores(field):
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        Manifest.model_validate(manifest(**{field: {"id": "untrusted", "cer": 0}}))


def test_report_metadata_cannot_override_checked_in_calibration(tmp_path, monkeypatch):
    from submission_app import profiles

    catalog = tmp_path / "calibration.json"
    trusted = {
        "id": "judge-a",
        "status": "uncalibrated",
        "calibration": {"fresh": None},
    }
    catalog.write_text(json.dumps({"schema_version": 1, "profiles": [trusted]}))
    monkeypatch.setattr(profiles, "CALIBRATION", catalog)
    root = tmp_path / "app"
    app = create_app(root, "testing-reviewer-password")
    with db(root) as conn:
        conn.execute(
            "INSERT INTO submissions(id,token_hash,created,updated,state,manifest,validation,audio_sha256,profile,report) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                "fixture",
                "unused",
                time.time(),
                time.time(),
                "published",
                json.dumps(manifest()),
                "{}",
                "fixture",
                "judge-a",
                json.dumps(
                    {"profile": {"id": "judge-a", "calibration": {"fresh": {"cer": 0}}}}
                ),
            ),
        )
    c = client(app)
    assert c.get("/api/judge-profiles").json()["profiles"] == [trusted]
    assert c.get("/api/leaderboard").json()["entries"][0]["profile"] == "judge-a"


def test_historical_study_snapshot_matches_named_paper_and_scope(tmp_path):
    import hashlib
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    app = create_app(tmp_path, "testing-reviewer-password")
    snapshot = client(app).get("/api/historical-study").json()
    assert (
        snapshot["source"]["sha256"]
        == hashlib.sha256((root / snapshot["source"]["path"]).read_bytes()).hexdigest()
    )
    scope = json.loads((root / "neptts-eval/documentation_scopes_v1.json").read_text())
    assert snapshot["counts"] == scope["scopes"]["paper_native_human_study"]
    assert len(snapshot["systems"]) == 10
    assert len(snapshot["human_baselines"]) == 2
    assert (
        sum(row["ratings"] for row in snapshot["systems"] + snapshot["human_baselines"])
        == 6962
    )
    assert snapshot["systems"][0] == {
        "name": "TingTing Asmita",
        "mos": 3.55,
        "ci95": [3.27, 3.88],
        "ratings": 651,
    }
