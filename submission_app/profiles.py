"""Display calibration independently of immutable evaluator/report identities."""

import json
from copy import deepcopy
from pathlib import Path

CALIBRATION = (
    Path(__file__).resolve().parents[1]
    / "benchmark/judge_calibration_v1/calibration_public.json"
)


def judge_profiles():
    """Only aggregate, reviewed calibration belongs in the public catalog."""
    if not CALIBRATION.exists():
        return []
    payload = json.loads(CALIBRATION.read_text())
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("profiles"), list
    ):
        raise ValueError("Unsupported judge calibration catalog.")
    profiles = payload["profiles"]
    ids = [p["id"] for p in profiles]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate judge calibration profile IDs.")
    calibrated = next(
        (
            profile
            for profile in profiles
            if profile.get("id") == "d32-v2-step60000-fullcontext-legacy-v1"
            and profile.get("stage") == "d32_full"
            and profile.get("primary_policy") == "translit"
        ),
        None,
    )
    if calibrated is not None:
        # The existing evaluator installs the repository-local toolkit path.
        # These imports load public identity code only, never private weights.
        from neptts_eval import d32

        from .evaluation import BENCHMARK

        instrument = d32.profile(BENCHMARK)
        if instrument["id"] in ids:
            raise ValueError("Duplicate executable D32 profile ID.")
        executable = deepcopy(calibrated)
        executable.update(
            id=instrument["id"],
            name="D32 v2 step 60000 · private offline",
            status="private owner-operated offline profile; independent qualification incomplete",
            public_enabled=False,
            offline_enabled=True,
            availability=(
                "Private owner-operated offline scoring only; no public worker. "
                "Call-trained weights are not released yet; per the owner decision on weight release, "
                "public release is planned after Tihar 2026. Until then scores are "
                "reproducible only by Ampixa."
            ),
            source_calibration_id=calibrated["id"],
            instrument=instrument,
        )
        profiles.append(executable)
    profiles.extend(rani_profiles(ids))
    return profiles


def primary_profile_id():
    """The judge that ranks the main leaderboard: Rani, once its weights are pinned."""
    from neptts_eval import rani

    from .evaluation import BENCHMARK

    return rani.profile(BENCHMARK)["id"] if rani.is_pinned() else None


def rani_profiles(existing_ids):
    """Rani appears only once its contract pins exact weights and runtime."""
    from neptts_eval import rani

    from .evaluation import BENCHMARK

    if not rani.is_pinned():
        return []
    instrument = rani.profile(BENCHMARK)
    if instrument["id"] in existing_ids:
        raise ValueError("Duplicate Rani profile ID.")
    return [{
        "id": instrument["id"],
        "name": "Rani · Ampixa Labs",
        "stage": "rani",
        "primary_policy": "neptel_v02",
        "calibration_policy": instrument["normalization"]
        + "; digits to spoken Nepali words; CER excludes spaces",
        "loanword_policy": "English in either script maps to one Devanagari form (Rani latin_map).",
        "status": "primary judge · intelligibility proxy; real-speech calibration pending",
        "public_enabled": True,
        "provisional": True,
        "availability": "Arena worker profile. Credit: " + instrument["attribution"]
        + ". Weights public with gated access approval.",
        "limitations": [
            "Trained for conversational and phone speech, so it may forgive mispronounced minimal pairs that listeners would notice.",
            "Upstream scorer spells numbers with short-i forms, giving every system the same small penalty on the one prompt with digits.",
            "Rani and the arena are both run by Ampixa; submissions that used Rani for selection must declare it as judge-family use.",
        ],
        "instrument": instrument,
    }]


def selection_group(manifest):
    """Older free-text declarations cannot establish single-take eligibility."""
    mode = manifest.get("generation_mode")
    required = (
        "candidate_count",
        "selector_model",
        "judge_family_used",
        "judge_family_details",
    )
    if not all(key in manifest for key in required):
        return "unknown"
    count = manifest["candidate_count"]
    selector = manifest["selector_model"]
    if (
        type(count) is not int
        or not isinstance(selector, str)
        or not selector.strip()
        or type(manifest["judge_family_used"]) is not bool
        or not isinstance(manifest["judge_family_details"], str)
        or not manifest["judge_family_details"].strip()
    ):
        return "unknown"
    if mode == "single_take" and count == 1 and selector.lower() == "none":
        return "single_take"
    if mode == "best_of_n" and count >= 2 and selector.lower() != "none":
        return "best_of_n"
    return "unknown"
