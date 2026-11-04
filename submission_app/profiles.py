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
        from .evaluation import BENCHMARK
        from neptts_eval import d32

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
                "Call-trained weights are not released. Scores are reproducible only "
                "by Ampixa until the owner decision on weight release."
            ),
            source_calibration_id=calibrated["id"],
            instrument=instrument,
        )
        profiles.append(executable)
    return profiles


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
