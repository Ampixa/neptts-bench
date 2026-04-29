"""Compute per-system audio duration statistics for the TTS-9 set.

Output: benchmark/results/audio_stats.json
        Per system: {n, total_min, mean_s, median_s, min_s, max_s, std_s}
        Plus an aggregate row covering the whole TTS-9 set.
"""
import json
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
AUDIO_ROOT = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs")
OUT = ROOT / "benchmark" / "results" / "audio_stats.json"

SYSTEMS = [
    "edge_tts/hemkala",
    "edge_tts/sagar",
    "gtts",
    "gemini",
    "piper",
    "tingting_asmita",
    "tingting_sambriddhi",
    "tingting_subina",
    "elevenlabs",
]


def duration_s(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def stats(arr):
    a = np.asarray(arr)
    return {
        "n": int(a.size),
        "total_min": float(a.sum() / 60.0),
        "mean_s": float(a.mean()),
        "median_s": float(np.median(a)),
        "min_s": float(a.min()),
        "max_s": float(a.max()),
        "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "p10_s": float(np.percentile(a, 10)),
        "p90_s": float(np.percentile(a, 90)),
    }


def main():
    per_system = {}
    all_durations = []
    for sys_name in SYSTEMS:
        d = AUDIO_ROOT / sys_name
        files = sorted(
            f for f in (list(d.glob("*.mp3")) + list(d.glob("*.wav")))
            if not f.name.startswith("._")
        )
        durations = []
        for f in files:
            try:
                durations.append(duration_s(f))
            except Exception as e:
                print(f"  ERR {f.name}: {e}")
        per_system[sys_name] = stats(durations)
        all_durations.extend(durations)
        s = per_system[sys_name]
        print(f"  {sys_name:<28} n={s['n']:>4}  total={s['total_min']:6.1f}m  "
              f"med={s['median_s']:5.2f}s  mean={s['mean_s']:5.2f}s  "
              f"min={s['min_s']:4.2f}  max={s['max_s']:5.2f}")

    aggregate = stats(all_durations)
    print(f"  {'TTS-9 (combined)':<28} n={aggregate['n']:>4}  "
          f"total={aggregate['total_min']:6.1f}m  med={aggregate['median_s']:5.2f}s")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(
        {"per_system": per_system, "aggregate": aggregate}, indent=2))
    print(f"\nWrote: {OUT}")


if __name__ == "__main__":
    main()
