#!/usr/bin/env python3
"""Run automated MOS prediction on all TTS outputs.

Uses SCOREQ (NeurIPS 2024) for cross-domain TTS quality scoring.
Optionally uses UTMOS22 if available.
"""

import json
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import numpy as np
import torch
import torchaudio

# Patch torchaudio.load to use soundfile (avoids torchcodec/CUDA dep)
def _patched_load(filepath, *args, **kwargs):
    data, sr = sf.read(str(filepath), dtype='float32')
    if data.ndim == 1:
        data = data[np.newaxis, :]
    else:
        data = data.T
    return torch.from_numpy(data), sr

torchaudio.load = _patched_load

ROOT = Path(__file__).resolve().parent.parent
TTS_DIR = ROOT / "benchmark" / "data" / "tts_outputs"
HUMAN_DIR = ROOT / "benchmark" / "data" / "hetzner_recordings" / "audio"
RESULTS_DIR = ROOT / "benchmark" / "data" / "eval_results"


def find_all_audio():
    """Find all audio files organized by system."""
    systems = {}

    # TTS outputs
    for system_dir in sorted(TTS_DIR.iterdir()):
        if not system_dir.is_dir():
            continue
        name = system_dir.name
        if name == "edge_tts":
            for voice_dir in sorted(system_dir.iterdir()):
                if voice_dir.is_dir():
                    key = f"edge_tts/{voice_dir.name}"
                    files = sorted(voice_dir.glob("*.mp3"))
                    if files:
                        systems[key] = files
        elif name == "mms_tts":
            continue  # empty
        else:
            files = []
            for ext in ("*.mp3", "*.wav", "*.webm", "*.flac"):
                files.extend(sorted(system_dir.glob(ext)))
            if files:
                systems[name] = files

    # Human recordings (best speaker)
    if HUMAN_DIR.exists():
        best = max(
            (d for d in HUMAN_DIR.iterdir() if d.is_dir()),
            key=lambda d: sum(1 for _ in d.iterdir()),
            default=None,
        )
        if best:
            files = sorted(f for f in best.iterdir() if f.suffix in (".webm", ".wav"))
            if files:
                systems["human"] = files

    return systems


def run_scoreq(systems):
    """Run SCOREQ on all systems."""
    from scoreq import Scoreq

    print("Loading SCOREQ (NR synthetic)...")
    sq = Scoreq(data_domain="synthetic", mode="nr")

    results = {}
    for sys_name, files in sorted(systems.items()):
        print(f"\n=== {sys_name} ({len(files)} files) ===")
        scores = []
        errors = 0
        t0 = time.time()

        for i, f in enumerate(files):
            try:
                score = float(sq.predict(test_path=str(f)))
                scores.append({"file": f.stem, "scoreq_mos": round(score, 3)})
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"  ERROR {f.name}: {e}")

            if (i + 1) % 50 == 0:
                avg = sum(s["scoreq_mos"] for s in scores) / len(scores) if scores else 0
                print(f"  {i+1}/{len(files)} — avg: {avg:.3f}")

        elapsed = time.time() - t0
        avg = sum(s["scoreq_mos"] for s in scores) / len(scores) if scores else 0
        print(f"  Done: {len(scores)} scored, {errors} errors, avg={avg:.3f} ({elapsed:.1f}s)")

        results[sys_name] = {
            "scores": scores,
            "avg_mos": round(avg, 3),
            "n_scored": len(scores),
            "n_errors": errors,
        }

    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("AUTOMATED MOS RESULTS (SCOREQ)")
    print("=" * 60)
    print(f"{'System':<25} {'Files':>6} {'SCOREQ MOS':>12}")
    print("-" * 50)
    for sys_name in sorted(results, key=lambda k: results[k]["avg_mos"], reverse=True):
        r = results[sys_name]
        print(f"{sys_name:<25} {r['n_scored']:>6} {r['avg_mos']:>12.3f}")


def main():
    systems = find_all_audio()
    total = sum(len(f) for f in systems.values())
    print(f"Found {len(systems)} systems, {total} files total")

    results = run_scoreq(systems)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "auto_mos_scoreq.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
