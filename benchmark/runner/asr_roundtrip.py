"""ASR round-trip evaluation: Whisper transcription + CER/WER.

Loads generated audio from manifest, transcribes with Whisper,
computes CER/WER per item and per category, checks against thresholds.

Usage:
    python asr_roundtrip.py --manifest benchmark/results/edge-tts-hemkala/manifest.json
"""

import argparse
import sys
import time
from pathlib import Path

import jiwer
import numpy as np
import whisper

from utils import load_json, normalize_nepali_text, save_json


# Thresholds from asr_roundtrip_protocol.json
THRESHOLDS = {
    "cer": {
        "overall": {"pass": 0.10, "minimum": 0.20},
        "phonological_core": {"pass": 0.15, "minimum": 0.25},
        "conversational": {"pass": 0.08, "minimum": 0.18},
        "edge_cases": {"pass": 0.15, "minimum": 0.25},
        "english": {"pass": 0.12, "minimum": 0.22},
    },
    "wer": {
        "overall": {"pass": 0.15, "minimum": 0.25},
        "phonological_core": {"pass": 0.20, "minimum": 0.30},
        "conversational": {"pass": 0.12, "minimum": 0.22},
        "edge_cases": {"pass": 0.20, "minimum": 0.30},
        "english": {"pass": 0.15, "minimum": 0.25},
    },
}

# Map item categories to threshold categories
CATEGORY_MAP = {
    "phonological_minimal_pairs": "phonological_core",
    "contrastive_stress": "conversational",
    "question_intonation": "conversational",
    "emotion": "conversational",
    "homographs": "conversational",
    "robustness": "edge_cases",
    "long_form": "conversational",
    "phrases": "conversational",
}


def compute_cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    if len(ref_chars) == 0:
        return 0.0 if len(hyp_chars) == 0 else 1.0
    return jiwer.cer(reference, hypothesis)


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    if not reference.strip():
        return 0.0 if not hypothesis.strip() else 1.0
    return jiwer.wer(reference, hypothesis)


def judge(value: float, thresholds: dict) -> str:
    """Judge a metric value against pass/minimum thresholds."""
    if value <= thresholds["pass"]:
        return "pass"
    elif value <= thresholds["minimum"]:
        return "marginal"
    else:
        return "fail"


def run_asr(manifest_path: Path, model_size: str = "medium") -> dict:
    """Run ASR round-trip evaluation on all manifest items."""
    manifest = load_json(manifest_path)
    output_dir = manifest_path.parent

    # Filter to items with valid audio
    items = [m for m in manifest if m.get("audio_path") and m["status"] in ("ok", "cached")]
    print(f"Loaded {len(items)} items with audio (of {len(manifest)} total)")

    # Load Whisper model
    print(f"Loading Whisper {model_size} model...")
    t0 = time.time()
    model = whisper.load_model(model_size)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    results = []
    t0 = time.time()

    for i, item in enumerate(items):
        audio_path = item["audio_path"]
        reference = normalize_nepali_text(item["text"])

        # Transcribe
        result = model.transcribe(audio_path, language="ne", task="transcribe")
        hypothesis = normalize_nepali_text(result["text"])

        cer = compute_cer(reference, hypothesis)
        wer = compute_wer(reference, hypothesis)

        threshold_cat = CATEGORY_MAP.get(item["category"], "conversational")
        cer_thresholds = THRESHOLDS["cer"].get(threshold_cat, THRESHOLDS["cer"]["overall"])
        wer_thresholds = THRESHOLDS["wer"].get(threshold_cat, THRESHOLDS["wer"]["overall"])

        item_result = {
            "id": item["id"],
            "category": item["category"],
            "subcategory": item.get("subcategory", ""),
            "threshold_category": threshold_cat,
            "reference": reference,
            "hypothesis": hypothesis,
            "cer": round(cer, 4),
            "wer": round(wer, 4),
            "cer_judgment": judge(cer, cer_thresholds),
            "wer_judgment": judge(wer, wer_thresholds),
        }
        results.append(item_result)

        if (i + 1) % 25 == 0 or (i + 1) == len(items):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  [{i+1}/{len(items)}] {rate:.1f} items/s | last CER={cer:.3f} WER={wer:.3f}")

    # Aggregate by category
    aggregates = compute_aggregates(results)

    output = {
        "model": model_size,
        "total_items": len(results),
        "elapsed_seconds": round(time.time() - t0, 1),
        "aggregates": aggregates,
        "per_item": results,
    }

    out_path = output_dir / "asr_results.json"
    save_json(out_path, output)
    print(f"\nASR results saved to {out_path}")

    print_summary(aggregates)
    return output


def compute_aggregates(results: list[dict]) -> dict:
    """Compute aggregate CER/WER by category and overall."""
    agg = {}

    # Overall
    all_cer = [r["cer"] for r in results]
    all_wer = [r["wer"] for r in results]
    agg["overall"] = {
        "count": len(results),
        "cer_mean": round(float(np.mean(all_cer)), 4),
        "cer_median": round(float(np.median(all_cer)), 4),
        "cer_std": round(float(np.std(all_cer)), 4),
        "wer_mean": round(float(np.mean(all_wer)), 4),
        "wer_median": round(float(np.median(all_wer)), 4),
        "wer_std": round(float(np.std(all_wer)), 4),
        "cer_judgment": judge(float(np.mean(all_cer)), THRESHOLDS["cer"]["overall"]),
        "wer_judgment": judge(float(np.mean(all_wer)), THRESHOLDS["wer"]["overall"]),
        "cer_pass_rate": round(sum(1 for r in results if r["cer_judgment"] == "pass") / len(results), 4),
        "wer_pass_rate": round(sum(1 for r in results if r["wer_judgment"] == "pass") / len(results), 4),
    }

    # By item category
    categories = sorted(set(r["category"] for r in results))
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_cer = [r["cer"] for r in cat_results]
        cat_wer = [r["wer"] for r in cat_results]
        threshold_cat = CATEGORY_MAP.get(cat, "conversational")
        agg[cat] = {
            "count": len(cat_results),
            "threshold_category": threshold_cat,
            "cer_mean": round(float(np.mean(cat_cer)), 4),
            "cer_median": round(float(np.median(cat_cer)), 4),
            "wer_mean": round(float(np.mean(cat_wer)), 4),
            "wer_median": round(float(np.median(cat_wer)), 4),
            "cer_judgment": judge(float(np.mean(cat_cer)),
                                  THRESHOLDS["cer"].get(threshold_cat, THRESHOLDS["cer"]["overall"])),
            "wer_judgment": judge(float(np.mean(cat_wer)),
                                  THRESHOLDS["wer"].get(threshold_cat, THRESHOLDS["wer"]["overall"])),
            "cer_pass_rate": round(sum(1 for r in cat_results if r["cer_judgment"] == "pass") / len(cat_results), 4),
        }

    return agg


def print_summary(agg: dict):
    """Print a human-readable summary of ASR results."""
    print("\n=== ASR Round-Trip Results ===")
    print(f"{'Category':<30} {'Count':>5} {'CER':>7} {'WER':>7} {'CER Judge':>10} {'WER Judge':>10} {'Pass%':>6}")
    print("-" * 85)
    for cat in sorted(agg.keys()):
        a = agg[cat]
        print(f"{cat:<30} {a['count']:>5} {a['cer_mean']:>7.4f} {a['wer_mean']:>7.4f} "
              f"{a['cer_judgment']:>10} {a['wer_judgment']:>10} {a.get('cer_pass_rate', 0)*100:>5.1f}%")
    print()


def main():
    parser = argparse.ArgumentParser(description="ASR round-trip evaluation")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to manifest.json from generate_audio")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large", "large-v3"],
                        help="Whisper model size")
    args = parser.parse_args()

    run_asr(args.manifest, args.model)


if __name__ == "__main__":
    main()
