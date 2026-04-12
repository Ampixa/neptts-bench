#!/usr/bin/env python3
"""ASR round-trip evaluation: TTS audio → Whisper transcription → CER/WER.

Measures intelligibility by checking if synthesized speech can be correctly
transcribed back to the original text.

Usage:
    python scripts/eval_asr_roundtrip.py
"""

import json
import os
import re
import sqlite3
import unicodedata
from pathlib import Path

import whisper

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "benchmark" / "data" / "hetzner_recordings" / "recordings.db"
TTS_BASE = ROOT / "benchmark" / "data" / "tts_outputs"
OUTPUT = ROOT / "benchmark" / "data" / "eval_results"


def normalize_text(text):
    """Normalize text for comparison: NFC, strip punctuation, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    # Remove common punctuation
    text = re.sub(r'[।!?,;:\.\-\(\)\[\]"\'।॥]', '', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def char_error_rate(ref, hyp):
    """Compute character error rate using edit distance."""
    ref = list(ref.replace(' ', ''))
    hyp = list(hyp.replace(' ', ''))
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def word_error_rate(ref, hyp):
    """Compute word error rate."""
    ref = ref.split()
    hyp = hyp.split()
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def load_sentences():
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    c.execute("SELECT sent_id, text_dev, category, contrast_word FROM sentences WHERE CAST(REPLACE(sent_id, 'sent_', '') AS INTEGER) <= 174")
    sents = {}
    for row in c.fetchall():
        sents[row[0]] = {"text": row[1], "category": row[2], "contrast_word": row[3]}
    db.close()
    return sents


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentences")

    # large-v3 needs >4GB VRAM; medium fits on GTX 1650 (4GB)
    whisper_model = os.environ.get("WHISPER_MODEL", "small")
    print(f"Loading Whisper {whisper_model}...")
    # GTX 1650 needs fp32 (fp16 produces NaN)
    model = whisper.load_model(whisper_model, device="cpu")
    model = model.to("cuda").float()
    print(f"  Loaded on GPU (fp32)")

    systems = [
        ("edge_tts/hemkala", "*.mp3"),
        ("edge_tts/sagar", "*.mp3"),
        ("gtts", "*.mp3"),
        ("gemini", "*.wav"),
        ("piper", "*.wav"),
        ("tingting_asmita", "*.mp3"),
        ("tingting_sambriddhi", "*.mp3"),
        ("tingting_subina", "*.mp3"),
        ("elevenlabs", "*.mp3"),
    ]

    all_results = {}

    for system_name, glob_pattern in systems:
        system_dir = TTS_BASE / system_name
        if not system_dir.exists():
            print(f"SKIP {system_name}: not found")
            continue

        print(f"\n=== {system_name} ===")
        files = sorted(system_dir.glob(glob_pattern))
        results = []

        for i, audio_file in enumerate(files):
            sent_id = audio_file.stem
            if sent_id not in sentences:
                continue

            ref_text = normalize_text(sentences[sent_id]["text"])

            try:
                result = model.transcribe(
                    str(audio_file),
                    language="ne",
                    task="transcribe",
                    fp16=False,
                )
                hyp_text = normalize_text(result["text"])
            except Exception as e:
                print(f"  ERROR {sent_id}: {e}")
                continue

            cer = char_error_rate(ref_text, hyp_text)
            wer = word_error_rate(ref_text, hyp_text)

            results.append({
                "sent_id": sent_id,
                "ref": ref_text,
                "hyp": hyp_text,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "category": sentences[sent_id]["category"],
                "contrast_word": sentences[sent_id]["contrast_word"],
            })

            if (i + 1) % 20 == 0:
                avg_cer = sum(r["cer"] for r in results) / len(results)
                print(f"  {i+1}/{len(files)} — running avg CER: {avg_cer:.3f}")

        # Compute aggregates
        if results:
            avg_cer = sum(r["cer"] for r in results) / len(results)
            avg_wer = sum(r["wer"] for r in results) / len(results)
            median_cer = sorted(r["cer"] for r in results)[len(results) // 2]

            # Per-category breakdown
            by_cat = {}
            for r in results:
                cat = r["category"] or "other"
                if cat not in by_cat:
                    by_cat[cat] = []
                by_cat[cat].append(r["cer"])

            summary = {
                "system": system_name,
                "num_files": len(results),
                "avg_cer": round(avg_cer, 4),
                "median_cer": round(median_cer, 4),
                "avg_wer": round(avg_wer, 4),
                "per_category": {
                    cat: round(sum(cers) / len(cers), 4)
                    for cat, cers in sorted(by_cat.items())
                },
                "worst_5": sorted(results, key=lambda x: -x["cer"])[:5],
            }

            all_results[system_name] = {"summary": summary, "details": results}

            print(f"\n  Results: {len(results)} files")
            print(f"  Avg CER: {avg_cer:.3f}  Median CER: {median_cer:.3f}  Avg WER: {avg_wer:.3f}")
            print(f"  Per category:")
            for cat, avg in sorted(summary["per_category"].items()):
                print(f"    {cat}: CER {avg:.3f}")

    # Save results
    output_path = OUTPUT / "asr_roundtrip.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {output_path}")

    # Print comparison table
    print(f"\n{'='*60}")
    print(f"ASR ROUND-TRIP COMPARISON")
    print(f"{'='*60}")
    print(f"{'System':<25} {'Files':>6} {'CER':>8} {'WER':>8}")
    print("-" * 50)
    for sys_name in ["edge_tts/hemkala", "edge_tts/sagar", "gtts", "gemini"]:
        if sys_name in all_results:
            s = all_results[sys_name]["summary"]
            print(f"{sys_name:<25} {s['num_files']:>6} {s['avg_cer']:>8.3f} {s['avg_wer']:>8.3f}")


if __name__ == "__main__":
    main()
