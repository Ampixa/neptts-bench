"""Run Whisper large-v2 (faster-whisper, int8_float16) on all TTS-9 audio.

Closes nb-qvt: Rupak P0 ask. Whisper large-v3-turbo (MLX) hallucinates;
this gives us a stable large-class baseline.

Inputs:
  - Audio: /home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs/<system>/
  - Sentences: /home/cdjk/gt/bolne/crew/bolne/benchmark/data/hetzner_recordings/recordings.db

Output: benchmark/results/asr_roundtrip_large_v2.json
"""
import json
import re
import sqlite3
import time
import unicodedata
from pathlib import Path

from faster_whisper import WhisperModel

ROOT = Path(__file__).resolve().parent.parent
AUDIO_ROOT = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs")
SENT_DB = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/hetzner_recordings/recordings.db")
OUT = ROOT / "benchmark" / "results" / "asr_roundtrip_large_v2.json"

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


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'[।!?,;:\.\-\(\)\[\]"\'।॥]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def char_error_rate(ref: str, hyp: str) -> float:
    ref = list(ref.replace(' ', ''))
    hyp = list(hyp.replace(' ', ''))
    n, m = len(ref), len(hyp)
    if n == 0:
        return float(m > 0)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m] / max(n, 1)


def word_error_rate(ref: str, hyp: str) -> float:
    ref_w = ref.split()
    hyp_w = hyp.split()
    n, m = len(ref_w), len(hyp_w)
    if n == 0:
        return float(m > 0)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_w[i - 1] == hyp_w[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[n][m] / max(n, 1)


def load_sentences():
    db = sqlite3.connect(str(SENT_DB))
    c = db.cursor()
    c.execute("SELECT sent_id, text_dev, category, contrast_word FROM sentences")
    sents = {}
    for sid, text, cat, cw in c.fetchall():
        sents[sid] = {"text": text, "category": cat, "contrast_word": cw}
    db.close()
    return sents


def transcribe_one(model, audio_path: Path) -> str:
    segments, _ = model.transcribe(
        str(audio_path),
        language="ne",
        task="transcribe",
        beam_size=5,
        condition_on_previous_text=False,
    )
    return "".join(s.text for s in segments)


def main():
    print("Loading Whisper large-v2 (int8_float16, CUDA)...")
    model = WhisperModel("large-v2", device="cuda", compute_type="int8_float16")
    print("Loaded.")

    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentence references.")

    all_results = {}
    grand_t0 = time.time()

    for sys_name in SYSTEMS:
        d = AUDIO_ROOT / sys_name
        if not d.exists():
            print(f"SKIP {sys_name}: no dir")
            continue

        files = sorted(
            f for f in (list(d.glob("*.mp3")) + list(d.glob("*.wav")))
            if not f.name.startswith("._")
        )
        print(f"\n=== {sys_name} ({len(files)} files) ===")

        results = []
        t0 = time.time()
        for i, audio in enumerate(files):
            sent_id = audio.stem
            if sent_id not in sentences:
                continue
            ref = normalize_text(sentences[sent_id]["text"])
            try:
                hyp_raw = transcribe_one(model, audio)
            except Exception as e:
                print(f"  ERR {sent_id}: {e}")
                continue
            hyp = normalize_text(hyp_raw)
            cer = char_error_rate(ref, hyp)
            wer = word_error_rate(ref, hyp)
            results.append({
                "sent_id": sent_id,
                "ref": ref,
                "hyp": hyp,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "category": sentences[sent_id]["category"],
                "contrast_word": sentences[sent_id]["contrast_word"],
            })
            if (i + 1) % 25 == 0:
                avg = sum(r["cer"] for r in results) / len(results)
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(files) - i - 1) / rate
                print(f"  {i+1}/{len(files)}  avg CER={avg:.3f}  "
                      f"({rate:.1f} it/s, ETA {eta:.0f}s)")

        if not results:
            continue
        cers = [r["cer"] for r in results]
        wers = [r["wer"] for r in results]
        cers_sorted = sorted(cers)
        median_cer = cers_sorted[len(cers_sorted) // 2]
        by_cat = {}
        for r in results:
            by_cat.setdefault(r["category"] or "other", []).append(r["cer"])
        summary = {
            "system": sys_name,
            "num_files": len(results),
            "avg_cer": round(sum(cers) / len(cers), 4),
            "median_cer": round(median_cer, 4),
            "max_cer": round(max(cers), 4),
            "pct_cer_gt_1": round(100 * sum(1 for c in cers if c > 1.0) / len(cers), 2),
            "avg_wer": round(sum(wers) / len(wers), 4),
            "per_category": {k: round(sum(v) / len(v), 4) for k, v in sorted(by_cat.items())},
            "worst_5": sorted(results, key=lambda x: -x["cer"])[:5],
        }
        all_results[sys_name] = {"summary": summary, "details": results}
        print(f"  -> avg CER={summary['avg_cer']:.3f}  med={summary['median_cer']:.3f}  "
              f"pct>1.0={summary['pct_cer_gt_1']}%  "
              f"({(time.time()-t0):.0f}s, {len(results)} files)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f"\nTotal time: {(time.time()-grand_t0)/60:.1f} min")
    print(f"Wrote: {OUT}")

    # Comparison summary
    print("\n=== Whisper large-v2 summary ===")
    print(f"{'System':<28} {'n':>5} {'avg CER':>9} {'med CER':>9} {'>1.0':>6}")
    for s in SYSTEMS:
        if s in all_results:
            S = all_results[s]["summary"]
            print(f"{s:<28} {S['num_files']:>5} {S['avg_cer']:>9.3f} "
                  f"{S['median_cer']:>9.3f} {S['pct_cer_gt_1']:>5.1f}%")


if __name__ == "__main__":
    main()
