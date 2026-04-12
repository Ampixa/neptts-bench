"""Main entry point for NepTTS-Bench benchmark runner.

Orchestrates the full pipeline: generate audio → ASR round-trip →
phonological ABX → basic audio quality → write results.json

Usage:
    python run_benchmark.py --system edge-tts --voice ne-NP-HemkalaNeural
    python run_benchmark.py --system edge-tts --voice ne-NP-HemkalaNeural --skip-generate
"""

import argparse
import time
from pathlib import Path

import numpy as np

from utils import get_project_root, load_json, save_json, load_audio, compute_snr, detect_silence


def compute_audio_quality(manifest_path: Path) -> dict:
    """Compute audio quality metrics: duration, silence, SNR, and DNSMOS."""
    import librosa
    from speechmos import dnsmos

    manifest = load_json(manifest_path)
    output_dir = manifest_path.parent

    items = [m for m in manifest if m.get("audio_path") and m["status"] in ("ok", "cached")]
    print(f"\nComputing audio quality for {len(items)} items...")

    quality_results = []
    errors = 0

    for i, item in enumerate(items):
        try:
            audio, sr = load_audio(item["audio_path"])
            silence = detect_silence(audio, sr)
            snr = compute_snr(audio, sr)

            # DNSMOS (needs 16kHz)
            audio_16k, _ = librosa.load(item["audio_path"], sr=16000)
            mos = dnsmos.run(audio_16k, sr=16000)

            quality_results.append({
                "id": item["id"],
                "category": item["category"],
                "duration_ms": silence["total_duration_ms"],
                "leading_silence_ms": silence["leading_silence_ms"],
                "trailing_silence_ms": silence["trailing_silence_ms"],
                "snr_db": round(snr, 1),
                "dnsmos_ovrl": round(float(mos["ovrl_mos"]), 3),
                "dnsmos_sig": round(float(mos["sig_mos"]), 3),
                "dnsmos_bak": round(float(mos["bak_mos"]), 3),
                "dnsmos_p808": round(float(mos["p808_mos"]), 3),
                "sample_rate": sr,
                "samples": len(audio),
            })
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  Error processing {item['id']}: {e}")

        if (i + 1) % 50 == 0 or (i + 1) == len(items):
            print(f"  [{i+1}/{len(items)}] processed")

    # Aggregates
    durations = [r["duration_ms"] for r in quality_results]
    snrs = [r["snr_db"] for r in quality_results]
    ovrl = [r["dnsmos_ovrl"] for r in quality_results]
    sig = [r["dnsmos_sig"] for r in quality_results]
    bak = [r["dnsmos_bak"] for r in quality_results]
    p808 = [r["dnsmos_p808"] for r in quality_results]

    def stats(vals):
        return {
            "mean": round(float(np.mean(vals)), 3),
            "median": round(float(np.median(vals)), 3),
            "min": round(float(np.min(vals)), 3),
            "max": round(float(np.max(vals)), 3),
            "std": round(float(np.std(vals)), 3),
        }

    aggregates = {
        "count": len(quality_results),
        "errors": errors,
        "duration_ms": stats(durations),
        "snr_db": stats(snrs),
        "dnsmos_ovrl": stats(ovrl),
        "dnsmos_sig": stats(sig),
        "dnsmos_bak": stats(bak),
        "dnsmos_p808": stats(p808),
        "leading_silence_ms": {
            "mean": round(float(np.mean([r["leading_silence_ms"] for r in quality_results])), 1),
        },
        "trailing_silence_ms": {
            "mean": round(float(np.mean([r["trailing_silence_ms"] for r in quality_results])), 1),
        },
    }

    output = {
        "aggregates": aggregates,
        "per_item": quality_results,
    }

    out_path = output_dir / "quality_results.json"
    save_json(out_path, output)
    print(f"Audio quality results saved to {out_path}")

    print(f"\n=== Audio Quality Summary ===")
    d = aggregates["duration_ms"]
    print(f"Duration: mean={d['mean']:.0f}ms, median={d['median']:.0f}ms, "
          f"range=[{d['min']:.0f}, {d['max']:.0f}]ms")
    print(f"SNR: mean={aggregates['snr_db']['mean']:.1f}dB, "
          f"median={aggregates['snr_db']['median']:.1f}dB")
    print(f"DNSMOS OVRL: mean={aggregates['dnsmos_ovrl']['mean']:.3f}, "
          f"median={aggregates['dnsmos_ovrl']['median']:.3f}")
    print(f"DNSMOS SIG:  mean={aggregates['dnsmos_sig']['mean']:.3f}")
    print(f"DNSMOS BAK:  mean={aggregates['dnsmos_bak']['mean']:.3f}")
    print(f"DNSMOS P808: mean={aggregates['dnsmos_p808']['mean']:.3f}")
    print(f"Leading silence: mean={aggregates['leading_silence_ms']['mean']:.0f}ms")
    print(f"Trailing silence: mean={aggregates['trailing_silence_ms']['mean']:.0f}ms")

    return output


def compile_results(output_dir: Path, system: str, voice: str) -> dict:
    """Compile all results into a single results.json."""
    results = {
        "system": system,
        "voice": voice,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Load each result file if it exists
    for name in ("asr_results", "abx_results", "quality_results", "fsd_results"):
        path = output_dir / f"{name}.json"
        if path.exists():
            data = load_json(path)
            if name == "asr_results":
                results["asr_roundtrip"] = {
                    "model": data.get("model"),
                    "total_items": data.get("total_items"),
                    "elapsed_seconds": data.get("elapsed_seconds"),
                    "aggregates": data.get("aggregates"),
                }
            elif name == "abx_results":
                results["phonological_abx"] = {
                    "model": data.get("model"),
                    "total_pairs": data.get("total_pairs"),
                    "overall_accuracy": data.get("overall_accuracy"),
                    "per_category": data.get("per_category"),
                }
            elif name == "quality_results":
                results["audio_quality"] = data.get("aggregates")
            elif name == "fsd_results":
                results["frechet_speech_distance"] = {
                    "overall_fsd": data.get("overall_fsd"),
                    "cosine_similarity_means": data.get("cosine_similarity_means"),
                    "n_reference_files": data.get("n_reference_files"),
                    "per_category": data.get("per_category"),
                }

    # Load manifest for summary stats
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        ok = sum(1 for m in manifest if m["status"] in ("ok", "cached"))
        err = sum(1 for m in manifest if m["status"] not in ("ok", "cached"))
        results["generation"] = {
            "total_items": len(manifest),
            "ok": ok,
            "errors": err,
        }

    out_path = output_dir / "results.json"
    save_json(out_path, results)
    print(f"\nFinal results compiled to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="NepTTS-Bench runner")
    parser.add_argument("--system", default="edge-tts", help="TTS system name")
    parser.add_argument("--voice", default="ne-NP-HemkalaNeural", help="Voice name")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory (default: benchmark/results/{system}-{voice}/)")
    parser.add_argument("--whisper-model", default="medium",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size")
    parser.add_argument("--skip-generate", action="store_true",
                        help="Skip audio generation (use existing)")
    parser.add_argument("--skip-asr", action="store_true",
                        help="Skip ASR round-trip")
    parser.add_argument("--skip-abx", action="store_true",
                        help="Skip ABX discrimination")
    parser.add_argument("--skip-fsd", action="store_true",
                        help="Skip Frechet Speech Distance")
    parser.add_argument("--ref-stats", type=Path, default=None,
                        help="Path to FSD reference stats .npz (default: benchmark/runner/ref_stats_clsril23.npz)")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="TTS generation concurrency")
    args = parser.parse_args()

    root = get_project_root()

    # Determine output dir
    if args.output:
        output_dir = args.output
    else:
        voice_slug = args.voice.lower().replace(" ", "-")
        output_dir = root / "benchmark" / "results" / f"{args.system}-{voice_slug}"
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.json"
    t_total = time.time()

    # Step 1: Generate audio
    if not args.skip_generate:
        print("=" * 60)
        print("STEP 1: Audio Generation")
        print("=" * 60)
        from generate_audio import extract_test_items, generate_all
        import asyncio

        items = extract_test_items(root)
        print(f"Found {len(items)} test items")
        results = asyncio.run(generate_all(items, args.voice, output_dir, args.concurrency))
        save_json(manifest_path, results)

        ok = sum(1 for r in results if r["status"] in ("ok", "cached"))
        print(f"Generation complete: {ok}/{len(results)} ok")
    else:
        print("Skipping audio generation (--skip-generate)")
        if not manifest_path.exists():
            print(f"ERROR: No manifest found at {manifest_path}")
            return

    # Step 2: Audio quality (fast, no model needed)
    print("\n" + "=" * 60)
    print("STEP 2: Audio Quality Metrics")
    print("=" * 60)
    compute_audio_quality(manifest_path)

    # Step 3: ASR round-trip
    if not args.skip_asr:
        print("\n" + "=" * 60)
        print("STEP 3: ASR Round-Trip (Whisper)")
        print("=" * 60)
        from asr_roundtrip import run_asr
        run_asr(manifest_path, args.whisper_model)
    else:
        print("\nSkipping ASR round-trip (--skip-asr)")

    # Step 4: Phonological ABX
    if not args.skip_abx:
        print("\n" + "=" * 60)
        print("STEP 4: Phonological ABX Discrimination")
        print("=" * 60)
        from phonological_abx import run_abx
        run_abx(manifest_path, args.whisper_model)
    else:
        print("\nSkipping ABX discrimination (--skip-abx)")

    # Step 5: Frechet Speech Distance
    if not args.skip_fsd:
        ref_stats = args.ref_stats
        if ref_stats is None:
            ref_stats = root / "benchmark" / "runner" / "ref_stats_clsril23.npz"
        if ref_stats.exists():
            print("\n" + "=" * 60)
            print("STEP 5: Frechet Speech Distance (CLSRIL-23)")
            print("=" * 60)
            from frechet_speech_distance import score_tts
            score_tts(manifest_path, ref_stats)
        else:
            print(f"\nSkipping FSD: reference stats not found at {ref_stats}")
            print("  Build with: python frechet_speech_distance.py build-ref --audio-dir <natural-speech-dir> --output ref_stats_clsril23.npz")
    else:
        print("\nSkipping Frechet Speech Distance (--skip-fsd)")

    # Step 6: Compile results
    print("\n" + "=" * 60)
    print("STEP 6: Compile Results")
    print("=" * 60)
    results = compile_results(output_dir, args.system, args.voice)

    elapsed = time.time() - t_total
    print(f"\nTotal benchmark time: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Results directory: {output_dir}")


if __name__ == "__main__":
    main()
