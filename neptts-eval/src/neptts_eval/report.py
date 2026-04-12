"""Report generation with baseline comparison."""

import json
import sys
from datetime import datetime
from pathlib import Path


BASELINES_PATH = Path(__file__).parent / "baselines.json"


def load_baselines() -> dict:
    with open(BASELINES_PATH) as f:
        return json.load(f)


def generate_report(
    scoreq_results: dict | None,
    asr_results: dict | None,
    n_files: int,
    system_name: str = "user_system",
) -> dict:
    """Generate evaluation report with baseline comparison."""
    baselines = load_baselines()

    report = {
        "system": system_name,
        "benchmark_version": baselines.get("version", "1.0"),
        "eval_date": datetime.utcnow().isoformat()[:10],
        "n_files_evaluated": n_files,
    }

    if scoreq_results:
        report["scoreq"] = {
            "avg_mos": scoreq_results["avg_mos"],
            "n_scored": scoreq_results["n_scored"],
        }

    if asr_results:
        report["asr_roundtrip"] = {
            "avg_cer": asr_results["avg_cer"],
            "avg_wer": asr_results["avg_wer"],
            "n_files": asr_results["n_files"],
            "per_category": asr_results.get("per_category", {}),
        }

    # Rank against baselines
    comparison = []
    for sys_name, scores in baselines.get("systems", {}).items():
        entry = {"system": sys_name}
        entry.update(scores)
        comparison.append(entry)

    # Add user system
    user_entry = {"system": system_name}
    if scoreq_results:
        user_entry["scoreq_mos"] = scoreq_results["avg_mos"]
    if asr_results:
        user_entry["whisper_small_cer"] = asr_results["avg_cer"]
    comparison.append(user_entry)

    report["comparison"] = comparison
    return report


def print_table(report: dict):
    """Pretty-print comparison table to stdout."""
    comparison = report.get("comparison", [])
    user_sys = report.get("system", "user_system")

    # Sort by SCOREQ MOS descending
    comparison.sort(key=lambda x: x.get("scoreq_mos", 0), reverse=True)

    print()
    print("=" * 65)
    print("NepTTS-Bench Evaluation Report")
    print("=" * 65)

    if "scoreq" in report:
        print(f"  SCOREQ MOS: {report['scoreq']['avg_mos']:.2f}")
    if "asr_roundtrip" in report:
        print(f"  Whisper CER: {report['asr_roundtrip']['avg_cer']:.3f}")
    print(f"  Files evaluated: {report['n_files_evaluated']}")
    print()

    print(f"{'Rank':<5} {'System':<25} {'SCOREQ':>8} {'Chirp2':>8} {'MMS':>8} {'Whisper':>8}")
    print("-" * 65)

    for i, entry in enumerate(comparison, 1):
        name = entry["system"]
        marker = " <<" if name == user_sys else ""
        scoreq = f"{entry['scoreq_mos']:.2f}" if "scoreq_mos" in entry else "—"
        chirp2 = f"{entry['chirp2_cer']:.3f}" if "chirp2_cer" in entry else "—"
        mms = f"{entry['mms_cer']:.3f}" if "mms_cer" in entry else "—"
        whisper = f"{entry['whisper_small_cer']:.3f}" if "whisper_small_cer" in entry else "—"
        print(f"{i:<5} {name:<25} {scoreq:>8} {chirp2:>8} {mms:>8} {whisper:>8}{marker}")

    print()

    # Per-category breakdown
    if "asr_roundtrip" in report and report["asr_roundtrip"].get("per_category"):
        print("Per-category CER:")
        for cat, cer in sorted(report["asr_roundtrip"]["per_category"].items()):
            print(f"  {cat:<35} {cer:.3f}")
        print()
