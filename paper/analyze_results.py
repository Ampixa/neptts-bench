#!/usr/bin/env python3
"""Generate all paper tables and analysis from benchmark results.

Outputs:
  - LaTeX tables for the paper
  - Cross-metric correlation matrix
  - Per-category phonological analysis
  - Summary statistics
"""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "data" / "eval_results"
SENTENCES_PATH = ROOT / "data" / "sentences_fixed.json"


def load_all_results():
    """Load all evaluation results into a unified structure."""
    results = {}

    # SCOREQ auto-MOS
    p = RESULTS_DIR / "auto_mos_scoreq.json"
    if p.exists():
        with open(p) as f:
            scoreq = json.load(f)
        for sys_name, data in scoreq.items():
            if sys_name not in results:
                results[sys_name] = {}
            results[sys_name]["scoreq_mos"] = data.get("avg_mos", 0)

    # Whisper small ASR
    p = RESULTS_DIR / "asr_roundtrip.json"
    if p.exists():
        with open(p) as f:
            asr = json.load(f)
        for sys_name, data in asr.items():
            if sys_name not in results:
                results[sys_name] = {}
            # Handle nested summary structure
            summary = data.get("summary", data)
            results[sys_name]["whisper_small_cer"] = summary.get("avg_cer", 0)
            results[sys_name]["whisper_small_wer"] = summary.get("avg_wer", 0)
            # Store per-category for analysis
            if "per_category" in summary:
                results[sys_name]["_whisper_categories"] = summary["per_category"]
            if "details" in data:
                results[sys_name]["_whisper_files"] = data["details"]

    # Whisper large-v3-turbo (MLX)
    p = RESULTS_DIR / "asr_roundtrip_mlx.json"
    if p.exists():
        with open(p) as f:
            mlx = json.load(f)
        for sys_name, data in mlx.items():
            if sys_name not in results:
                results[sys_name] = {}
            results[sys_name]["whisper_large_cer"] = data.get("avg_cer", 0)

    # MMS ASR
    p = RESULTS_DIR / "asr_results_multi.json"
    if p.exists():
        with open(p) as f:
            multi = json.load(f)
        for asr_name, sys_results in multi.items():
            for sys_name, data in sys_results.items():
                if sys_name not in results:
                    results[sys_name] = {}
                results[sys_name][f"{asr_name}_cer"] = data.get("avg_cer", 0)

    # Chirp2 ASR
    p = RESULTS_DIR / "asr_results_chirp2.json"
    if p.exists():
        with open(p) as f:
            chirp2 = json.load(f)
        for sys_name, data in chirp2.items():
            if sys_name not in results:
                results[sys_name] = {}
            results[sys_name]["chirp2_cer"] = data.get("avg_cer", 0)

    # Filter out systems with no real data
    results = {k: v for k, v in results.items()
               if any(val > 0 for key, val in v.items() if not key.startswith("_"))}

    return results


def load_sentences():
    with open(SENTENCES_PATH) as f:
        return json.load(f)


def print_main_table(results):
    """Print the main comparison table (Table 1 in paper)."""
    metrics = ["scoreq_mos", "chirp2_cer", "xlsr_nepali_cer", "mms_cer", "whisper_small_cer"]
    headers = ["System", "SCOREQ", "Chirp2", "XLS-R", "MMS", "Whisper-S"]

    print("\n" + "=" * 80)
    print("TABLE 1: System Comparison (Auto-MOS and ASR Round-Trip CER)")
    print("=" * 80)

    # Sort by Chirp2 CER (best ASR metric)
    sorted_systems = sorted(results.keys(),
                           key=lambda k: results[k].get("chirp2_cer", 99))

    print(f"{'System':<22}", end="")
    for h in headers[1:]:
        print(f"{h:>10}", end="")
    print()
    print("-" * 72)

    for sys_name in sorted_systems:
        r = results[sys_name]
        print(f"{sys_name:<22}", end="")
        for m in metrics:
            val = r.get(m, 0)
            if val > 0:
                if "mos" in m:
                    print(f"{val:>10.2f}", end="")
                else:
                    print(f"{val:>10.3f}", end="")
            else:
                print(f"{'—':>10}", end="")
        print()


def print_latex_table(results):
    """Generate LaTeX table for paper."""
    print("\n" + "=" * 80)
    print("LATEX TABLE")
    print("=" * 80)

    sorted_systems = sorted(results.keys(),
                           key=lambda k: results[k].get("chirp2_cer", 99))

    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Automated evaluation of 9 Nepali TTS systems. SCOREQ MOS ($\uparrow$) measures perceived quality; CER ($\downarrow$) measures intelligibility via ASR round-trip.}")
    print(r"\label{tab:main_results}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"System & SCOREQ & Chirp2 & MMS & XLS-R \\")
    print(r" & MOS $\uparrow$ & CER $\downarrow$ & CER $\downarrow$ & CER $\downarrow$ \\")
    print(r"\midrule")

    for sys_name in sorted_systems:
        r = results[sys_name]
        scoreq = f"{r.get('scoreq_mos', 0):.2f}" if r.get('scoreq_mos', 0) > 0 else "—"
        chirp2 = f"{r.get('chirp2_cer', 0):.3f}" if r.get('chirp2_cer', 0) > 0 else "—"
        mms = f"{r.get('mms_cer', 0):.3f}" if r.get('mms_cer', 0) > 0 else "—"
        xlsr = f"{r.get('xlsr_nepali_cer', 0):.3f}" if r.get('xlsr_nepali_cer', 0) > 0 else "—"

        # Bold best values
        display_name = sys_name.replace("_", r"\_")
        print(f"{display_name} & {scoreq} & {chirp2} & {mms} & {xlsr} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


def compute_correlations(results):
    """Compute Spearman rank correlations between all metric pairs."""
    print("\n" + "=" * 80)
    print("TABLE 2: Spearman Rank Correlations Between Metrics")
    print("=" * 80)

    metrics = ["scoreq_mos", "chirp2_cer", "mms_cer", "xlsr_nepali_cer", "whisper_small_cer"]
    labels = ["SCOREQ", "Chirp2", "MMS", "XLS-R", "Whisper-S"]

    # Get systems that have all metrics
    valid_systems = [s for s in results
                     if all(results[s].get(m, 0) > 0 for m in metrics)]

    if len(valid_systems) < 4:
        print(f"Only {len(valid_systems)} systems have all metrics. Need at least 4.")
        return

    vectors = {}
    for m in metrics:
        vectors[m] = [results[s][m] for s in valid_systems]
        # Flip SCOREQ sign so higher = better aligns with lower CER = better
        if "mos" in m:
            vectors[m] = [-v for v in vectors[m]]

    print(f"\nSystems used: {valid_systems}")
    print(f"\n{'':>12}", end="")
    for l in labels:
        print(f"{l:>10}", end="")
    print()

    for i, (m1, l1) in enumerate(zip(metrics, labels)):
        print(f"{l1:>12}", end="")
        for j, (m2, l2) in enumerate(zip(metrics, labels)):
            if i == j:
                print(f"{'1.000':>10}", end="")
            else:
                rho, pval = stats.spearmanr(vectors[m1], vectors[m2])
                sig = "*" if pval < 0.05 else " "
                print(f"{rho:>9.3f}{sig}", end="")
        print()

    print("\n* p < 0.05")


def per_category_analysis(results):
    """Analyze CER by sentence category for each system."""
    print("\n" + "=" * 80)
    print("TABLE 3: Per-Category CER Analysis (Whisper Small)")
    print("=" * 80)

    sentences = load_sentences()
    sent_cats = {s["sent_id"]: s.get("category", "unknown") for s in sentences
                 if not s["sent_id"].startswith("chirp")}

    # Get per-file CER from Whisper results
    asr_path = RESULTS_DIR / "asr_roundtrip.json"
    if not asr_path.exists():
        print("No ASR results file")
        return

    with open(asr_path) as f:
        asr = json.load(f)

    categories = sorted(set(sent_cats.values()) - {"natural_speech", "unknown", ""})

    print(f"\n{'System':<22}", end="")
    for cat in categories:
        short = cat[:8]
        print(f"{short:>10}", end="")
    print()
    print("-" * (22 + 10 * len(categories)))

    for sys_name in sorted(asr.keys()):
        files = asr[sys_name].get("files", [])
        if not files:
            continue

        cat_cers = defaultdict(list)
        for f in files:
            cat = sent_cats.get(f["sent_id"], "unknown")
            if cat in categories:
                cat_cers[cat].append(f["cer"])

        print(f"{sys_name:<22}", end="")
        for cat in categories:
            vals = cat_cers.get(cat, [])
            if vals:
                avg = sum(vals) / len(vals)
                print(f"{avg:>10.3f}", end="")
            else:
                print(f"{'—':>10}", end="")
        print()


def summary_stats(results):
    """Print summary statistics for the paper."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    n_systems = len(results)
    print(f"TTS systems evaluated: {n_systems}")

    # Count metrics
    all_metrics = set()
    for r in results.values():
        all_metrics.update(k for k in r.keys() if not k.startswith("_"))
    print(f"Evaluation metrics: {len(all_metrics)}")
    for m in sorted(all_metrics):
        vals = [r.get(m, 0) for r in results.values() if r.get(m, 0) > 0]
        if vals:
            print(f"  {m}: min={min(vals):.3f}, max={max(vals):.3f}, mean={sum(vals)/len(vals):.3f} ({len(vals)} systems)")

    sentences = load_sentences()
    bench = [s for s in sentences if not s["sent_id"].startswith("chirp")]
    chirp = [s for s in sentences if s["sent_id"].startswith("chirp")]
    print(f"\nBenchmark sentences: {len(bench)}")
    print(f"Natural speech (Chirp2): {len(chirp)}")

    # Category breakdown
    cats = defaultdict(int)
    for s in bench:
        cats[s.get("category", "unknown")] += 1
    print("\nSentence categories:")
    for cat, n in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {n}")

    # Minimal pairs
    pairs = set(s.get("pair_id", "") for s in bench if s.get("pair_id"))
    pair_sents = [s for s in bench if s.get("pair_id")]
    print(f"\nMinimal pairs: {len(pairs)} pairs ({len(pair_sents)} sentences)")


def main():
    results = load_all_results()

    if not results:
        print("No results found!")
        sys.exit(1)

    summary_stats(results)
    print_main_table(results)
    compute_correlations(results)
    per_category_analysis(results)
    print_latex_table(results)


if __name__ == "__main__":
    main()
