"""Compute statistical rigor numbers for the NepTTS-Bench paper.

Inputs:
  - benchmark/data/ratings_prod.db (production rating DB snapshot, native raters only)
  - benchmark/results/auto_mos_scoreq.json
  - benchmark/results/asr_results_chirp2.json
  - benchmark/results/asr_results_multi.json (mms + xlsr_nepali)
  - benchmark/results/asr_roundtrip.json (whisper-small)

Outputs (printed):
  1. Per-system human MOS with cluster-bootstrap 95% CIs (clustered by rater)
  2. Krippendorff's alpha (interval) across raters
  3. Spearman rho for each automated metric vs human MOS
  4. Steiger's Z test for whether rho(Chirp2 vs Human) significantly differs from rho(Whisper-small vs Human)
  5. Benjamini-Hochberg-corrected p-values for the cross-metric correlation table
"""
import json
import math
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import krippendorff
from scipy.stats import spearmanr, false_discovery_control

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "benchmark" / "data" / "ratings_prod.db"
RESULTS = ROOT / "benchmark" / "results"

# ---------- 1. Load native-rater ratings ----------
conn = sqlite3.connect(DB)
c = conn.cursor()
c.execute("""SELECT r.rater_id, r.system_name, r.sent_id, r.score
             FROM ratings r JOIN raters rr ON r.rater_id = rr.id
             WHERE rr.native_speaker = 1""")
rows = c.fetchall()
print(f"Loaded {len(rows)} native-rater ratings.")

# ---------- 2. Per-system MOS + cluster bootstrap CI (cluster by rater) ----------
by_system = defaultdict(list)         # system -> list of scores
rater_by_system = defaultdict(lambda: defaultdict(list))  # system -> rater -> list of scores
for rater, system, sent, score in rows:
    by_system[system].append(score)
    rater_by_system[system][rater].append(score)

def cluster_bootstrap_ci(rater_scores, n_resamples=2000, seed=42):
    """Cluster-bootstrap 95% CI on the mean: resample raters with replacement."""
    rng = np.random.default_rng(seed)
    raters = list(rater_scores.keys())
    means = []
    for _ in range(n_resamples):
        pick = rng.choice(len(raters), size=len(raters), replace=True)
        all_scores = []
        for i in pick:
            all_scores.extend(rater_scores[raters[i]])
        means.append(np.mean(all_scores))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)

print()
print(f"{'System':<28} {'n':>6} {'mean':>7} {'95% CI (cluster bootstrap by rater)':>40}")
print("-" * 90)
human_mos = {}
for sys in sorted(by_system, key=lambda s: -np.mean(by_system[s])):
    scores = by_system[sys]
    mean = float(np.mean(scores))
    lo, hi = cluster_bootstrap_ci(rater_by_system[sys])
    human_mos[sys] = mean
    half_width = (hi - lo) / 2
    print(f"{sys:<28} {len(scores):>6} {mean:>7.3f}   [{lo:.3f}, {hi:.3f}]  (±{half_width:.3f})")

# ---------- 3. Krippendorff's alpha ----------
# Build (rater x sample) reliability matrix where sample = (system, sent)
rater_index = {}
sample_index = {}
matrix_dict = defaultdict(dict)
for rater, system, sent, score in rows:
    if rater not in rater_index:
        rater_index[rater] = len(rater_index)
    sample_key = (system, sent)
    if sample_key not in sample_index:
        sample_index[sample_key] = len(sample_index)
    matrix_dict[rater_index[rater]][sample_index[sample_key]] = score

n_raters = len(rater_index)
n_samples = len(sample_index)
matrix = np.full((n_raters, n_samples), np.nan)
for ri, sample_dict in matrix_dict.items():
    for si, score in sample_dict.items():
        matrix[ri, si] = score

print()
print(f"Reliability matrix: {n_raters} raters x {n_samples} samples")
filled = np.sum(~np.isnan(matrix))
print(f"Filled cells: {filled} / {n_raters * n_samples} ({100*filled/(n_raters*n_samples):.1f}%)")

alpha_interval = krippendorff.alpha(reliability_data=matrix, level_of_measurement="interval")
alpha_ordinal = krippendorff.alpha(reliability_data=matrix, level_of_measurement="ordinal")
print(f"Krippendorff alpha (interval): {alpha_interval:.4f}")
print(f"Krippendorff alpha (ordinal):  {alpha_ordinal:.4f}")

# ---------- 4. Spearman rho per automated metric vs human MOS (TTS-9 only) ----------
def load_summary(path):
    with open(path) as f: d = json.load(f)
    out = {}
    for sys, val in d.items():
        if 'summary' in val and 'avg_cer' in val.get('summary', {}):
            out[sys] = val['summary']['avg_cer']
        elif 'avg_cer' in val:
            out[sys] = val['avg_cer']
        elif 'avg_mos' in val:
            out[sys] = val['avg_mos']
        elif 'files' in val:
            cers = [f['cer'] for f in val['files'] if f.get('cer') is not None]
            out[sys] = float(np.mean(cers)) if cers else None
    return out

scoreq = load_summary(RESULTS / "auto_mos_scoreq.json")
chirp2 = load_summary(RESULTS / "asr_results_chirp2.json")
multi = json.load(open(RESULTS / "asr_results_multi.json"))
mms = {k: v["avg_cer"] for k, v in multi["mms"].items()}
xlsr = {k: v["avg_cer"] for k, v in multi["xlsr_nepali"].items()}
whisper_small = load_summary(RESULTS / "asr_roundtrip.json")

# Map TTS audio dir name to human MOS system_name
DIR_TO_SYS = {
    "edge_tts/hemkala": "edge_tts/hemkala",
    "edge_tts/sagar": "edge_tts/sagar",
    "gtts": "gtts",
    "gemini": "gemini",
    "piper": "piper",
    "tingting_asmita": "tingting_asmita",
    "tingting_sambriddhi": "tingting_sambriddhi",
    "tingting_subina": "tingting_subina",
    "elevenlabs": "elevenlabs",
}

# Build aligned vectors over TTS-9 systems
tts_systems = list(DIR_TO_SYS.keys())
human_v = []
metric_vecs = {"SCOREQ": [], "Chirp2": [], "MMS": [], "XLS-R": [], "Whisper-small": []}
for s in tts_systems:
    human_v.append(human_mos[DIR_TO_SYS[s]])
    metric_vecs["SCOREQ"].append(scoreq.get(s) if scoreq.get(s) is not None else scoreq.get("human"))
    metric_vecs["Chirp2"].append(chirp2.get(s))
    metric_vecs["MMS"].append(mms.get(s))
    metric_vecs["XLS-R"].append(xlsr.get(s))
    metric_vecs["Whisper-small"].append(whisper_small.get(s))

print()
print("Per-metric Spearman rho vs human MOS (n=9 TTS systems):")
metric_results = {}
for name, vec in metric_vecs.items():
    if any(v is None for v in vec):
        print(f"  {name}: missing data, skipping")
        continue
    # CER metrics: lower = better, so invert sign for rho with human MOS (higher = better)
    if name == "SCOREQ":
        rho, p = spearmanr(human_v, vec)
    else:
        # Higher CER = worse = lower MOS expected, so we expect negative rho with raw CER.
        # Negate CER so that the rho matches the conventional "agreement with human ranking".
        rho, p = spearmanr(human_v, [-v for v in vec])
    metric_results[name] = (rho, p, vec)
    print(f"  {name:<14} rho = {rho:+.4f}   p = {p:.4f}   n = {len(vec)}")

# ---------- 5. Steiger's Z for rho(Chirp2) vs rho(Whisper-small) ----------
def steiger_z(r12, r13, r23, n):
    """Steiger's Z for difference between two dependent correlations sharing one variable."""
    R = (1 - r12**2 - r13**2 - r23**2) + 2*r12*r13*r23
    rho_bar = (r12 + r13) / 2
    f = (1 - r23) / (2*(1 - rho_bar**2))
    h = (1 - f * rho_bar**2) / (1 - rho_bar**2)
    z = (math.atanh(r12) - math.atanh(r13)) * math.sqrt((n-3) / (2*(1-r23)*h))
    return z

# r12 = rho(human, Chirp2), r13 = rho(human, Whisper-small), r23 = rho(Chirp2, Whisper-small)
chirp_vec = [-v for v in metric_vecs["Chirp2"]]
whisper_vec = [-v for v in metric_vecs["Whisper-small"]]
r12 = spearmanr(human_v, chirp_vec)[0]
r13 = spearmanr(human_v, whisper_vec)[0]
r23 = spearmanr(chirp_vec, whisper_vec)[0]

print()
print("Steiger's Z (Chirp2 vs Whisper-small, both vs human MOS):")
print(f"  r(human, Chirp2)        = {r12:+.4f}")
print(f"  r(human, Whisper-small) = {r13:+.4f}")
print(f"  r(Chirp2, Whisper)      = {r23:+.4f}")
z = steiger_z(r12, r13, r23, n=9)
from scipy.stats import norm
p_two = 2 * (1 - norm.cdf(abs(z)))
print(f"  Steiger's Z = {z:+.4f}, two-sided p = {p_two:.4f}")

# ---------- 6. BH correction across the cross-metric correlation panel ----------
print()
print("Cross-metric correlation table (TTS-9):")
print("  SCOREQ orientation: higher = better; CER metrics negated so higher = better.")
metric_names = ["SCOREQ", "Chirp2", "MMS", "XLS-R", "Whisper-small"]
mvecs = {}
for n in metric_names:
    if not metric_vecs.get(n): continue
    if any(v is None for v in metric_vecs[n]): continue
    if n == "SCOREQ":
        mvecs[n] = list(metric_vecs[n])
    else:
        mvecs[n] = [-v for v in metric_vecs[n]]

# Also include human as a metric column
mvecs["Human"] = list(human_v)

names = list(mvecs.keys())
print(f"{'pair':<35} {'rho':>8} {'raw p':>10} {'BH p':>10}")
pvals = []
labels = []
rhos = []
for i, a in enumerate(names):
    for b in names[i+1:]:
        rho, p = spearmanr(mvecs[a], mvecs[b])
        if not (np.isnan(p) or p < 0 or p > 1):
            labels.append(f"{a} - {b}")
            pvals.append(p)
            rhos.append(rho)

if pvals:
    adj = false_discovery_control(pvals, method="bh")
    for label, rho, p, ap in zip(labels, rhos, pvals, adj):
        print(f"{label:<35} {rho:+.4f}  {p:.6f}  {ap:.6f}")
