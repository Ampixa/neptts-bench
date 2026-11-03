# Arena judge calibration and local rank sensitivity

Generated 2026-11-03T20:37:17.771422+00:00, Codex `/root` (generator authored by `/root/docs_reconcile`), job `arena_judges_v1`; owner Ampixa / Arbiter `nenglish-73`. Automatic measurements for review. **No public submission audio was accessed or rescored; private audio and D32 weights remain local.**

Across five local-system judges, legacy-policy Kendall τ-b spans 0.00–1.00. This measures instrument sensitivity on fixed saved outputs, not an overall TTS winner.

## Real speech

CER %, estimate [95% CI]. PV: 100 read recordings / four speakers. Fresh: 96 silver-transcribed `real_yt` recordings / twelve voices. All targets included. Corpus CER weights characters; sentence CER weights utterances. CIs: 10,000 paired speaker-cluster draws, seed 20261103, conditional on fixed texts/sessions.

| Judge | Set | Raw corpus | Raw sentence | Legacy corpus | Legacy sentence |
| --- | --- | --- | --- | --- | --- |
| LONG1C | PV | 5.43 [4.33, 6.09] | 5.67 [3.91, 7.48] | 3.82 [2.53, 4.62] | 3.78 [2.77, 4.68] |
| LONG1C | Fresh | 22.41 [17.25, 28.24] | 22.01 [17.12, 27.30] | 9.08 [6.62, 12.05] | 8.71 [6.51, 11.39] |
| D32 320 ms | PV | 13.73 [8.81, 17.11] | 12.18 [7.96, 16.62] | 11.80 [7.50, 15.00] | 10.36 [6.76, 14.40] |
| D32 320 ms | Fresh | 29.17 [23.99, 34.46] | 28.92 [23.75, 33.93] | 18.27 [14.82, 21.85] | 18.08 [14.49, 21.87] |
| D32 full | PV | 9.44 [6.19, 11.35] | 8.59 [6.61, 10.76] | 7.88 [4.71, 9.57] | 6.98 [4.51, 9.21] |
| D32 full | Fresh | 26.25 [21.39, 31.54] | 25.85 [21.27, 30.79] | 13.69 [10.90, 16.70] | 13.32 [10.69, 16.23] |
| MMS npi | PV | 7.62 [6.10, 8.17] | 7.32 [5.16, 8.53] | 7.62 [6.10, 8.17] | 7.32 [5.16, 8.53] |
| MMS npi | Fresh | 29.48 [25.36, 33.06] | 29.45 [25.09, 33.23] | 29.48 [25.36, 33.06] | 29.45 [25.09, 33.23] |
| Whisper small | PV | 47.90 [40.54, 61.85] | 49.71 [41.43, 57.65] | 47.90 [40.54, 61.85] | 49.71 [41.43, 57.65] |
| Whisper small | Fresh | 111.89 [102.37, 120.88] | 123.99 [107.10, 143.00] | 111.89 [102.37, 120.88] | 123.99 [107.10, 143.00] |
| Whisper v3 turbo | PV | 33.63 [29.60, 39.15] | 36.68 [30.34, 43.03] | 33.63 [29.60, 39.15] | 36.68 [30.34, 43.03] |
| Whisper v3 turbo | Fresh | 129.00 [91.92, 165.90] | 129.28 [90.08, 171.29] | 128.74 [91.89, 165.31] | 128.86 [89.96, 170.64] |

Raw uses frozen phase1b suite scoring, including spaces. Legacy adds exact `score2.py` conditional transliteration: convert Latin-containing hypotheses only when the reference has no ASCII Latin. This compatibility rule has no new native accepted-script validation.

### Current arena: Whisper small

`whisper-small-ne-048909d9dafd`: **uncalibrated for spontaneous Nepali**, with measured numbers. Arena normalization removes punctuation, maps digits to ASCII, lowercases, excludes spaces from character CER and does not transliterate loanwords. Its macro sentence metric is separate from phase1b.

| Set | Arena corpus CER | Arena sentence CER |
| --- | --- | --- |
| PV | 49.01 [40.35, 66.09] | 52.03 [41.47, 62.54] |
| Fresh | 115.45 [104.41, 125.88] | 126.49 [109.48, 144.33] |

Weights, package versions and CPU decoding options match the historical profile; v3 turbo is a distinct candidate and cannot calibrate small. The archived profile does not record FFmpeg, the host architecture or all transitive dependencies; matching its declared settings is not proof of cross-host bitwise transcript equality. The measured environment is recorded in [environment_small.json](calibration/environment_small.json).

D32 streaming uses the existing int8 c16/left-128 ONNX export; full context uses the strict-loaded FP32 PyTorch checkpoint with chunk/left context −1. Their difference includes context, precision and execution implementation, so it cannot isolate the causal effect of context alone.

## Controls and cold retests

Paired **corpus-CER Δpp**, control minus real [95% CI]; worse/n. Small uses arena raw, others legacy. Degradation and within-cohort text derangement exactly reuse phase1b; ASRs receive audio only.

| Judge | Set | Degraded Δ; worse/n | Shuffled Δ; worse/n |
| --- | --- | --- | --- |
| LONG1C | PV | +63.45 [+46.08, +79.50]; 96/100 | +126.05 [+84.04, +258.27]; 100/100 |
| LONG1C | Fresh | +63.51 [+54.22, +71.55]; 95/96 | +83.29 [+75.89, +92.47]; 96/96 |
| D32 320 ms | PV | +73.76 [+70.06, +80.65]; 100/100 | +115.82 [+79.15, +239.89]; 100/100 |
| D32 320 ms | Fresh | +64.05 [+59.32, +69.28]; 96/96 | +71.78 [+64.52, +80.42]; 96/96 |
| D32 full | PV | +63.00 [+56.89, +70.38]; 98/100 | +120.56 [+81.83, +247.85]; 100/100 |
| D32 full | Fresh | +61.43 [+55.58, +66.86]; 96/96 | +76.90 [+69.79, +85.09]; 96/96 |
| MMS npi | PV | +49.16 [+38.62, +64.08]; 100/100 | +122.23 [+80.46, +254.31]; 100/100 |
| MMS npi | Fresh | +44.32 [+37.33, +49.58]; 94/96 | +59.38 [+53.04, +67.05]; 96/96 |
| Whisper small | PV | +139.91 [+123.20, +181.57]; 99/100 | +93.14 [+26.77, +250.62]; 96/100 |
| Whisper small | Fresh | +47.95 [+33.22, +64.38]; 72/96 | +22.06 [+7.93, +38.45]; 76/96 |
| Whisper v3 turbo | PV | +234.33 [+149.82, +313.00]; 97/100 | +100.99 [+49.29, +250.43]; 99/100 |
| Whisper v3 turbo | Fresh | +122.25 [+83.57, +160.81]; 75/96 | +25.58 [+7.70, +42.09]; 80/96 |

≥90% worsening plus positive paired lower bounds is a coarse sanity screen, not subtle-TTS qualification. Corpus/sentence conclusions may differ. CER >100% is retained.

| Judge | Pass 1 changed | Pass 2 changed | Cold-to-cold changed |
| --- | --- | --- | --- |
| LONG1C | 0/32 | 0/32 | 0/32 |
| D32 320 ms | 0/32 | 0/32 | 0/32 |
| D32 full | 0/32 | 0/32 | 0/32 |
| MMS npi | 0/32 | 0/32 | 0/32 |
| Whisper small | 0/32 | 0/32 | 0/32 |
| Whisper v3 turbo | 0/32 | 0/32 | 0/32 |

One real/degraded input per voice: 32/392 unique inputs (8.16%), below the prospective 10% sampling gate. These recognizer retests do not measure TTS seed spread or new sessions.

## Five systems × 205 prompts

**FLEURS reference condition**, fixed saved takes. Vox1/Vox2: VoxCPM2 best8 run1/run2; Cosy: CosyVoice3 neutral; Stock: stock VoxCPM2. Best8 uses a LONG1C selector, making LONG1C dependent evidence for selected outputs. No synthesis/selection was rerun.

Legacy **mean sentence CER % [95% CI]**. Paired prompt bootstrap conditions on saved takes, reference voice and settings; it does not cover new voices/seeds.

| System | Small | MMS | LONG1C | D32s | D32f |
| --- | --- | --- | --- | --- | --- |
| Swar | 138.90 [71.33, 238.74] | 17.21 [14.04, 20.53] | 5.15 [3.72, 6.74] | 10.97 [9.00, 13.04] | 7.76 [6.30, 9.31] |
| Vox1 | 159.35 [85.69, 251.85] | 20.76 [17.68, 24.03] | 2.65 [1.98, 3.37] | 16.68 [14.09, 19.51] | 9.53 [7.89, 11.26] |
| Vox2 | 170.13 [93.78, 266.65] | 19.09 [15.98, 22.39] | 3.19 [2.37, 4.14] | 19.22 [16.14, 22.44] | 10.47 [8.54, 12.49] |
| Cosy | 129.63 [70.16, 208.15] | 17.97 [15.29, 20.84] | 6.83 [5.43, 8.35] | 17.25 [14.44, 20.36] | 9.93 [8.13, 11.83] |
| Stock | 184.04 [128.46, 257.38] | 50.48 [39.98, 63.93] | 32.09 [22.55, 44.19] | 46.89 [39.39, 55.83] | 33.72 [27.16, 41.54] |

Point order runs from lower to higher CER (`→`); no statistically distinct winner is implied.

| Judge | Suite raw order | Legacy order | Arena raw order |
| --- | --- | --- | --- |
| Small | Cosy → Swar → Vox1 → Vox2 → Stock | Cosy → Swar → Vox1 → Vox2 → Stock | Cosy → Swar → Vox1 → Vox2 → Stock |
| MMS | Swar → Cosy → Vox2 → Vox1 → Stock | Swar → Cosy → Vox2 → Vox1 → Stock | Swar → Cosy → Vox2 → Vox1 → Stock |
| LONG1C | Vox1 → Vox2 → Swar → Cosy → Stock | Vox1 → Vox2 → Swar → Cosy → Stock | Vox1 → Vox2 → Swar → Cosy → Stock |
| D32s | Swar → Vox1 → Cosy → Vox2 → Stock | Swar → Vox1 → Cosy → Vox2 → Stock | Swar → Vox1 → Cosy → Vox2 → Stock |
| D32f | Swar → Vox1 → Vox2 → Cosy → Stock | Swar → Vox1 → Cosy → Vox2 → Stock | Swar → Vox1 → Vox2 → Cosy → Stock |

Tie-aware Kendall τ-b [95% CI], 10,000 paired prompt draws, seed 20261104. These are percentile intervals, not simultaneous multiple-comparison bounds.

| Judge pair | Suite raw τ-b | Legacy τ-b | Arena raw τ-b |
| --- | --- | --- | --- |
| Small / MMS | 0.60 [-0.20, 1.00] | 0.60 [-0.20, 1.00] | 0.60 [0.00, 1.00] |
| Small / LONG1C | 0.00 [-0.60, 0.80] | 0.00 [-0.60, 0.80] | 0.00 [-0.60, 0.60] |
| Small / D32s | 0.60 [-0.20, 1.00] | 0.60 [-0.20, 1.00] | 0.60 [0.00, 1.00] |
| Small / D32f | 0.40 [-0.40, 0.80] | 0.60 [-0.20, 1.00] | 0.40 [-0.20, 1.00] |
| MMS / LONG1C | 0.00 [-0.20, 0.40] | 0.00 [-0.20, 0.40] | 0.00 [-0.20, 0.40] |
| MMS / D32s | 0.60 [0.20, 1.00] | 0.60 [0.20, 1.00] | 0.60 [0.20, 1.00] |
| MMS / D32f | 0.40 [0.00, 1.00] | 0.60 [0.20, 1.00] | 0.40 [0.20, 1.00] |
| LONG1C / D32s | 0.40 [0.00, 0.60] | 0.40 [0.00, 0.60] | 0.40 [0.00, 0.60] |
| LONG1C / D32f | 0.60 [0.00, 0.60] | 0.40 [0.00, 0.60] | 0.60 [0.00, 0.60] |
| D32s / D32f | 0.80 [0.40, 1.00] | 1.00 [0.40, 1.00] | 0.80 [0.40, 1.00] |

**Legacy paired system-minus-Swar mean CER Δpp [95% CI]**; negative means fewer ASR errors under that judge, not an isolated best-of-N effect or human preference.

| Judge | System − Swar | Legacy Δpp |
| --- | --- | --- |
| Small | Vox1 | +20.45 [-39.23, +94.59] |
| Small | Vox2 | +31.23 [-34.92, +109.82] |
| Small | Cosy | -9.27 [-58.20, +42.17] |
| Small | Stock | +45.13 [-17.25, +106.14] |
| MMS | Vox1 | +3.55 [-0.01, +7.07] |
| MMS | Vox2 | +1.87 [-0.94, +4.66] |
| MMS | Cosy | +0.76 [-2.57, +3.92] |
| MMS | Stock | +33.27 [+22.86, +46.35] |
| LONG1C | Vox1 | -2.50 [-4.10, -1.09] |
| LONG1C | Vox2 | -1.96 [-3.62, -0.45] |
| LONG1C | Cosy | +1.68 [-0.20, +3.50] |
| LONG1C | Stock | +26.94 [+17.30, +39.02] |
| D32s | Vox1 | +5.71 [+3.01, +8.54] |
| D32s | Vox2 | +8.25 [+4.95, +11.76] |
| D32s | Cosy | +6.29 [+3.24, +9.55] |
| D32s | Stock | +35.92 [+28.36, +44.75] |
| D32f | Vox1 | +1.77 [+0.14, +3.37] |
| D32f | Vox2 | +2.71 [+0.94, +4.53] |
| D32f | Cosy | +2.17 [+0.15, +4.15] |
| D32f | Stock | +25.96 [+19.41, +33.78] |

### Minimal-pair dependence sensitivity

The 205 prompts contain **133 groups: 72 complete minimal pairs and 61 singleton prompts**. Resampling these groups keeps paired prompts together and preserves utterance-weighted estimands. Point scores/ranks are unchanged. This checks one known text dependence structure; it is not a voice/seed CI or proof that all prompt dependence is captured.

Legacy **Vox1 minus Swar** mean CER Δpp [95% CI], comparing the 205-prompt bootstrap with the 133-group bootstrap:

| Judge | 205 prompts | 133 groups |
| --- | --- | --- |
| Small | +20.45 [-39.23, +94.59] | +20.45 [-43.04, +93.60] |
| MMS | +3.55 [-0.01, +7.07] | +3.55 [+0.18, +6.81] |
| LONG1C | -2.50 [-4.10, -1.09] | -2.50 [-4.44, -0.99] |
| D32s | +5.71 [+3.01, +8.54] | +5.71 [+2.97, +8.56] |
| D32f | +1.77 [+0.14, +3.37] | +1.77 [+0.10, +3.45] |

Legacy τ-b [95% CI] versus LONG1C, using the same two resampling units:

| Judge pair | 205 prompts | 133 groups |
| --- | --- | --- |
| Small / LONG1C | 0.00 [-0.60, 0.80] | 0.00 [-0.60, 0.80] |
| MMS / LONG1C | 0.00 [-0.20, 0.40] | 0.00 [-0.20, 0.40] |
| D32s / LONG1C | 0.40 [0.00, 0.60] | 0.40 [0.00, 0.60] |
| D32f / LONG1C | 0.40 [0.00, 0.60] | 0.40 [0.00, 0.60] |

[ranking/numbers.json](ranking/numbers.json) contains **all three policies’** system-pair and per-system judge-change paired CIs, corpus CIs, aggregates and τ-b/tie diagnostics. [numbers_pair_clusters.json](ranking/numbers_pair_clusters.json) contains the corresponding complete grouped sensitivity. [calibration/numbers.json](calibration/numbers.json) contains every raw/legacy/arena control sentence/corpus CI. Rasa is separate and never pooled as 410 independent prompts.

## Limits and evidence

Real ASR error is not a hard 10% discrimination floor and is not subtracted from synthetic CER. Controls do not establish native intelligibility, voice identity, style or overall quality. **No new human listening/MOS, transcription/accepted-script review or independent speaker/session qualification was performed.** Native-rater work remains separate.

[REVIEW_GATES.md](REVIEW_GATES.md) and [ranking/contamination.json](ranking/contamination.json) record finite corpus screens, known benchmark tuning, Fresh silver/admission bias and private-training unknowns. Text absence does not prove decontamination or new speakers; 205-track short-phrase overlap and reference-source exposure remain confounds.

**D32 is call-trained; a public 32M judge profile depends on the owner’s open publication decision.** No weights were uploaded. Other candidates await executable public worker/import support; only the existing small profile is enabled. Separate ranks and required take/N/selector/judge-family disclosures avoid silent mixing, but declarations are not execution attestation.

Calibration JSON records source/runtime/model hashes and verified input/normalizer pins. Every reused raw/legacy score and paired interval reproduces frozen phase1b. D32 strict loading and checkpoint/tokenizer hashes are recorded without private text or weights.

Reproduce after complete aggregates: `python3 benchmark/judge_calibration_v1/calibration/report.py`. Missing six-candidate coverage, five-judge primary results, paired CIs or finite values are fatal. [arena.yml](../../.github/workflows/arena.yml) specifies Python/Ruff/JS checks and receiver compilation; actual runs belong in the PR and verify software only. [Source-install evidence](../../neptts-eval/source_install_validation.json) is separate.

| Input | SHA256 |
| --- | --- |
| calibration/numbers.json | `bb3df479d3cf35f234115669f85648554bbd3788f06b2271b2b53e10275118a5` |
| ranking/numbers.json | `a3c0b3853a27dca24945f53708243e124dad3b326e2054acd59f9a4c73d086a5` |
| ranking/numbers_pair_clusters.json | `9bf37b7ff907ff280bdc6e98c9d651ef57ec3e8481a9ba0ea36be2df9224a1ce` |
| calibration/report.py | `e8fe3b7b06e928582080892552f22cfb7dd1e03ca07797e05db9fe31d8a85c23` |

[README.md](README.md) owns provenance, inventory, copy verification and retention. These conditional automatic measurements do not qualify a universal TTS leaderboard.
