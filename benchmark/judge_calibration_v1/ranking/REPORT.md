# Judge choice changes the local 205-prompt ranking

Generated 2026-11-03 20:00 UTC by Codex `/root` using a report generator prepared by `/root/rank_assets`, job `arena_judges_v1`. Owner: Ampixa / Arbiter nenglish-73. Reference evidence for PR review; next review 2026-11-11. No audio or weights are included. NOT on HF.

These are five saved local systems on the same **205 arena prompt IDs/texts**, under the head-to-head FLEURS-reference generation condition. The five approved public arena reports were not rescored or accessed. Rasa-reference outputs were inventoried but not rescored; the two conditions are not pooled as 410 independent prompts.

The main comparison applies the frozen phase1b legacy loanword rule and suite normalizer to every judge. Both corpus CER and mean sentence CER are percentages. Lower is better. The unchanged arena normalization is separately available under `arena_raw`; it removes spaces from character counting and does not transliterate Latin tokens. Consequently this table uses the arena Whisper decoding instrument with an explicitly different, common comparison policy.

## Mean sentence CER with paired-prompt 95% intervals

| System | Whisper-small | MMS npi | LONG1C | D32 320 ms | D32 full context |
|---|---|---|---|---|---|
| Swar | 138.90 [71.33, 238.74] | 17.21 [14.04, 20.53] | 5.15 [3.72, 6.74] | 10.97 [9.00, 13.04] | 7.76 [6.30, 9.31] |
| VoxCPM2 run1 | 159.35 [85.69, 251.85] | 20.76 [17.68, 24.03] | 2.65 [1.98, 3.37] | 16.68 [14.09, 19.51] | 9.53 [7.89, 11.26] |
| VoxCPM2 run2 | 170.13 [93.78, 266.65] | 19.09 [15.98, 22.39] | 3.19 [2.37, 4.14] | 19.22 [16.14, 22.44] | 10.47 [8.54, 12.49] |
| CosyVoice3 neutral | 129.63 [70.16, 208.15] | 17.97 [15.29, 20.84] | 6.83 [5.43, 8.35] | 17.25 [14.44, 20.36] | 9.93 [8.13, 11.83] |
| Stock VoxCPM2 | 184.04 [128.46, 257.38] | 50.48 [39.98, 63.93] | 32.09 [22.55, 44.19] | 46.89 [39.39, 55.83] | 33.72 [27.16, 41.54] |

## Corpus CER with paired-prompt 95% intervals

| System | Whisper-small | MMS npi | LONG1C | D32 320 ms | D32 full context |
|---|---|---|---|---|---|
| Swar | 92.15 [66.08, 122.10] | 13.31 [10.87, 16.14] | 4.36 [3.31, 5.58] | 10.54 [8.55, 12.68] | 7.47 [6.09, 8.97] |
| VoxCPM2 run1 | 93.25 [68.40, 121.79] | 18.39 [15.51, 21.66] | 2.88 [2.10, 3.74] | 17.07 [14.33, 20.00] | 9.69 [8.00, 11.50] |
| VoxCPM2 run2 | 99.40 [73.11, 129.21] | 15.81 [13.27, 18.60] | 3.57 [2.58, 4.70] | 17.73 [15.07, 20.58] | 11.28 [8.87, 13.87] |
| CosyVoice3 neutral | 76.56 [56.45, 100.47] | 16.28 [13.86, 18.97] | 6.62 [5.08, 8.31] | 17.07 [14.36, 19.93] | 10.46 [8.42, 12.71] |
| Stock VoxCPM2 | 155.12 [118.86, 195.83] | 42.38 [33.44, 53.62] | 27.20 [19.40, 37.43] | 42.05 [35.43, 50.17] | 30.33 [24.61, 37.39] |

## Rank sensitivity

Kendall tau-b compares the five system means. Intervals resample the same prompts or groups jointly across every system and judge; they are conditional rank uncertainty on this panel, not population uncertainty over all TTS systems.

| Judges | Tau-b | 205-prompt 95% CI | 133-group 95% CI |
|---|---:|---|---|
| Whisper-small / MMS npi | 0.60 | [-0.20, 1.00] | [-0.20, 1.00] |
| Whisper-small / LONG1C | 0.00 | [-0.60, 0.80] | [-0.60, 0.80] |
| Whisper-small / D32 320 ms | 0.60 | [-0.20, 1.00] | [-0.20, 1.00] |
| Whisper-small / D32 full context | 0.60 | [-0.20, 1.00] | [-0.20, 1.00] |
| MMS npi / LONG1C | 0.00 | [-0.20, 0.40] | [-0.20, 0.40] |
| MMS npi / D32 320 ms | 0.60 | [0.20, 1.00] | [0.20, 1.00] |
| MMS npi / D32 full context | 0.60 | [0.20, 1.00] | [0.20, 1.00] |
| LONG1C / D32 320 ms | 0.40 | [0.00, 0.60] | [0.00, 0.60] |
| LONG1C / D32 full context | 0.40 | [0.00, 0.60] | [0.00, 0.60] |
| D32 320 ms / D32 full context | 1.00 | [0.40, 1.00] | [0.40, 1.00] |

## Paired system contrasts

**Difference = Swar minus the named VoxCPM2 run**, in sentence-CER percentage points. Positive favors that VoxCPM2 run; negative favors Swar. All 50 system contrasts, including every other system pair, are retained per normalization in the two JSON artifacts. These are descriptive intervals without a multiple-comparison correction.

| Judge | Comparator | Difference | 205-prompt 95% CI | 133-group 95% CI |
|---|---|---:|---|---|
| Whisper-small | VoxCPM2 run1 | -20.45 | [-94.59, 39.23] | [-93.60, 43.04] |
| Whisper-small | VoxCPM2 run2 | -31.23 | [-109.82, 34.92] | [-109.60, 37.41] |
| MMS npi | VoxCPM2 run1 | -3.55 | [-7.07, 0.01] | [-6.81, -0.18] |
| MMS npi | VoxCPM2 run2 | -1.87 | [-4.66, 0.94] | [-4.74, 0.97] |
| LONG1C | VoxCPM2 run1 | +2.50 | [1.09, 4.10] | [0.99, 4.44] |
| LONG1C | VoxCPM2 run2 | +1.96 | [0.45, 3.62] | [0.33, 3.90] |
| D32 320 ms | VoxCPM2 run1 | -5.71 | [-8.54, -3.01] | [-8.56, -2.97] |
| D32 320 ms | VoxCPM2 run2 | -8.25 | [-11.76, -4.95] | [-11.63, -5.08] |
| D32 full context | VoxCPM2 run1 | -1.77 | [-3.37, -0.14] | [-3.45, -0.10] |
| D32 full context | VoxCPM2 run2 | -2.71 | [-4.53, -0.94] | [-4.74, -0.79] |

## Interpretation and limits

- Both fine-tuned VoxCPM2 runs used eight candidates and chose the lowest **LONG1C streaming c16 CER against the target text**, with lower-seed tie breaking. Their frozen receipts explicitly say no MMS or ECAPA selector. LONG1C therefore evaluates outputs selected by that same judge family; its apparent advantage is optimistic by construction. The cross-judge reversal is evidence for separate judge profiles and selection disclosure, not proof of a universally superior synthesizer.
- Stock VoxCPM2 had retry machinery enabled but all 205 primary receipt rows record zero bad-case retries. Swar and CosyVoice3 have saved one-take seed-zero generation receipts; the comparison conditions remain fixed and differ from the eight-candidate selection conditions. No new synthesis or seed study was performed.
- Ten thousand paired draws use seed 20261104. The primary unit is the prompt. The sensitivity analysis resamples 133 known `pair_id` groups: 72 complete two-prompt minimal pairs and 61 singleton prompts. It preserves both members together, but does not capture every possible text dependence. Neither interval is a new-voice, new-reference, human-rater or synthesis-seed interval.
- These are intelligibility proxies. Real-speech calibration, degraded/shuffled controls and cold-process repeatability are reported by the parent calibration package. Real-speech CER is not a correction term to subtract from synthetic CER. Voice identity, naturalness and overall quality are not established by these scores.
- Public text contamination checks reuse the hash-pinned audit after verifying exact identity of all 205 queries, and add Common Voice 17 (1,446 rows, 703 normalized unique strings): zero exact/15-character-containment matches in that finite CV snapshot. Short common-phrase matches occur in Rasa, IndicVoices and OpenSLR; FLEURS/Rasa reference-source exposure and public test tuning remain confounds. No finite text screen proves speaker or training exclusion.
- D32 is call-trained. Its weights were only loaded locally on k2 and were never uploaded. A public D32 judge profile depends on the owner’s unresolved publication decision.
- Whisper reproduction matches the declared source/weights/package versions and decoding options. The original public profile omitted FFmpeg, OS and architecture, so cross-host bitwise equivalence is unverified. The current environment receipt records FFmpeg binary SHA/version and macOS arm64. Both earlier failed FFmpeg and extra-interop/seed attempts were explicitly excluded; no partial hypotheses from them enter final numbers.

## Artifacts and reproduction

- [numbers.json](numbers.json): all three normalization policies; five judges × five systems; corpus/macro CER and 50 paired contrasts plus 10 tau comparisons per policy, with 205-prompt CIs.
- [numbers_pair_clusters.json](numbers_pair_clusters.json): the same point estimates and comparisons with paired 133-group CIs.
- [contamination.json](contamination.json): compact finite public-text checks, query IDs and source hashes; no corpus transcript quotes.
- [copy_verification.json](copy_verification.json): all 1,025 transported WAVs verified byte-identical to the frozen original synthesis receipts. No WAV is included here.
- `prepare.py`, `mms_worker.py`, `whisper_worker.py`, `launch_whisper_exact.py`, `analyze.py`, and this `report.py` reproduce input admission, raw CPU inference, uncertainty and this document. Every legacy normalization/code/lexical dependency is checked against the phase1b frozen SHA256 manifest before scoring. Raw hypotheses and detailed execution receipts remain in the documented internal job directories; final aggregate files are hash-inventoried.

Source and runtime provenance, ownership, licences, retention and excluded-run details are in [README.md](README.md).
