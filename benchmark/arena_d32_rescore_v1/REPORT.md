# D32 full-context rescore: five approved arena submissions

All five approved submissions now have complete 205-prompt D32 offline reports. **Nothing has been imported to the live arena.** Merge/deployment and separate reviewer import/approval remain with the Arbiter; [IMPORT.md](IMPORT.md) gives exact commands. The five existing Whisper reports and audio bindings were rechecked unchanged in [legacy_preservation.json](legacy_preservation.json).

Instrument: `d32v2-s60000-full-translit-279b835f5fe9`; 33,447,414 parameters, step 60000, FP32 full context, CPU, one thread, greedy, chunk/left context −1/−1, frozen legacy score2 conditional loanword transliteration. Call-trained private weights remain on k2. **Scores are reproducible only by Ampixa until the owner decides on weight release.** The profile card states this limitation; a public numeric report does not make the judge public.

## Real-speech instrument error beside the arena scores

| Exact judge and policy | PV 100 real targets, corpus CER | Fresh 96 real targets, corpus CER |
|---|---:|---:|
| D32 full / legacy transliteration | **7.88%** | **13.69%** |
| Historical Whisper-small / arena normalization | **49.01%** | **115.45%** |

The final executable reran every real calibration input: PV 7.875536% and Fresh 13.689444%, both **0.000000 percentage points** from the calibration record, against a maximum permitted difference 0.05 pt. All 196 hypotheses were identical to the original full-context run. [calibration_gate.json](calibration_gate.json) binds these observations to the final code/profile hash; [instrument_verification.json](instrument_verification.json) records checkpoint, tokenizer and original five recipe pins. The profile additionally pins the 21-module local recipe import closure, clean icefall revision, all 15 normalizer/resource files and runtime versions. No weights were copied.

These panels have 4 PV and 12 Fresh voices and fixed sessions/texts. Existing speaker-bootstrap 95% CIs are PV [4.71,9.57]% and Fresh [10.90,16.70]%. Instrument error must never be subtracted from synthetic error or treated as a hard discrimination floor.

## Approved submissions

Every row covers the same 205 required prompts. CER/WER below are **macro means of sentence error rates, percent**, matching arena aggregation; brackets are 95% paired-prompt bootstrap intervals. The shared draws preserve pairing for differences. Real-speech D32 corpus CER is **PV 7.88% / Fresh 13.69%**; Whisper is **49.01% /115.45%** under its own historical policy.

| D32 rank | Approved system | D32 CER [95% CI] | D32 WER [95% CI] | Whisper CER [95% CI] | Whisper rank |
|---:|---|---:|---:|---:|---:|
| 1 | WiseAI_TTS | 7.29 [5.71, 9.09] | 24.21 [19.86, 29.04] | 188.81 [102.01, 296.89] | 3 |
| 2 | WiseAI_TTS_Aayush | 7.92 [6.12, 9.92] | 26.37 [21.80, 31.41] | 133.89 [56.29, 236.82] | 1 |
| 3 | WiseAI_TTS_Sujata | 10.35 [8.01, 13.01] | 29.79 [24.93, 35.16] | 177.93 [94.50, 283.69] | 2 |
| 4 | WiseAI_TTS_Prakriti | 11.11 [8.83, 13.66] | 32.17 [27.47, 37.29] | 214.18 [134.54, 308.54] | 4 |
| 5 | Astadhwani | 16.46 [13.88, 19.15] | 39.88 [34.74, 45.34] | 318.40 [242.05, 403.02] | 5 |

The five-entry ranking changes from **Aayush → Sujata → WiseAI_TTS → Prakriti → Astadhwani** to **WiseAI_TTS → Aayush → Sujata → Prakriti → Astadhwani** (lowest mean CER first). Kendall tau-b is **0.60**, paired 95% CI [0.20,1.00]; the 133-group minimal-pair sensitivity gives the same point and interval. WiseAI_TTS moves from third to first, reversing its order against Aayush and Sujata. These are observed instrument-specific rankings, not a human-quality result.

The top two D32 systems are unresolved: WiseAI_TTS minus Aayush is −0.62 CER points [−2.72,+1.36] and −2.16 WER points [−5.61,+1.24]. Corpus weighting even reverses their point order: Aayush 6.86% versus WiseAI_TTS 7.22% corpus CER; their paired corpus difference +0.36 pt [−0.90,+1.79] also includes zero. A first-place point estimate is not evidence of a winner.

The comparison changes both recognizer and normalization: historical Whisper preserves Latin script and excludes spaces from CER; calibrated D32 conditionally transliterates Latin hypotheses when references contain no ASCII Latin and uses suite CER including spaces. The stored legacy reports contain numeric scores, not raw hypotheses, so this job cannot causally separate the ASR change from the loanword/space-policy change. Historical values remain untouched.

## Reference context, not arena entries

D32 real-speech corpus CER: **PV 7.88% / Fresh 13.69%**. Each synthetic reference below uses the same 205 prompts and the existing FLEURS enrollment/reference condition. The 615 original WAV hashes were rechecked on both Ubuntu and k2. The saved full-context hypotheses match the earlier committed source pin; CER/WER were recomputed through the final scorer normalizer. No additional synthesis or API was used.

| Reference system | Mean CER [95% CI] | Mean WER [95% CI] | Corpus CER | Corpus WER |
|---|---:|---:|---:|---:|
| Swar | 7.76 [6.30, 9.31] | 27.97 [23.50, 32.85] | 7.47% | 26.82% |
| VoxCPM2 stock, observed single attempt | 33.72 [27.16, 41.54] | 58.40 [51.75, 65.70] | 30.33% | 56.06% |
| CosyVoice3 neutral | 9.93 [8.13, 11.83] | 31.07 [26.26, 36.25] | 10.46% | 30.81% |

VoxCPM2 stock has zero bad-case retries in all 205 item receipts. The two finetuned VoxCPM2 beat runs used LONG1C-selected best-of-eight outputs and are excluded from this single-attempt context. Swar/Cosy receipts record no best-of-N selector but lack an explicit retry-count field. [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) records exact revisions, hashes and limits.

**No real target recordings exist for the designed 205 prompts** in the source manifest. The 82 real recordings belong to a different cloning track; enrollment/reference audio is not target speech. Therefore no misleading same 205 real row is added. PV/Fresh calibration is shown separately.

## Paired differences and uncertainty

10,000 shared resamples, seed 20261104, pair identical prompt IDs across all eight systems. The main analysis resamples 205 prompts; sensitivity resamples 133 minimal-pair families/singletons (72 pairs + 61 singletons) while keeping both pair members. Intervals condition on saved systems, enrollment voice and synthesis take; they do not cover new speakers, synthesis seeds or human judgments. No multiple-comparison correction is applied; all 28 pairwise contrasts and both weighting conventions are retained in [numbers.json](numbers.json).

| D32 difference (first minus second), points | Mean CER [paired 95% CI] | Mean WER [paired 95% CI] | 133-group CER CI |
|---|---:|---:|---:|
| WiseAI_TTS − WiseAI_TTS_Aayush | -0.62 [-2.72, +1.36] | -2.16 [-5.61, +1.24] | [-2.72, +1.30] |
| WiseAI_TTS − WiseAI_TTS_Sujata | -3.06 [-5.88, -0.41] | -5.58 [-9.72, -1.52] | [-5.91, -0.35] |
| WiseAI_TTS − swar | -0.47 [-2.46, +1.62] | -3.76 [-7.61, +0.10] | [-2.47, +1.58] |
| WiseAI_TTS − cosy3_elx_neutral | -2.64 [-4.52, -0.67] | -6.86 [-10.68, -3.02] | [-4.54, -0.65] |
| swar − cosy3_elx_neutral | -2.17 [-4.15, -0.15] | -3.10 [-7.32, +0.98] | [-4.17, -0.15] |

The sample resolves several coarse differences, while WiseAI_TTS/Aayush and WiseAI_TTS/Swar include zero. Narrower fixed-prompt intervals do not repair judge bias or training exposure. Corpus and macro ranking sensitivity is explicitly retained.

## Qualification, privacy and reproducibility

- Known-good/bad evidence: original matched degradation worsened 98/100 PV and 96/96 Fresh items; transcript shuffles worsened all targets. Their paired corpus-CER lower bounds are positive. Reused calibration controls establish coarse content discrimination; ASR cannot score speaker identity when the words are correct. No wrong-speaker identity or human-preference claim is made.
- Repeatability: this final decoder reproduced all 196 original calibration hypotheses. A separate cold process redecoded 21/205 prompts per approved submission (10.24%, 105 total), with zero changed CER/WER/edit-count rows. No private submission hypotheses were stored, so this is numeric repeatability. Earlier calibration has two 32-input cold hypothesis retests; no TTS seed spread was measured.
- Contamination: the exact 205-prompt public-text screen and input hashes are reused from [the calibration audit](../judge_calibration_v1/ranking/contamination.json), covering FLEURS, Rasa, IndicVoices/IVR, OpenSLR 43/54 and Common Voice snapshots. Twelve short common-phrase matches and benchmark tuning remain known confounds; FLEURS/Rasa enrollment exposure is explicit. Actual private D32 training exclusion and submitter training membership are unverified. Fresh is silver-transcribed and historically ASR/speaker-selected. No sealed or universally uncontaminated ranking is claimed.
- Scope corrections to inherited premises are in [REVIEW_GATES.md](../judge_calibration_v1/REVIEW_GATES.md): 9.8% real MMS error is not a mathematical floor; historical 82 recordings / 50 text clusters differ from the current advertised 64 prompt pairs; best-of-N comparisons are confounded and do not prove deliberate gaming.
- Credentials were read only in VM memory. Submitted packages streamed VM→k2 without a Mac audio copy. All inference used k2 CPU, one thread, after memory-pressure checks; no live hat ports/processes, arena results or proper-voices service were changed. The initial missing-pydantic preflight failed before scoring; validator dependencies were installed only in this job directory, with the shared runtime unchanged.
- All five new ZIP copies and transient extracted WAVs were deleted after consumers finished and lsof found no open inputs. Package bytes removed: 63,823,040; allocated package bytes: 65,339,392; measured free-space increase: 64,221,184 bytes (concurrent host activity can affect the delta). Zero WAV/ZIP files remain in the job tree. [audio_cleanup.json](audio_cleanup.json) records IDs, hashes and deletion reasons; the full per-ID audio hash manifest remains on k2. Arena originals remain intact; ZIP-container byte history may not be restorable, while original WAV bytes can be re-downloaded by authorized reviewers.
- Software validation: 104 Python tests, 2 Node tests, Ruff, JS syntax, receiver compilation and diff checks passed locally. Lightweight CI cannot load the private model; the actual 196-input calibration gate and 1,025 approved-prompt scores were run on k2.

Regenerate the numbers with `PYTHONPATH=neptts-eval/src python3 benchmark/arena_d32_rescore_v1/analyze.py`, then this report with `python3 benchmark/arena_d32_rescore_v1/report.py`. These commands use public numeric evidence only. See [README.md](README.md) for artifact inventories/retention and [neptts-eval documentation](../../neptts-eval/README.md#private-d32-reviewer-judge) for private offline prerequisites. Initial effort estimate: 8–16 agent-hours plus CPU/transfer waits, native human listening 0; no human listening study was performed.

Prepared 2026-11-04 UTC by Codex `/root`, job `arena_d32_rescore_v1`; owner Ampixa / Arbiter nenglish-73. Reference evidence for this PR; review by 2026-11-11 or PR closure. NOT on HF. This document records its own provenance; source/report SHA256 inventory is in README.
