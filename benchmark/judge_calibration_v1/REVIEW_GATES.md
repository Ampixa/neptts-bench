# Judge qualification and exposure review

Prepared 2026-11-04 UTC by Codex `/root/docs_reconcile` (GPT-6 family; exact serving model unknown), job `arena_judges_v1`, on the local `Ampixa/neptts-bench` checkout. Owner: Ampixa / Arbiter `nenglish-73`. Status: reference for this PR and Arbiter review; last review 2026-11-04, next review 2026-11-11; retain until superseded by reviewed qualification evidence. This document records its own provenance. It contains aggregate results and source identities, no audio, private transcripts or model weights. NOT on HF. Code/document licence: repository MIT; original audio/model/corpus restrictions remain with their custodians. Reproduce by reading the named source schemas and recomputing the stated aggregate counts; source hashes are below. No new human listening, synthesis, membership experiment or decontamination was performed for this review. No blog submission is authorized by this handoff.

## The supplied criticisms, checked against evidence

| Premise | Supported conclusion and correction | Evidence |
|---------|-------------------------------------|----------|
| Swar trained on FLEURS and Rasa; these supply two suites | The [model card](https://huggingface.co/Aarjan/swar#training-data), read 2026-11-04, declares both corpora, IndicVoices-R, OpenSLR 43/54, LibriTTS-R and permissioned podcasts. The historical head-to-head has two **reference conditions** on the same 205 designed texts, one FLEURS-derived and one Rasa-train-derived, plus FLEURS-test cloning. Corpus exposure is established; training on the exact FLEURS-test recordings is **unproven**. | E1 §1; E2 pinned card/source audit |
| MMS reads real speech at about 9.8%, so cannot separate systems below 10% | The historical 82-target real FLEURS result is 9.774% corpus CER / 10.135% mean sentence CER. This is instrument error, not a mathematical discrimination floor. The paired Cosy−Swar corpus-CER difference is +2.788 pp [1.294, 4.590], despite both synthetic scores being below 10%. It does not establish human intelligibility. | E1 §§2–3; E4 `mms_real` and `paired` |
| Cloning uses 50 sentences / four voices | The audited historical set has 82 recordings but 50 normalized text clusters, with per-voice counts 25/25/25/7. CIs must preserve repeated texts; four voices cannot support broad speaker-population claims. The current public card instead advertises **v1.1, 64 distinct prompt+sentence pairs** from four voices. These snapshots must not be conflated. | E1 §3; E4; [current card](https://huggingface.co/Aarjan/swar#results-neptts-bench-205-sentences-one-take) |
| Best-of-N ASR selection can game CER | The selector's own score is not independent validation. The earlier single→best8 comparisons change multiple settings. A matched 24-pair same-settings comparison gives independent MMS change −0.793 pp [−2.876, +1.217] and UTMOS −0.0212 [−0.0777, +0.0258]; neither detects a gain. This demonstrates the need for disclosure and independent judges, not a measured human-quality gain or proven deliberate gaming. | E1 §4 |
| Voice match is one ECAPA model | That is the historical cloning score. Its real same-speaker ceiling depends on enrollment/text/channel. The later two-verifier PV calibration gives ECAPA EER 21.33% [0,40] and WavLM EER 19.00% [0,44], with four people; adding easy cross-cohort negatives must not hide this failure. | E1 §5; E5 current phase1b section |

The historical source audit documents test reuse for prompt choice, inference settings, frontend rules and ancestor-checkpoint selection. That makes the existing sets development evidence rather than an independent held-out ranking. The weight-only membership experiment is **inconclusive**: test−dev −0.00104 [−0.00437, +0.00237]; its deliberate-fine-tuning positive control also failed significance. It proves neither membership nor absence. E3 supersedes E2's earlier assertion that FLEURS dev/test voices were disjoint: the empirical clusters overlap.

## Contamination screen: coverage is finite, decontamination unresolved

E6 screens all **196 phase1b calibration targets** (100 PV / 96 Fresh) against pinned text indexes. Normalization is NFC/lowercase with letters, marks and numbers retained. It checks exact matches and bidirectional containment when the contained string has at least 15 characters. Recognition never receives these index texts. Counts below are aggregate observations, not training-membership probabilities.

| Corpus | Screen actually run | Rows / unique normalized variants | Observed exact / containment hits | What remains unverified |
|--------|---------------------|-----------------------------------|-----------------------------------|-------------------------|
| FLEURS | Indexed ne_np train/dev/test text | 4,363 / 1,993 | 0 / 0 | Exhaustive versions, fuzzy/audio/speaker overlap; historical cloning audio is separately known FLEURS-test-derived |
| Rasa | Pinned Nepali indexed text | 31,255 / 23,637 | 0 / 0 | Other revisions, actual final training CSVs, fuzzy/audio/speaker overlap |
| IndicVoices-R (IVR) | Indexed text/verbatim/normalized fields | 42,576 / 69,758 | 0 / 0 | Actual training subset, other versions, fuzzy/audio/speaker overlap |
| IndicVoices | Separate indexed field screen | 249,422 / 354,237 | 0 / 0 | Same limitations; this is additional to IVR |
| OpenSLR 43 | Included in combined 43/54 index | Combined: 159,969 / 87,921 | No exact hit attributed to 43; **seven combined containment hits lack corpus attribution** | A separate 43-only containment count cannot be recovered from E6's aggregate; do not label independently clean |
| OpenSLR 54 | Included in the same combined index | Same combined count, not additive | One exact PV hit explicitly marked `slr54`; seven combined containment hits | Exact hit retained; per-corpus containment allocation, fuzzy/audio/speaker overlap unresolved |
| Common Voice 17 ne-NP | train/dev/test/validated TSV sentence fields | 1,446 row observations / 703 unique | 0 / 0 | TSV views overlap and are not independent samples; other versions and fuzzy/audio/speaker overlap unverified |

The combined OpenSLR screen has **eight hit targets: seven PV and one Fresh** (one exact PV, six PV containment, one Fresh containment). No hit was removed from calibration. This report publishes no matched text or speaker identity. Zero hits in a finite text screen does not certify unseen people, unique audio or exclusion from private training. Fresh remains silver-transcribed, historically Conformer/ECAPA-selected; 72/96 real targets and all 12 voices have documented Audio8 exposure in E1. Swar podcast overlap and LONG1C/D32 private-training exclusion are unknown. These are calibration/development cohorts, not a universally clean TTS test set.

The **205-prompt track is a separate contamination question**: E1/E2 find 12/205 exact public-corpus matches, all short common phrases, and documented benchmark tuning. Counts above must not be described as a new 205-prompt decontamination audit.

## Instrument gates and limits for this PR

| Required evidence | Existing evidence | Qualification limit / gate |
|-------------------|-------------------|----------------------------|
| Real-speech calibration | Phase1b includes all 100 PV / 96 Fresh, raw and legacy-transliterated corpus/sentence CER and CIs | Report the exact model, decoder and normalization. V3-turbo results do **not** calibrate arena Whisper-small. D32 streaming/full-context need separate measurements. No ASR score is human MOS. |
| Good/bad controls | All 196 matched degradations and within-cohort transcript shuffles in phase1b; wrong-speaker trials for SV | Prior coarse ASR screen requires ≥90% worsened items plus positive paired lower bounds. LONG1C/MMS pass coarse ordering; V3-turbo Fresh fails (75/96 degraded, 80/96 shuffled worsen). Its shuffled **sentence**-CER CI includes zero although corpus-CER CI is positive. Coarse passing does not qualify subtle TTS differences. |
| Repeatability / seeds | Two cold-process retests per phase1b ASR, 32 inputs per pass, all voices, zero changed hypotheses | 32/392 unique audio inputs = 8.16%, below the prospective 10% sampling gate. No TTS seed spread, cross-device equivalence or human test–retest established. Record missing/current-profile retests separately. |
| Useful paired uncertainty | Phase1b 10,000 speaker-cluster bootstrap draws, seed 20261103; four PV / 12 Fresh identities | Descriptive fixed-session/text intervals, not two-way population qualification. Historical cloning has only 50 text clusters. The 205-track comparisons must pair the identical prompt IDs and disclose reference condition; mixing two reference conditions as independent prompts is invalid. |
| Contamination | Finite public text screens above and historical source/test-use audit | Actual training membership, private sources, fuzzy/audio/speaker exclusion and a sealed unseen test remain unresolved. No claim that all models have clean training exclusions. |
| Selection and judge independence | New contract can distinguish single take / best-of-N and disclose selector/judge-family use | Legacy unknown declarations cannot silently become single take. A LONG1C-based selector cannot independently validate its own selected output using LONG1C, nor can D32 automatically be treated as independent of related family training. Disclosure is necessary but not execution attestation. |
| Public judge availability | Local calibration can evaluate D32 without releasing weights | A public D32 profile depends on the owner's open weight/public-judge decision. Do not upload weights or infer permission from capability results. |

Native Nepali transcription/accepted-script review, human quality/listening calibration, expanded independent speakers/sessions, speaker-identity qualification and sealed-text evaluation remain separate work. They need separate agent-hours, external waits and native-rater human moments; this PR supplies no new human-listening evidence. Arena workflow tests and JS/lint checks verify software behavior only.

## Exact evidence identities

Sources live under `~/github/tts/experiments/` on the local workstation, with their original-host/copy provenance retained in each source README/inventory. This note is a synthesis, not a new backup or replacement for those sources. No source data was copied here.

- **E1:** `nepclone_bench_v1/AUDIT.md`, current phase1b corrections included.
- **E2:** `swar_contamination_audit_v1/FINDINGS.md`; its Mac directory contains no README, so this is the local evidence entry point. Full root recorded there: `ubuntu:/mnt/wd_data/swar_contamination_audit_v1/`. Pinned Swar card revision `8ca6f1bbc9da567d92158822e24673c2084e7e86`, source `32fd7b457786d6675d49449fa25db55187e63d89`, cloning source `0db18a7ea419e689c41bf6f3606dd45802aa76ba`.
- **E3:** `swar_membership_v1/README.md`, empirical membership/speaker study and failed positive-control gate.
- **E4:** `nepclone_bench_v1/audit_numbers.json`, independently inspected keys before reading aggregate values.
- **E5:** `nepclone_bench_v1/PILOT.md` **current phase1b section**, `phase1b/README.md`, `phase1b/numbers.json`; old pilot values are explicitly superseded.
- **E6:** `nepclone_bench_v1/phase1b/contamination.json` and `contamination.py`. Input-index SHA256 values are embedded in the JSON; the script establishes exact field/containment semantics. Aggregate schema and every hit's non-text metadata were inspected; matched strings were not accessed.

The live card was read via browser 2026-11-04. The browser could not open its pinned historical revision; that revision's historical conclusions are attributed to E2 rather than falsely claimed as newly fetched. `swar_stress_v1/` also has no local README; its source was not used as quantitative evidence here.

| Local source | Bytes | SHA256 |
|--------------|------:|--------|
| `nepclone_bench_v1/AUDIT.md` | 13489 | `864ad4281a046fe84e3243dc89b59711a41bfb766ee4a5d6a997705f01d21fa1` |
| `swar_contamination_audit_v1/FINDINGS.md` | 12902 | `9f0af3f49b2759ddaca318f2ee1796595adca6e74bd8c0484d81986ef8b9ecad` |
| `swar_membership_v1/README.md` | 15208 | `c6685784543e336caa2196e7d36d4c1f9cc99873ee9bf686544c621d1dbea2f5` |
| `nepclone_bench_v1/audit_numbers.json` | 26953 | `96a4183c919a8cb7120d5756df5e715552760012ec095f5ddc0a1818efa55d1f` |
| `nepclone_bench_v1/PILOT.md` | 31830 | `87a740cac491be33438dc629159fdd775430428c77026e2c76b2590d2e5a15a3` |
| `nepclone_bench_v1/phase1b/README.md` | 10450 | `46763399f413f27a78a99dd5728935b4377ecdeebac9cdee350e2ad49ce311e5` |
| `nepclone_bench_v1/phase1b/numbers.json` | 105574 | `7edaf82328596bfeb6c9699d66118c1c89d217c3e429996323a1c708ca661181` |
| `nepclone_bench_v1/phase1b/contamination.json` | 3575 | `f5cc200161ead6d069fa8b89a523cb8b37a25f54a36278cc5c748c5bba28fa91` |
| `nepclone_bench_v1/phase1b/contamination.py` | 3517 | `edeb13371558dda7b4edb847a19b325bffa3d620593d2b036a31e677960aa5b6` |
