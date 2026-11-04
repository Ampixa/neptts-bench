# Reference-system reuse audit

Created 2026-11-04 UTC by Codex `/root/reference_context` (GPT-6 family, exact serving model unknown), job `arena_d32_rescore_v1`. Owner: Ampixa / Arbiter nenglish-73. Status reference; retained for the current rescore PR. Last review 2026-11-04; next review 2026-11-11 or PR closure. This document records its own provenance; no audio, hypothesis text, model weights or personal information is included. NOT on HF. Original model/audio usage limits remain in the head-to-head and calibration READMEs.

The requested reference comparison can reuse **Swar, stock VoxCPM2 (`vox2_clone`) and CosyVoice3 neutral (`cosy3_elx_neutral`)**, each on all 205 designed prompt IDs, using the existing FLEURS-reference generation condition. Their saved D32 full-context hypotheses pass the checks below. These are comparison context; they must not become arena entries.

## What was checked

The actual Ubuntu head-to-head `manifest.json` keys are `bootstrap`, `counts`, `created_utc`, `expected_conditions`, `input_hashes`, `jobs`, `missing_conditions`, `seed`. Designed prompt job keys are `benchmark`, `category`, `condition`, `id`, `key`, `reference`, `text`. The manifest contains 205 FLEURS-reference jobs, 205 Rasa-reference jobs and 82 cloning jobs. Its SHA256 is `9b0662aa6318162da0cf782f438c2eea0049065ceae044789b17d533199ad8ca`.

The existing ranking preparation verifies exact prompt ID/text identity against the arena prompt file before admitting source WAVs. This audit independently rehashed all **615 selected reference WAVs on Ubuntu and all corresponding 615 copies on k2** against those frozen tasks: all matched. The retained D32 hypothesis file contains 1,025 unique successful system/prompt observations and matches every one of its 1,025 k2 task hash identities. Only the three systems named above are needed here. The SHA256 of `k2:~/bench2/arena_judges_v1/ranking/d32_full_fleurs.jsonl` is `54970ffe2eb44c14ba183e2026736d71865a67a049a9a382ca96598a2faf01b9`, exactly equal to the source pin in committed `benchmark/judge_calibration_v1/ranking/numbers.json`.

Existing aggregate results contain CER only. WER for this job must be recomputed from the verified retained hypotheses with the same frozen legacy score2 transliteration and suite metric code. No hypothesis or private audio needs to be copied to the public checkout.

Both D32 ranking shard metadata files match the calibration instrument: checkpoint SHA256 `97392051e0fd5aa764b0098c7a39ebe2223e521c76701bfffdf5b99f1f39b213`; BPE SHA256 `efef2f02304881a9a5aa269aec829b9ee1ba1ada0ff48f1f2d3feae7fb51394f`; 33,447,414 parameters; chunk size −1; left-context frames −1; CPU, one thread; Torch 2.8.0; Lhotse 1.33.0; worker SHA256 `674848466fae613dfb0e006db5d1c897a2b65a59c27a98b172af1f76c773bbb4`; all five recipe-module hashes equal the committed calibration record. The metadata files are the `part0` and `part1` files; there is no merged `d32_full_fleurs.meta.json`.

| Source receipt | SHA256 |
|---|---|
| k2 ranking tasks | `a440d4e2c68c6ed72eb47a91cdd20f4aa69aa63f0885ba4d396a0619558b0e18` |
| k2 D32 full ranking part0 metadata | `098958abcad436c48d1ee18e73965fa5683f4c4f535a29e7bc1cdaf387da297e` |
| k2 D32 full ranking part1 metadata | `e96a9cee6be0ac316e027c6a487a4b87b1cebf588289e8574d28c7acd7da455c` |
| k2 D32 full calibration metadata | `9fd369d567813a57b7a613884a55f4463d0eee6f59ff3d9149555fff38c3179b` |
| Ubuntu ranking source receipts | `a60c055cd93b3156a3185b5ba08b6db2dcc49a8e60356684b1b745e5203d6a04` |

## Generation-policy identities

| System | Observed saved generation condition | Model receipt SHA256 | Item receipt SHA256 |
|---|---|---|---|
| Swar | Seed 0, NFE 16, CFG 2.0, prompt-matched pace, silence removal. No best-of-N selector is recorded; item schema has no explicit retry-count field. | `24379f8cfa6f0b532d2ddf780207fdd68ea909e67dc69a5c6eda2d5580487845` | `d109214a0860d878b655597c43c71e91c0504c604897ff064313b80c361e7c2b` |
| Stock VoxCPM2 | `openbmb/VoxCPM2` revision `32279effe8c19989596f05d353d1447f51d9e915`, CFG 2.0, 10 steps, seed 0. Retry machinery was enabled, but all 205 FLEURS-reference item receipts record `badcase_retries=0`. This is the available observed single-attempt panel. | `6109941285b227ddd68cda008de757f329b79a82bfdda3286ed4ae4b51307308` | `cfad3037a10b8b506658e1544025c05d085295dad7e3dd91e46089f2daf38470` |
| CosyVoice3 neutral | Base revision `29e01c4e8d000f4bcd70751be16fa94bf3d85a18`; saved neutral instruct2 condition, seed 0, no reference transcript, speed 1.0. No best-of-N selector is recorded; item schema has no explicit retry-count field. | `6a15cedfee97ea6abb89fed8cb9fad106e94e9730740de51a91ab4486ee919db` | `e78b01f279fe55f6252793f1f462173ee6c90c7e399beab188120865d72ef2c9` |

The fine-tuned `vox2_beat1` and `vox2_beat2` conditions each selected the lowest LONG1C CER from eight recorded candidates, with additional internal retries incompletely reconstructed. They must not be described as single take or substituted for stock `vox2_clone`. Their model/checkpoint/generation differences prevent a causal best-of-N conclusion from these saved outputs. Their existing diagnostic results remain in the earlier calibration package.

## Real recordings and qualification limits

There are **no target real recordings for the 205 designed prompts** in the inspected head-to-head manifest: zero FLEURS/Rasa designed jobs carry `real_audio`; all 82 real target recordings belong to the separate cloning track. The FLEURS/Rasa `reference` object supplies enrollment/reference speech, not a recording of the designed target sentence. A same-205 real-speech score cannot be produced from those enrollment files. Use the separately labelled 100-PV/96-Fresh calibration beside the scores; do not pool different target panels.

The exact D32 full-context legacy-transliterated calibration is PV corpus CER **7.8755%** and Fresh **13.6894%**. Prior matched degradations worsened 98/100 PV and 96/96 Fresh items, with corpus-CER difference lower bounds +56.89 and +55.58 percentage points. Transcript shuffles worsened every item in each cohort, with positive paired corpus-CER lower bounds. These are coarse content controls; a content judge is not expected to detect a wrong speaker saying the correct words, and these numbers do not establish voice identity or subtle human preference.

The prior cold-process retests changed zero hypotheses on 32 inputs per pass, with zero cold-to-cold changes. The 32/392 unique retest fraction is 8.16%, below the prospective 10% sampling gate. No new synthesis-seed spread, new-voice generalization or native human listening was measured. Job-specific reruns and their results must be reported separately by the parent job.

Reuse the finite public-text contamination audit in `benchmark/judge_calibration_v1/ranking/contamination.json`, whose complete file SHA256 is `b4b51d1198a87092f153adbd374e210fc9c8fd13bc815fe3715e27ceb1741a7e`, after retaining the exact prompt identity check. It screens FLEURS, Rasa, IndicVoices/IVR, OpenSLR and Common Voice snapshots; short common-phrase matches and benchmark tuning remain. FLEURS/Rasa enrollment-source exposure is a known confound. The private call-trained D32 training exclusion is unknown. Zero finite text matches cannot certify unseen people/audio/training. `benchmark/judge_calibration_v1/REVIEW_GATES.md` gives the independently verified corrections to the inherited Swar-bench premises.

Paired prompt intervals condition on these fixed saved systems, reference voice and synthesis take. A 133-group minimal-pair sensitivity can preserve the 72 complete pairs and 61 singleton prompts. Neither interval supplies a new-speaker or human-rater population claim. D32 real-speech error is an instrument property, not a term to subtract from synthetic CER.

No original data was changed, inference launched, model loaded, service touched or new backup created by this audit. The existing hypothesis retention is owned by `arena_judges_v1` and its README; this job reuses it read-only. Reproduce this audit by inspecting the schemas above, rehashing selected reference WAVs against frozen tasks on both hosts, verifying saved result/source pins and comparing each metadata identity with committed calibration `numbers.json`.
