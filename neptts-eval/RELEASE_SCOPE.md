# Documentation and release scopes — 2026-11-04

This source-controlled note and [`documentation_scopes_v1.json`](documentation_scopes_v1.json) were created by Codex `/root/docs_reconcile` (GPT-6 family, exact serving model unknown), job `arena_judges_v1`, in the local `Ampixa/neptts-bench` checkout. Owner: Ampixa / Arbiter `nenglish-73`. Status: reference for review; last review 2026-11-04 UTC, next review 2026-11-11. Retain until replaced by a reviewed release/filter manifest. Git history records code/document revisions; the JSON records exact input sizes and SHA256. These compact documentation artifacts contain no audio, weights, private rating rows or call content and are NOT on HF. Repository documentation/software is MIT; input audio/model permissions remain separate.

## Which count means what

| Scope | Count | Source and filter | Used by |
|-------|-------|-------------------|---------|
| Current checked-in prompt inventory | 365 = 205 synthesis + 160 natural references | `benchmark/sentences.json`, SHA256 in manifest; `sent_*` vs `chirp_*` IDs | Root README, source CLI prompt loading |
| Current arena submission | 205 synthesis prompts | `sent_*` inventory only, exact IDs/content hashed by arena | Arena submission contract |
| Current phonological inventory | 149 prompts, 77 pair IDs = 72 complete pairs + 5 singleton probes | `category=phonological_minimal_pairs`, group by `pair_id` | Root README; no claim of 77 complete pairs |
| Historical paper human study | 6,962 ratings, 180 contributing self-declared native speakers, 10 TTS + 2 human baselines | Checked-in paper's human-MOS table and rater-pool paragraph; 207 native registrations include noncontributors | README historical table, with reported rater-cluster CIs |
| Predictor matched targets | 1,697 utterances from 5,865 matched-audio ratings, including 5,804 native-registration ratings | `paper/REVISION_NOTES.md`, “Sample-size correction” | Predictor docs; distinct from study-wide counts |
| Research CLI saved baselines | 9 TTS systems | `neptts-eval/src/neptts_eval/baselines.json` | CLI comparison report |
| Earlier README | 164+ listeners, 5,760+ ratings, 193 sentences | Older documentation; exact export date/filter not established here | Superseded, not a current release total |
| Broader Ubuntu export described in prior review | 7,249 ratings / 3,547 audio files | Prior review reports a different export scope; row-level mapping unresolved | Excluded from current study totals |
| Hub storage metadata | 1,894 audio paths | Prior review's 2026-11-03 listing at dataset revision `d07d5495968f5e1ec4e18e852bb926f8babc1335` | Storage inventory only; not a count of independently rated clips |

The root README's MOS values and intervals now match the checked-in paper table. That is a documentation reconciliation against a named historical source, not a fresh analysis of private ratings. The paper's aggregate statements still call 77 IDs “minimal pairs”; the current prompt inventory requires the singleton qualification above. This update does not silently rewrite the paper, Hub card or live portal historical table.

## Remaining release gate

A single reviewed row-level study/release/filter manifest connecting the rating export, prompt coverage, public media and per-source permissions is **not available in this change**. The differing export totals must not be added together or described as one independently verified release. A future release should record exact study membership, database/export revisions and media rights, then regenerate all public surfaces from that reviewed manifest. No private DB, public audio payload or checkpoint tensor was opened for this documentation correction.

The [checkpoint documentation](../model/README.md) identifies the actual v9 head-only default and a public full-state candidate. Tensor/config verification and score recalibration remain a separate unresolved reproduction gate; correcting the docs does not claim it passed.

## Reproduce the inventory

Read JSON schemas first. Count the `sent_*` and `chirp_*` IDs in `benchmark/sentences.json`; group its phonological records by `pair_id`; compare source SHA256 with `documentation_scopes_v1.json`. Study values come from `paper/neptts_bench.tex` Table `human_mos`, with 95% rater-cluster bootstrap intervals (2,000 resamples), and predictor target counts from `paper/REVISION_NOTES.md`. No audio is required for these checks. The manifest lists unverified mappings explicitly.

## Source-install validation and disposable artifacts

[`source_install_validation.json`](source_install_validation.json) is a compact reference receipt from the same agent/job, owner, source checkout and review/retention schedule as this note. It records the clean Python 3.12 source/editable installs, dependency versions, successful CLI help and `pip check`, unrelated-directory offline prompt load (205 + 160), the regression test, source/log hashes and explicit earlier local Python failures. It contains no inference results. Its named current consumer is review of the documented install route; supersede it when a later reviewed release validates that route. NOT on HF; no weights/audio were downloaded. The disposable installation environments and logs were removed after verification; measured sizes, process check, reasons, space change and reproduction commands are in the receipt. No pre-existing data was removed, and original temporary log bytes are no longer restorable without rerunning their commands.

Final source-loader review on 2026-11-04 makes the versioned module-relative prompt bank take precedence over a stale 193-record user cache or an unrelated working-directory file. Three regression tests cover offline source loading, stale-cache precedence and chained download failures. The validation receipt preserves the original clean-install hashes separately from these final reviewed source hashes; dependency/build metadata did not change, so a second dependency installation was unnecessary. Test fixtures remained inside `.local/docs-tests` and were removed after the completed run, with file hashes and measured sizes in the receipt.
