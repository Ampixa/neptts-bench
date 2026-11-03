# Independent instrument review

Created 2026-11-03T19:01:05.001825+00:00 by Codex GPT-6 family session `/root/arena_profiles`, job `arena_judges_v1`, on owner Mac at `/Users/cdjk/github/neptts-bench/benchmark/judge_calibration_v1/calibration`. Owner: Ampixa / Arbiter `nenglish-73`. Status: reference for calibration/profile PR; last review 2026-11-03T19:14:01.271196+00:00; next review 2026-11-11; retain until superseded by a reviewed instrument revision. This Markdown records its own provenance. NOT on HF. No audio, transcripts, weights or call content were opened by this review. Model/source metadata and source code only were read. Repository software/document licence applies. Restore/reproduce by checking the hashes below, running `python3 -m pytest submission_app/tests/test_calibration_instruments.py --basetemp=.local/ui-tests -q`, and reviewing frozen phase1b aggregate/paired-bootstrap functions. Source Git history records code authorship; originals remain in this checkout and corresponding job root on ubuntu. This is not a backup.

## Verified interpretation

- Historical Whisper-small profile identity recomputes as `whisper-small-ne-048909d9dafd` from the declared model/settings/packages; concatenated evaluator, metrics and normalization code SHA256 is `9dea0cf2d556685cb4ca93e6daac794edb3fd34d875c190f863c981f46ef7bfb`. The arena's normalization keeps Latin script, normalizes punctuation/digits/case and excludes spaces from character CER. Synthetic normalization examples agree with the immutable arena functions.
- The corrected Whisper wrapper now matches the arena options: Nepali/transcribe, CPU, fp16 false, temperature 0, beam size 5, no previous-text conditioning and intraop threads 2. It retains the runtime's default interop setting, matching the immutable evaluator. The earlier attempt with an extra interop-thread override is superseded and excluded from final calibration. Matching declared source/settings/packages is checked; cross-host bitwise equality is not demonstrated, and FFmpeg/runtime default interop behavior is outside the historical profile declaration. Final data must come from the corrected worker source hash below.

- Frozen phase1b raw and conditional legacy-transliterated metrics remain distinct from `arena_raw`. Speaker bootstrap uses 10,000 fixed-seed draws and source voice clusters; paired degraded/shuffled contrasts reuse each draw. Four PV / twelve Fresh voices and fixed source text/sessions limit interpretation.
- D32 full context loads the pinned checkpoint with strict state-dict compatibility, pinned tokenizer, and explicit chunk/left-context settings (-1,-1). Streaming uses the actual c16 export; read-only export metadata confirmed chunk 16, left context 128, 33,447,414 parameters and the recorded checkpoint/tokenizer hashes.
- Analyzer streaming verification now requires those exact settings and actual/declared int8 encoder/decoder/joiner hashes. It separately pins `tokens.txt` to SHA256 `424bddc0b38d961991b7717b9c5424918cd2d4c58f2810db182675b1a49d2105`; the root agent verified all 4,096 token IDs against the pinned trained BPE (receipt `tokens_verification.json`, to be collected). Token files are not part of `export_meta.files`, so this gate is distinct from the three declared model hashes. It retains sanitized settings/pins and compares primary/cold-retest metadata; model/data paths are excluded. No running inference wrapper was changed. Updated analyzer was copied to `ubuntu:/mnt/wd_data/arena_judges_v1/calibration/analyze.py`; remote SHA256 matched the local value below.
- Catalog builder makes only the exact historical Whisper-small profile public-enabled. LONG1C/MMS/turbo/D32 are provisional diagnostic catalog entries. D32 public use remains owner-gated. UI labels the primary calibration policy explicitly; a different policy cannot be inferred to apply to historical scores.

## Limits and remaining verification

This review does not establish the final measured CER values or replicate private-input inference. Final catalog invariants and screenshots await the completed real-data catalog; no stand-in calibration numbers were created. Test-retest coverage is 32/392 unique inputs (below a prospective 10% gate), and cold-process determinism does not establish TTS seed spread. Input contamination/qualification and native listening remain separate gates.

## Reviewed source inventory

|Source|Bytes|SHA256|
|---|---:|---|
|`analyze.py`|29038|`702356719e7451596de49ea7c3ac400249d6f0ca7bc93dd4005feaf1f10d7358`|
|`run_zipformer.py`|7540|`674848466fae613dfb0e006db5d1c897a2b65a59c27a98b172af1f76c773bbb4`|
|`run_whisper.py`|3071|`ee1cc8f82e24e43858012f964ad4a134f53bd9d2592f643b36907456be658a5d`|
|`arena_normalize.py`|731|`b7198286f201131922cc5bf1183d1701f6bb85574fc1c55ae3f44a9f6b752630`|
|`arena_metrics.py`|1029|`e117eb03ba6814709d3fea2189064e285bd1378ea55a9b0c03495ba8214c4e0f`|
