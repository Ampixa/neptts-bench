# Changelog

## 0.2.0 — 2026-04-30

### Added

- **NepaliMOS auto-MOS evaluation.** Nepali-specific MOS predictor fine-tuned
  from IndicWav2Vec base on 6,962 native-rater MOS scores from NepTTS-Bench.
  System-level Spearman ρ = 0.90 against human MOS in the paper, vs SCOREQ's
  ρ = 0.40 (Steiger's Z, p = 0.028). New module
  `neptts_eval.nepalimos_eval.evaluate_nepalimos`.
- **CLI flags** `--skip-nepalimos` to opt out and `--nepalimos-ckpt PATH`
  (or `NEPALIMOS_CKPT` env) to override the default checkpoint with a local
  full-fine-tuned one.
- **NepaliMOS column** in the report's baseline comparison. The comparison
  table now sorts by NepaliMOS by default, since it is the only metric whose
  system-ranking agreement with humans reaches statistical significance at
  α = 0.05 in the paper. SCOREQ-only sort is preserved when NepaliMOS is
  skipped.
- **`baselines.json`** updated with per-system NepaliMOS means for the nine
  TTS systems in the paper's TTS-9 panel.
- **Optional dep group `[nepalimos]`** adding torch, torchaudio, s3prl,
  huggingface_hub, soundfile.

### Changed

- Default checkpoint downloads from huggingface.co/datasets/ampixa/neptts-bench
  at `model/neptts_mos_best.pt` (full fine-tuned, head + ssl_state_dict).

## 0.1.0

Initial release.

- SCOREQ auto-MOS evaluation
- Whisper-small ASR round-trip CER/WER
- Pre-generated audio (`--wav-dir`) or TTS command (`--tts-cmd`) modes
- Comparison against 9 baseline Nepali TTS systems
