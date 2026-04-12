# neptts-eval

Evaluation toolkit for Nepali TTS systems — [NepTTS-Bench](https://huggingface.co/datasets/bolne/neptts-bench).

## Install

```bash
pip install neptts-eval
```

## Usage

```bash
# Place your TTS outputs as sent_001.wav, sent_002.wav, ... in a directory
neptts-eval --wav_dir ./my_outputs/ --system-name "my_tts"
```

This runs:
- **SCOREQ** auto-MOS prediction (no GPU needed)
- **Whisper small** ASR round-trip CER/WER

And compares your system against 9 baseline Nepali TTS systems.

## Output

JSON report with per-file scores, per-category breakdown, and ranking against baselines.
