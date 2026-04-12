# NepTTS-Bench

The first comprehensive benchmark for evaluating Nepali text-to-speech systems.

## Results

Human MOS ratings from 164+ native Nepali speakers (5,760+ ratings):

| System | Human MOS | Type |
|--------|----------|------|
| Natural Speech | 3.91 | Human (YouTube) |
| Human (read) | 3.57 | Human (recorded) |
| TingTing Asmita | 3.49 | Nepali-specific |
| ElevenLabs v3 | 3.48 | Cloud (ElevenLabs) |
| Piper | 3.47 | Open source |
| TingTing Subina | 3.42 | Nepali-specific |
| Edge TTS Hemkala | 3.31 | Cloud (Microsoft) |
| Gemini (Hindi) | 3.29 | Cloud (Google) |
| Edge TTS Sagar | 3.28 | Cloud (Microsoft) |
| Gemini Flash | 3.19 | Cloud (Google) |
| TingTing Sambriddhi | 3.14 | Nepali-specific |
| gTTS | 2.56 | Cloud (Google Translate) |

## Quick Start

Evaluate your Nepali TTS system against our baselines:

```bash
pip install neptts-eval

# Place your TTS outputs as sent_001.wav, sent_002.wav, ... in a directory
neptts-eval --wav_dir ./my_outputs/ --system-name "my_tts"
```

This runs SCOREQ auto-MOS + Whisper ASR round-trip and compares against 12 baseline systems.

## What's Inside

- **`neptts-eval/`** — pip-installable evaluation package
- **`benchmark/`** — 365 phonologically-designed Nepali sentences + evaluation results
- **`rating-app/`** — Web app for collecting human MOS ratings ([live](https://tts.ampixa.com/rating))
- **`model/`** — Nepali MOS predictor (IndicWav2Vec fine-tuned, Spearman 0.587)
- **`paper/`** — Research paper source
- **`scripts/`** — TTS generation scripts for all systems

## Benchmark Sentences

193 benchmark sentences covering:
- 60+ phonological minimal pairs (aspiration, retroflexion, nasalization, gemination, schwa deletion)
- Question intonation, emotion, contrastive stress
- Homographs, newspaper text, robustness tests
- 160 natural speech references (Chirp2-transcribed YouTube)

## Evaluation Metrics

| Metric | Type | Best For |
|--------|------|----------|
| Human MOS | Human ratings (1-5) | Ground truth quality |
| SCOREQ | Auto-MOS (ONNX) | Quick quality estimate |
| Google Chirp2 CER | ASR round-trip | Intelligibility (best ASR) |
| Meta MMS CER | ASR round-trip | Intelligibility (open source) |
| XLS-R Nepali CER | ASR round-trip | Pronunciation accuracy |
| Whisper CER | ASR round-trip | Baseline intelligibility |

## Key Findings

1. **Automated MOS does not predict human preference for Nepali.** SCOREQ ranked ElevenLabs #1 but humans ranked it #4. Nepali-specific systems (TingTing) are preferred by native speakers.

2. **ASR model selection matters.** Whisper small gives unreliable rankings for Nepali (Spearman 0.38 with other metrics). Chirp2 is far more reliable.

3. **Open-source gap.** Piper (open source) scores 3.47, competitive with cloud services — but below the best commercial systems.

4. **First Nepali MOS predictor.** Fine-tuned IndicWav2Vec achieves Spearman 0.587 with human ratings.

## Links

- **Rate Nepali TTS:** [tts.ampixa.com/rating](https://tts.ampixa.com/rating)
- **Compare voices:** [tts.ampixa.com/rating/voices](https://tts.ampixa.com/rating/voices)
- **Minimal pairs test:** [tts.ampixa.com/rating/pairs](https://tts.ampixa.com/rating/pairs)
- **Dataset:** [HuggingFace](https://huggingface.co/datasets/ampixa/neptts-bench)

## Citation

```bibtex
@article{neptts-bench-2026,
  title={NepTTS-Bench: A Comprehensive Benchmark for Nepali Text-to-Speech Evaluation},
  author={Ampixa},
  year={2026}
}
```

## License

MIT
