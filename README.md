# NepTTS-Bench

Nepali speech evaluation with fixed prompts, a historical native-listener study, and reviewed ASR-based submissions in the [Ampixa Benchmarks Arena](https://ampixa.benchmarks.cloud/). Human MOS and ASR intelligibility scores answer different questions.

## Historical human study

The checked-in [paper's human-MOS table](paper/neptts_bench.tex) reports **6,962 ratings from 180 contributing, self-declared native Nepali speakers**, across 10 TTS systems and two human baselines. The table below reproduces that study snapshot; it is separate from the arena's 205-prompt submission track. The 95% intervals use 2,000 rater-cluster bootstrap resamples. Raw private rating rows were not re-exported for this documentation update.

| System | Human MOS | 95% CI | Ratings |
|--------|----------:|--------|--------:|
| Natural speech | 3.95 | [3.79, 4.15] | 597 |
| Human (read) | 3.64 | [3.44, 3.82] | 552 |
| TingTing Asmita | 3.55 | [3.27, 3.88] | 651 |
| Piper | 3.50 | [3.31, 3.69] | 515 |
| TingTing Subina | 3.47 | [3.24, 3.79] | 644 |
| ElevenLabs v3 | 3.47 | [3.28, 3.66] | 516 |
| Edge TTS Sagar | 3.38 | [3.02, 3.77] | 609 |
| Edge TTS Hemkala | 3.37 | [3.10, 3.67] | 619 |
| Gemini (Hindi) | 3.34 | [3.15, 3.53] | 514 |
| Gemini Flash | 3.25 | [2.90, 3.68] | 618 |
| TingTing Sambriddhi | 3.20 | [2.91, 3.55] | 617 |
| gTTS | 2.60 | [2.38, 2.83] | 510 |

The top four TTS systems have overlapping intervals; this table does not establish a unique winner. The paper reports low individual-rater agreement (Krippendorff's α = 0.17). Earlier README totals of 164+ listeners / 5,760+ ratings described an older snapshot and are superseded here. [Count and release scopes](neptts-eval/RELEASE_SCOPE.md) record the source, filters and remaining release-manifest gap.

## Source install

Use Python 3.12 and FFmpeg. From a source checkout:

```bash
git clone https://github.com/Ampixa/neptts-bench.git
cd neptts-bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./neptts-eval
neptts-eval --help

# Outputs must match the sent_NNN IDs in benchmark/sentences.json.
# Skip the optional NepaliMOS predictor unless its dependencies are installed.
neptts-eval --wav-dir ./my_outputs/ --system-name "my_tts" --skip-nepalimos
```

A bare `pip install neptts-eval` is not the documented distribution route. See the [evaluator README](neptts-eval/README.md) for optional dependencies and checkpoint choices. The research CLI compares against nine saved TTS baselines; it does not create an arena-approved report. Use the [arena submission workflow](submission_app/README.md) for that track.

## Prompt inventory

[`benchmark/sentences.json`](benchmark/sentences.json) contains **365 records: 205 synthesis prompts and 160 natural-speech references**. The arena requires all 205 `sent_NNN` prompts and excludes the references. These IDs are not consecutive. The earlier “193 sentences” description is obsolete for this file.

The synthesis prompts cover phonology, question intonation, emotion, contrastive stress, homographs, newspaper ambiguities, phrases and robustness. Of these, 149 phonological prompts have 77 pair IDs: **72 complete two-prompt pairs and five singleton probes**. The 160 `chirp_*` entries are transcribed natural references, not extra synthesis prompts. This inventory does not establish the number of clips or ratings in every historical experiment.

## What's inside

- **`neptts-eval/`** — source-installable research evaluator: SCOREQ, Whisper and optional NepaliMOS.
- **`benchmark/`** — prompt inventory, saved metric results and judge-calibration evidence.
- **`submission_app/`** — reviewed arena submissions with fixed coverage and separate judge profiles.
- **`rating-app/`** — historical human-MOS collection application.
- **`model/`** — NepaliMOS training source and [checkpoint documentation](model/README.md).
- **`paper/`** — study methods, results and limitations.
- **`scripts/`** — generation and analysis scripts.

NepaliMOS's same-panel system-level correlation (ρ = 0.90) does not establish ranking quality for new systems. The [saved leave-one-system-out result](benchmark/results/nepalimos_loso.json) has pooled utterance ρ = 0.429 and native-human system-mean ρ = −0.394 (10 systems, p = 0.260). ASR CER is also instrument-dependent; consult each arena judge's real-speech calibration and loanword policy before comparing systems.

## Links

- [Arena](https://ampixa.benchmarks.cloud/)
- [Rate Nepali TTS](https://tts.ampixa.com/rating)
- [Compare voices](https://tts.ampixa.com/rating/voices)
- [Pair task](https://tts.ampixa.com/rating/pairs)
- [Dataset on Hugging Face](https://huggingface.co/datasets/ampixa/neptts-bench)

## Citation

```bibtex
@article{neptts-bench-2026,
  title={NepTTS-Bench: A Comprehensive Benchmark for Nepali Text-to-Speech Evaluation},
  author={Ampixa},
  year={2026}
}
```

## License

Software: MIT. Recordings, provider-generated audio, datasets and model weights retain their source-specific permissions and usage limits; the software licence does not grant blanket redistribution rights.
