# neptts-eval

Source-installable research evaluator for [NepTTS-Bench](https://github.com/Ampixa/neptts-bench). The CLI produces a local report with SCOREQ, Whisper CER/WER and optional NepaliMOS, compared with nine saved TTS baselines. An arena submission uses the separate [reviewer workflow](../submission_app/README.md).

## Install from source

Use Python 3.12 and install FFmpeg for audio decoding:

```bash
git clone https://github.com/Ampixa/neptts-bench.git
cd neptts-bench
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ./neptts-eval
neptts-eval --help
```

A bare PyPI package name is not the supported install route documented here. Keep the checkout and record `git rev-parse HEAD` with results. The source loader can read the checked-in `benchmark/sentences.json` from another working directory after this local install. The versioned source prompt bank takes precedence over an existing `~/.cache/neptts-bench/sentences.json`; an unrelated working directory cannot replace it. Outside a source checkout, the loader uses that cache, then Hugging Face when no cache exists. A failed download retains its cause in the final error if no legacy source fallback exists.

## Usage

For pre-generated audio:

```bash
neptts-eval --wav-dir ./my_outputs/ --system-name "my_tts" --skip-nepalimos
```

Use the exact `sent_NNN.wav` IDs in [`benchmark/sentences.json`](../benchmark/sentences.json); the 205 synthesis IDs are not consecutive. The file also contains 160 natural references, for 365 total records. The command-generation mode excludes those references:

```bash
neptts-eval --tts-cmd "python my_tts.py '{text}' -o '{output}'" \
  --system-name "my_tts" --skip-nepalimos
```

The CLI invokes the supplied command for each synthesis prompt. This research CLI does not enforce the arena's full-coverage or generation-declaration contract. [Release scopes](RELEASE_SCOPE.md) distinguish current prompts from historical study counts.

## Metrics and checkpoint choice

- **SCOREQ:** no-reference auto-MOS baseline; its score is not a native-listener rating.
- **Whisper:** small model by default, Nepali transcription and normalized CER/WER. Judge choice and normalization affect rankings; see the arena's calibrated judge profiles.
- **NepaliMOS (optional):** Nepali-trained MOS predictor. Same-panel system-level ρ = 0.90 does not establish held-out system ranking. The saved LOSO artifact has pooled utterance ρ = 0.429 and native-human system-mean ρ = −0.394 (10 systems, p = 0.260).

The base install omits NepaliMOS dependencies, so pass `--skip-nepalimos` or install the extra from the repository root:

```bash
python -m pip install -e './neptts-eval[nepalimos]'
neptts-eval --wav-dir ./my_outputs/ --system-name "my_tts"
```

The **actual default is `model/neptts_mos_v9_best.pt`**, a head-only state (~791 KB), from the Hugging Face **dataset** `ampixa/neptts-bench`. Inference also downloads the `SYSPIN/IndicMOS` vanilla IndicWav2Vec backbone. Its scale differs from the full-model baselines in `baselines.json`; that comparison is not a calibration check.

The dataset publicly lists a full-state candidate, `model/neptts_mos_best.pt` (~381 MB). Its public tensor/config compatibility and score calibration have not been verified in this change. To select an independently verified local checkpoint:

```bash
neptts-eval --wav-dir ./my_outputs/ \
  --nepalimos-ckpt /path/to/verified_checkpoint.pt
```

`--nepalimos-ckpt` takes precedence over `NEPALIMOS_CKPT`, followed by the v9 default. See the [model README](../model/README.md) for pinned metadata, hashes and reproduction limits. This change corrects documentation; it does not switch defaults or claim that the public full-state candidate reproduces the paper.

## Output and comparison limits

The default output is `neptts_report.json`, containing per-file metrics, aggregates and saved-baseline comparisons. Use `--output PATH` to change it. The printed table sorts by NepaliMOS when available and SCOREQ otherwise; scores from different checkpoint/judge profiles need separate interpretation. Neither auto-MOS nor ASR CER alone establishes overall TTS quality.

## Links

- [Repository](https://github.com/Ampixa/neptts-bench)
- [Dataset on Hugging Face](https://huggingface.co/datasets/ampixa/neptts-bench)
- [Human study and current counts](../README.md)
