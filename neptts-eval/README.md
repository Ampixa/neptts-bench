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

## Private D32 reviewer judge

The owner-operated `d32v2-s60000-full-translit` profile scores audio locally on
CPU with the 33,447,414 parameter DISTILL-32M v2 step 60000 checkpoint. It uses
the full-context FP32 greedy lane calibrated in
[`judge_calibration_v1`](../benchmark/judge_calibration_v1/README.md), with real
recording corpus CER of **7.88% on PV / 13.69% on Fresh**. These are instrument
errors, not a floor to subtract from TTS scores. The model is call-trained and
private; independent reproduction requires owner access until the owner
decides to release weights. This software never downloads or uploads weights
or audio, and does not start a service.

Use the pinned calibration runtime already on the authorized reviewer machine.
The `d32` optional dependencies describe the Python packages; the matching CPU
`k2` build, the original training recipe/icefall source, the checkpoint,
tokenizer and frozen phase1b normalizer snapshot are supplied separately by the
owner. Runtime versions, recipe files, checkpoint, tokenizer and every snapshot
file are verified before use. The icefall checkout must match its pinned
commit and have a clean worktree. The recipe pins include the 21 local Python
modules reachable from the calibrated training/decoder imports. These private
runtime inputs are not bundled in the wheel. The exact
pins are in `src/neptts_eval/d32_contract.json`; that contract and `d32.py` both
participate in the profile code hash. The profile ID also hashes its complete
settings. Use `profile(benchmark)` to inspect public settings without loading
private models or audio.

Check `memory_pressure` on k2 before starting CPU work. Keep private inputs and
temporary extraction under the job directory. For an arena reviewer package,
use the package-binding CLI from the repository root. D32 extraction is
created beside the reviewer package and removed automatically, so keep the
package inside the authorized k2 job directory:

```sh
export PYTHONPATH="$PWD/neptts-eval/src"
export PYTHONDONTWRITEBYTECODE=1
python -m submission_app.offline \
  --profile d32v2-s60000-full-translit \
  --package "$JOB/evaluation-$SUBMISSION_ID.zip" \
  --output "$JOB/result-$SUBMISSION_ID.json" \
  --checkpoint "$D32_CHECKPOINT" --bpe "$D32_BPE" \
  --recipe "$D32_RECIPE" --icefall "$D32_ICEFALL" \
  --normalizer-root "$D32_NORMALIZER"
```

`D32_NORMALIZER` is the phase1b directory containing its pinned
`normalization_manifest.json` and `normalization_snapshot/`. The scorer imports
only the verified copies, so Ubuntu's original absolute paths are not needed.
The legacy score2 policy transliterates Latin hypothesis tokens only when the
reference has no Latin letters. It then applies the frozen suite normalizer
to both sides and counts spaces in CER. This differs from the historical arena
Whisper normalization. Each result includes sentence error counts, sentence
CER/WER, macro means and corpus ratios. Hypotheses, references and audio paths
are omitted from result JSON. Missing or failed inputs abort the report.

For a local prompt bank and WAV directory outside the arena package workflow,
the installed `neptts-eval-d32` command (or `python -m neptts_eval.d32`) takes
the same private path options plus `--wav-dir`, `--sentences` (a local JSON
sentence list), `--benchmark` and `--output`. It selects every `sent_` ID from
that file, prints the observed metadata keys, refuses duplicate IDs and an
existing output, and has no network fallback. Use the arena package workflow
for importable submission reports because it also binds the submission ID,
benchmark and audio hashes.

Audio and generated reports require a job README/provenance record under the
owner's artifact policy. Retain numeric evidence and remove private audio
copies after their consumer finishes, recording the deletion manifest. Never
place weights or submitter audio in Git or Hugging Face.
