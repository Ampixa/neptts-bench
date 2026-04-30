# neptts-eval

Evaluate any Nepali TTS system against the [NepTTS-Bench](https://github.com/Ampixa/neptts-bench) benchmark.

## Install

```bash
pip install neptts-eval
```

## Usage

### Option 1: Point to your TTS command (recommended)

```bash
neptts-eval --tts-cmd "python my_tts.py '{text}' -o '{output}'" --system-name "my_tts"
```

The tool calls your command for each of 193 benchmark sentences, replacing `{text}` with the Nepali sentence and `{output}` with the output WAV path.

### Option 2: Python decorator

```python
from neptts_eval import benchmark

@benchmark(system_name="my_tts")
def synthesize(text: str) -> bytes:
    # Your TTS code here
    return audio_bytes  # WAV/MP3 bytes
```

### Option 3: Python API

```python
from neptts_eval import generate_benchmark_audio
from neptts_eval.data import load_sentences
from neptts_eval.scoreq_eval import evaluate_scoreq

sentences = load_sentences()
audio_files = generate_benchmark_audio(my_tts_function, sentences)
results = evaluate_scoreq(audio_files)
print(f"SCOREQ MOS: {results['avg_mos']:.2f}")
```

### Option 4: Pre-generated audio directory

```bash
neptts-eval --wav-dir ./my_outputs/ --system-name "my_tts"
```

Files should be named `sent_001.wav`, `sent_002.wav`, etc.

## What it measures

- **NepaliMOS** auto-MOS prediction (Nepali-specific, fine-tuned from IndicWav2Vec; system-level rho=0.90 vs human MOS in the paper, vs SCOREQ's 0.40)
- **SCOREQ** auto-MOS prediction (English-trained baseline, no GPU needed)
- **Whisper** ASR round-trip CER/WER (intelligibility)
- Comparison against 9 baseline Nepali TTS systems

NepaliMOS requires the optional `nepalimos` extra:

```bash
pip install "neptts-eval[nepalimos]"
```

The first run downloads the public head-only checkpoint (~800 KB) plus the IndicWav2Vec base backbone (~360 MB) from HuggingFace. To reproduce the paper's full fine-tuned predictor (the one whose system-level rho=0.90 was reported), pass the full checkpoint via `--nepalimos-ckpt /path/to/neptts_mos_best.pt` or the `NEPALIMOS_CKPT` env var.

## Output

JSON report with scores + ranking against baselines (sorted by NepaliMOS):

```
==============================================================================
NepTTS-Bench Evaluation Report
==============================================================================
  NepaliMOS:   3.47
  SCOREQ MOS:  3.85
  Whisper CER: 0.142

Rank  System                     NepMOS  SCOREQ   Chirp2      MMS  Whisper
------------------------------------------------------------------------------
1     piper                        3.52    3.49    0.186    0.291    0.471
2     my_tts                       3.47    3.85       -        -    0.142 <<
3     elevenlabs                   3.46    4.45    0.106    0.278    0.470
...
```

## Links

- [NepTTS-Bench GitHub](https://github.com/Ampixa/neptts-bench)
- [Dataset on HuggingFace](https://huggingface.co/datasets/ampixa/neptts-bench)
- [Rate Nepali TTS](https://tts.ampixa.com/rating)
