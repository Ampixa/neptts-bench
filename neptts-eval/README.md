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

- **SCOREQ** auto-MOS prediction (no GPU needed, ONNX)
- **Whisper** ASR round-trip CER/WER (intelligibility)
- Comparison against 12 baseline Nepali TTS systems

## Output

JSON report with scores + ranking against baselines:

```
=================================================================
NepTTS-Bench Evaluation Report
=================================================================
  SCOREQ MOS: 3.85
  Whisper CER: 0.142

Rank  System                    SCOREQ   Chirp2      MMS  Whisper
-----------------------------------------------------------------
1     elevenlabs                  4.45    0.106    0.278    0.470
2     tingting_subina             4.27    0.120    0.181    0.514
3     my_tts                      3.85        —        —    0.142 <<
...
```

## Links

- [NepTTS-Bench GitHub](https://github.com/Ampixa/neptts-bench)
- [Dataset on HuggingFace](https://huggingface.co/datasets/ampixa/neptts-bench)
- [Rate Nepali TTS](https://tts.ampixa.com/rating)
