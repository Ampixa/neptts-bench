"""Generate benchmark audio from a user-provided TTS function."""

import io
import sys
import tempfile
import wave
from pathlib import Path
from typing import Callable, Union


def generate_benchmark_audio(
    tts_fn: Callable[[str], Union[bytes, str, Path]],
    sentences: dict[str, dict],
    output_dir: Path | None = None,
    verbose: bool = False,
) -> dict[str, Path]:
    """Call user's TTS function on all benchmark sentences.

    Args:
        tts_fn: Function that takes Nepali text and returns either:
            - bytes (raw audio: WAV, MP3, etc.)
            - str or Path (path to generated audio file)
        sentences: Dict of sent_id -> sentence info (from load_sentences())
        output_dir: Where to save generated audio. Uses temp dir if None.
        verbose: Print progress.

    Returns:
        Dict of sent_id -> Path to generated audio file.
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="neptts_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Only benchmark sentences (not chirp)
    bench_sents = {
        sid: s for sid, s in sentences.items()
        if not sid.startswith("chirp_")
    }

    audio_files = {}
    errors = 0
    total = len(bench_sents)

    for i, (sent_id, sent) in enumerate(sorted(bench_sents.items())):
        text = sent.get("text_dev", sent.get("text_devanagari", ""))
        if not text:
            continue

        out_path = output_dir / f"{sent_id}.wav"
        if out_path.exists() and out_path.stat().st_size > 100:
            audio_files[sent_id] = out_path
            continue

        try:
            result = tts_fn(text)

            if isinstance(result, (str, Path)):
                # User returned a file path
                result_path = Path(result)
                if result_path.exists():
                    import shutil
                    shutil.copy2(str(result_path), str(out_path))
                else:
                    raise FileNotFoundError(f"TTS returned path that doesn't exist: {result}")
            elif isinstance(result, bytes):
                # User returned raw audio bytes
                out_path.write_bytes(result)
            else:
                raise TypeError(f"TTS function must return bytes or Path, got {type(result)}")

            if out_path.exists() and out_path.stat().st_size > 100:
                audio_files[sent_id] = out_path
            else:
                errors += 1

        except Exception as e:
            errors += 1
            if verbose or errors <= 3:
                print(f"  Error {sent_id}: {e}", file=sys.stderr)

        if verbose and (i + 1) % 20 == 0:
            print(f"  Generated {i+1}/{total} ({len(audio_files)} ok, {errors} errors)", file=sys.stderr)

    if verbose:
        print(f"  Done: {len(audio_files)}/{total} generated, {errors} errors", file=sys.stderr)

    return audio_files


def benchmark(fn=None, *, system_name="my_system", output="neptts_report.json",
              skip_scoreq=False, skip_asr=False, verbose=True):
    """Decorator to benchmark a TTS function.

    Usage:
        from neptts_eval import benchmark

        @benchmark
        def my_tts(text: str) -> bytes:
            # your TTS code
            return audio_bytes

    Or with options:
        @benchmark(system_name="my_awesome_tts", verbose=True)
        def my_tts(text: str) -> bytes:
            ...
    """
    def decorator(tts_fn):
        import json
        from .data import load_sentences
        from .report import generate_report, print_table

        print(f"NepTTS-Bench: Evaluating '{system_name}'")
        print(f"Loading benchmark sentences...")
        sentences = load_sentences()

        print(f"Generating audio for {len([s for s in sentences if not s.startswith('chirp_')])} sentences...")
        audio_files = generate_benchmark_audio(tts_fn, sentences, verbose=verbose)
        print(f"Generated {len(audio_files)} audio files")

        scoreq_results = None
        asr_results = None

        if not skip_scoreq:
            print("\nRunning SCOREQ auto-MOS...")
            from .scoreq_eval import evaluate_scoreq
            scoreq_results = evaluate_scoreq(audio_files, verbose=verbose)
            print(f"  SCOREQ MOS: {scoreq_results['avg_mos']:.2f}")

        if not skip_asr:
            print("\nRunning Whisper ASR round-trip...")
            from .asr_eval import evaluate_whisper
            asr_results = evaluate_whisper(audio_files, sentences, verbose=verbose)
            print(f"  Whisper CER: {asr_results['avg_cer']:.3f}")

        report = generate_report(scoreq_results, asr_results, len(audio_files), system_name)

        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output}")
        print_table(report)

        return tts_fn

    if fn is not None:
        return decorator(fn)
    return decorator
