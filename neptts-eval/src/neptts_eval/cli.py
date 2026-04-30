"""CLI entry point for neptts-eval."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import click

from . import __version__


@click.command()
@click.option("--tts-cmd", default=None,
              help="Shell command to synthesize. Use {text} for input, {output} for output path. "
                   "Example: \"python my_tts.py '{text}' -o '{output}'\"")
@click.option("--wav-dir", "--wav_dir", default=None, type=click.Path(exists=True),
              help="Directory with pre-generated audio (sent_001.wav, etc.). Alternative to --tts-cmd.")
@click.option("--output", "-o", default="neptts_report.json",
              help="Output JSON report path (default: neptts_report.json)")
@click.option("--system-name", default="my_system",
              help="Name for your system in the report")
@click.option("--whisper-model", default="small",
              help="Whisper model size: tiny, base, small, medium (default: small)")
@click.option("--device", default="cpu",
              help="Device for Whisper: cpu or cuda (default: cpu)")
@click.option("--skip-scoreq", is_flag=True, help="Skip SCOREQ MOS evaluation")
@click.option("--skip-asr", is_flag=True, help="Skip ASR round-trip evaluation")
@click.option("--skip-nepalimos", is_flag=True, help="Skip NepaliMOS evaluation")
@click.option("--nepalimos-ckpt", default=None, type=click.Path(),
              help="Local NepaliMOS checkpoint path (default: download from HF ampixa/neptts-bench).")
@click.option("--verbose", "-v", is_flag=True, help="Print progress")
@click.version_option(version=__version__)
def main(tts_cmd, wav_dir, output, system_name, whisper_model, device,
         skip_scoreq, skip_asr, skip_nepalimos, nepalimos_ckpt, verbose):
    """Evaluate a Nepali TTS system against the NepTTS-Bench benchmark.

    Two modes:

    \b
    1. Provide a TTS command (recommended):
       neptts-eval --tts-cmd "python my_tts.py '{text}' -o '{output}'"

    \b
    2. Provide pre-generated audio:
       neptts-eval --wav-dir ./my_outputs/

    For mode 1, {text} is replaced with the Nepali sentence and {output}
    with the output WAV path. The command is called for each benchmark sentence.
    """
    from .data import load_sentences

    if not tts_cmd and not wav_dir:
        click.echo("Error: provide either --tts-cmd or --wav-dir", err=True)
        raise SystemExit(1)

    # Load sentences
    if verbose:
        click.echo("Loading benchmark sentences...")
    sentences = load_sentences()

    # Get audio files
    if tts_cmd:
        click.echo(f"Generating audio with: {tts_cmd}")
        audio_files = _generate_from_cmd(tts_cmd, sentences, verbose)
        click.echo(f"Generated {len(audio_files)} audio files")
    else:
        from .data import discover_audio_files
        audio_files = discover_audio_files(wav_dir)
        click.echo(f"Found {len(audio_files)} audio files in {wav_dir}")

    matched = {sid: path for sid, path in audio_files.items() if sid in sentences}
    click.echo(f"Matched {len(matched)}/{len(audio_files)} files to benchmark sentences")

    scoreq_results = None
    asr_results = None
    nepalimos_results = None

    # SCOREQ
    if not skip_scoreq:
        click.echo("\nRunning SCOREQ auto-MOS...")
        from .scoreq_eval import evaluate_scoreq
        scoreq_results = evaluate_scoreq(audio_files, verbose=verbose)
        click.echo(f"  SCOREQ MOS: {scoreq_results['avg_mos']:.2f} ({scoreq_results['n_scored']} files)")

    # NepaliMOS (Nepali-specific predictor; system-level rho_human=0.90 in the paper)
    if not skip_nepalimos:
        click.echo("\nRunning NepaliMOS auto-MOS...")
        from .nepalimos_eval import evaluate_nepalimos
        nepalimos_results = evaluate_nepalimos(
            audio_files, verbose=verbose, ckpt_path=nepalimos_ckpt, device=device,
        )
        click.echo(f"  NepaliMOS: {nepalimos_results['avg_mos']:.2f} ({nepalimos_results['n_scored']} files)")

    # ASR round-trip
    if not skip_asr:
        click.echo(f"\nRunning Whisper {whisper_model} ASR round-trip...")
        from .asr_eval import evaluate_whisper
        asr_results = evaluate_whisper(
            matched, sentences, model_size=whisper_model, device=device, verbose=verbose
        )
        click.echo(f"  Whisper CER: {asr_results['avg_cer']:.3f} ({asr_results['n_files']} files)")

    # Generate report
    from .report import generate_report, print_table
    report = generate_report(
        scoreq_results, asr_results, len(audio_files), system_name,
        nepalimos_results=nepalimos_results,
    )

    # Save
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    click.echo(f"\nReport saved to {output}")

    # Print comparison table
    print_table(report)


def _generate_from_cmd(cmd_template: str, sentences: dict, verbose: bool) -> dict[str, Path]:
    """Generate audio by calling a shell command for each sentence."""
    output_dir = Path(tempfile.mkdtemp(prefix="neptts_"))

    # Only benchmark sentences
    bench = {sid: s for sid, s in sentences.items() if not sid.startswith("chirp_")}

    audio_files = {}
    errors = 0

    for i, (sent_id, sent) in enumerate(sorted(bench.items())):
        text = sent.get("text_dev", sent.get("text_devanagari", ""))
        if not text:
            continue

        out_path = output_dir / f"{sent_id}.wav"

        # Replace placeholders
        cmd = cmd_template.replace("{text}", text).replace("{output}", str(out_path))

        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
            )
            if out_path.exists() and out_path.stat().st_size > 100:
                audio_files[sent_id] = out_path
            elif result.returncode != 0:
                errors += 1
                if verbose and errors <= 3:
                    click.echo(f"  Error {sent_id}: {result.stderr[:100]}", err=True)
            else:
                errors += 1
        except subprocess.TimeoutExpired:
            errors += 1
            if verbose:
                click.echo(f"  Timeout {sent_id}", err=True)

        if verbose and (i + 1) % 20 == 0:
            click.echo(f"  {i+1}/{len(bench)} ({len(audio_files)} ok, {errors} errors)")

    click.echo(f"  Generated {len(audio_files)}/{len(bench)}, {errors} errors")
    return audio_files


if __name__ == "__main__":
    main()
