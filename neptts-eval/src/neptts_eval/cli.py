"""CLI entry point for neptts-eval."""

import json
import sys

import click

from . import __version__


@click.command()
@click.option("--wav_dir", "--wav-dir", required=True, type=click.Path(exists=True),
              help="Directory containing sent_001.wav, sent_002.wav, etc.")
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
@click.option("--verbose", "-v", is_flag=True, help="Print progress")
@click.version_option(version=__version__)
def main(wav_dir, output, system_name, whisper_model, device, skip_scoreq, skip_asr, verbose):
    """Evaluate a Nepali TTS system against the NepTTS-Bench benchmark.

    Place your TTS audio files (sent_001.wav, sent_002.wav, ...) in a directory
    and run this tool to get SCOREQ MOS and ASR round-trip CER scores,
    compared against 9 baseline systems.
    """
    from .data import discover_audio_files, load_sentences

    # Discover files
    audio_files = discover_audio_files(wav_dir)
    click.echo(f"Found {len(audio_files)} audio files in {wav_dir}")

    # Load sentences
    if verbose:
        click.echo("Loading benchmark sentences...")
    sentences = load_sentences()
    matched = {sid: path for sid, path in audio_files.items() if sid in sentences}
    click.echo(f"Matched {len(matched)}/{len(audio_files)} files to benchmark sentences")

    scoreq_results = None
    asr_results = None

    # SCOREQ
    if not skip_scoreq:
        click.echo("\nRunning SCOREQ auto-MOS...")
        from .scoreq_eval import evaluate_scoreq
        scoreq_results = evaluate_scoreq(audio_files, verbose=verbose)
        click.echo(f"  SCOREQ MOS: {scoreq_results['avg_mos']:.2f} ({scoreq_results['n_scored']} files)")

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
    report = generate_report(scoreq_results, asr_results, len(audio_files), system_name)

    # Save
    with open(output, "w") as f:
        json.dump(report, f, indent=2)
    click.echo(f"\nReport saved to {output}")

    # Print comparison table
    print_table(report)


if __name__ == "__main__":
    main()
