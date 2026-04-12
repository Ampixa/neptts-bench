"""ASR round-trip evaluation using Whisper."""

import sys
from collections import defaultdict
from pathlib import Path

from .metrics import char_error_rate, word_error_rate
from .normalize import normalize_nepali_text


def evaluate_whisper(
    audio_files: dict[str, Path],
    sentences: dict[str, dict],
    model_size: str = "small",
    device: str = "cpu",
    verbose: bool = False,
) -> dict:
    """Run Whisper ASR round-trip on audio files.

    Returns per-file CER/WER and aggregates by category.
    """
    import whisper

    if verbose:
        print(f"  Loading Whisper {model_size}...", file=sys.stderr)

    model = whisper.load_model(model_size, device="cpu")
    if device == "cuda":
        import torch
        model = model.to("cuda").float()

    results = {}
    cat_cers = defaultdict(list)
    errors = 0

    for i, (sent_id, path) in enumerate(sorted(audio_files.items())):
        sent = sentences.get(sent_id)
        if not sent:
            continue

        ref_text = sent.get("text_devanagari", "")
        if not ref_text:
            continue

        try:
            result = model.transcribe(
                str(path), language="ne", task="transcribe", fp16=False
            )
            hyp_text = result.get("text", "").strip()

            ref_norm = normalize_nepali_text(ref_text)
            hyp_norm = normalize_nepali_text(hyp_text)

            cer = char_error_rate(ref_norm, hyp_norm)
            wer = word_error_rate(ref_norm, hyp_norm)

            results[sent_id] = {"cer": round(cer, 4), "wer": round(wer, 4)}

            category = sent.get("category", "other")
            if category:
                cat_cers[category].append(cer)

        except Exception as e:
            errors += 1
            if verbose and errors <= 5:
                print(f"  Whisper error {sent_id}: {e}", file=sys.stderr)

        if verbose and (i + 1) % 20 == 0:
            avg = sum(r["cer"] for r in results.values()) / len(results) if results else 0
            print(f"  Whisper: {i+1}/{len(audio_files)} — avg CER: {avg:.3f}", file=sys.stderr)

    avg_cer = sum(r["cer"] for r in results.values()) / len(results) if results else 0
    avg_wer = sum(r["wer"] for r in results.values()) / len(results) if results else 0

    per_category = {
        cat: round(sum(cers) / len(cers), 4)
        for cat, cers in sorted(cat_cers.items())
    }

    return {
        "avg_cer": round(avg_cer, 4),
        "avg_wer": round(avg_wer, 4),
        "n_files": len(results),
        "n_errors": errors,
        "per_file": results,
        "per_category": per_category,
    }
