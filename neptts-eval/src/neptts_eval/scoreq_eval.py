"""SCOREQ auto-MOS evaluation."""

import sys
from pathlib import Path


def _patch_torchaudio():
    """Patch torchaudio.load to use soundfile (avoids torchcodec/CUDA issues)."""
    try:
        import soundfile as sf
        import numpy as np
        import torch
        import torchaudio

        def patched_load(filepath, *args, **kwargs):
            data, sr = sf.read(str(filepath), dtype="float32")
            if data.ndim == 1:
                data = data[np.newaxis, :]
            else:
                data = data.T
            return torch.from_numpy(data), sr

        torchaudio.load = patched_load
    except ImportError:
        pass


def evaluate_scoreq(audio_files: dict[str, Path], verbose: bool = False) -> dict:
    """Run SCOREQ on audio files. Returns per-file and aggregate scores."""
    _patch_torchaudio()

    from scoreq import Scoreq

    sq = Scoreq(data_domain="synthetic", mode="nr")

    scores = {}
    errors = 0

    for i, (sent_id, path) in enumerate(sorted(audio_files.items())):
        try:
            score = float(sq.predict(test_path=str(path)))
            scores[sent_id] = round(score, 3)
        except Exception as e:
            errors += 1
            if verbose and errors <= 5:
                print(f"  SCOREQ error {sent_id}: {e}", file=sys.stderr)

        if verbose and (i + 1) % 50 == 0:
            avg = sum(scores.values()) / len(scores) if scores else 0
            print(f"  SCOREQ: {i+1}/{len(audio_files)} — avg: {avg:.3f}", file=sys.stderr)

    avg_mos = sum(scores.values()) / len(scores) if scores else 0

    return {
        "avg_mos": round(avg_mos, 3),
        "n_scored": len(scores),
        "n_errors": errors,
        "per_file": scores,
    }
