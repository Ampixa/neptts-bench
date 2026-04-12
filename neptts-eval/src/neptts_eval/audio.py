"""Audio loading utilities."""

from pathlib import Path

import numpy as np
import soundfile as sf


def load_audio(path: str | Path, target_sr: int = 16000) -> tuple[np.ndarray, int]:
    """Load audio file as mono float32. Resamples if needed."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        # Simple resampling via interpolation
        duration = len(audio) / sr
        n_samples = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, n_samples)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)
        sr = target_sr

    return audio, sr
