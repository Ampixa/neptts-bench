"""Blind playback copies: one format, one loudness, no identifying metadata.

Submissions arrive at 16-48 kHz with different levels. Served as-is, the sample
rate, file size, header and loudness would let a listener (or their browser's
dev tools) recognise a system. Every clip a listener hears is therefore:

- mono, resampled to 24 kHz, PCM16 WAV written by one encoder (identical headers);
- loudness-matched to the same active-speech RMS level (-23 dBFS), peak-limited;
- cached outside the upload directory and served under an opaque battle token.

Upsampling a 16 kHz system to 24 kHz adds no content above 8 kHz. Real
bandwidth differences remain audible on purpose: they are part of quality.
"""

import hashlib
import math
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

RATE = 24000
TARGET_DBFS = -23.0
PEAK = 0.97
FRAME = 480  # 20 ms at 24 kHz


def _load(path):
    from scipy.signal import resample_poly

    samples, rate = sf.read(path, dtype="float64", always_2d=True)
    samples = samples.mean(axis=1)
    if rate != RATE:
        g = math.gcd(rate, RATE)
        samples = resample_poly(samples, RATE // g, rate // g)
    return samples


def active_rms(samples):
    """RMS over frames within 40 dB of the loudest frame, so pauses don't count."""
    usable = len(samples) // FRAME * FRAME
    if usable == 0:
        return float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
    frames = samples[:usable].reshape(-1, FRAME)
    energy = np.mean(frames**2, axis=1)
    if energy.max() <= 0:
        return 0.0
    active = energy[energy >= energy.max() * 1e-4]
    return float(np.sqrt(active.mean()))


def normalize(samples):
    rms = active_rms(samples)
    if rms > 0:
        samples = samples * (10 ** (TARGET_DBFS / 20) / rms)
    peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
    if peak > PEAK:
        samples = samples * (PEAK / peak)
    return samples


def degrade(samples, seed):
    """Quality-check clip: noise at 0 dB SNR plus 5-bit quantisation.

    A listener who prefers this over the clean original is not listening.
    """
    rng = np.random.default_rng(seed)
    rms = active_rms(samples) or 1e-3
    noisy = samples + rng.normal(0, rms, len(samples))
    levels = 2**4
    return np.round(np.clip(noisy, -1, 1) * levels) / levels


def playback(root: Path, sid: str, sent_id: str, degraded=False):
    """Return the cached blind copy, creating it on first use. Duration in seconds too."""
    name = f"{sent_id}.degraded.wav" if degraded else f"{sent_id}.wav"
    target = root / "arena_cache" / sid / name
    if not target.exists():
        samples = _load(root / "uploads" / sid / f"{sent_id}.wav")
        if degraded:
            seed = int.from_bytes(hashlib.sha256(f"{sid}/{sent_id}".encode()).digest()[:4], "big")
            samples = degrade(normalize(samples), seed)
        samples = normalize(samples)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_name(f".{uuid.uuid4().hex}.tmp")
        sf.write(temp, samples, RATE, format="WAV", subtype="PCM_16")
        temp.replace(target)
    return target, sf.info(target).duration
