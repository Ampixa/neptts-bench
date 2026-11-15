"""Instant signal checks for donated recordings, run before anything is stored.

The browser recorder sends 16-bit mono WAV at the device rate. Here we reject
what no reviewer should have to hear: wrong format, too short or long, silent,
too quiet, clipped, or too noisy. Whether the sentence was actually read is the
ASR check's job, done later by the worker.
"""

import io

import numpy as np
import soundfile as sf

from .arena_audio import active_rms

MIN_RATE, MAX_RATE = 16000, 48000
MIN_SECONDS, MAX_SECONDS = 1.0, 20.0
MAX_BYTES = 4 * 1024 * 1024
MIN_LEVEL_DB = -40.0
MIN_SNR_DB = 15.0
MAX_CLIPPED = 0.001
FRAME_SECONDS = 0.02
NOISE_FRAMES = 5

MESSAGES = {
    "format": "Upload a mono 16-bit WAV recording.",
    "rate": "Sample rate must be between 16 and 48 kHz.",
    "too_short": "The recording is shorter than one second.",
    "too_long": "The recording is longer than 20 seconds.",
    "silent": "No speech was detected. Check that the right microphone is selected.",
    "too_quiet": "The recording is too quiet. Move closer to the microphone.",
    "clipped": "The recording is distorted. Move a little further from the microphone.",
    "noisy": "There is too much background noise. Try a quieter room.",
}


def db(x):
    return 20 * np.log10(max(x, 1e-10))


def measure(data: bytes):
    """Return (metrics, problems). Problems are keys of MESSAGES, empty when usable."""
    if len(data) > MAX_BYTES:
        return {}, ["too_long"]
    try:
        info = sf.info(io.BytesIO(data))
        samples, rate = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    except (RuntimeError, ValueError, TypeError):
        return {}, ["format"]
    if info.format != "WAV" or info.subtype != "PCM_16" or samples.shape[1] != 1:
        return {}, ["format"]
    samples = samples[:, 0]
    seconds = len(samples) / rate
    metrics = {"sample_rate": rate, "seconds": round(seconds, 3)}
    if not MIN_RATE <= rate <= MAX_RATE:
        return metrics, ["rate"]
    if seconds < MIN_SECONDS:
        return metrics, ["too_short"]
    if seconds > MAX_SECONDS:
        return metrics, ["too_long"]
    frame = int(rate * FRAME_SECONDS)
    frames = samples[: len(samples) // frame * frame].reshape(-1, frame)
    energy = np.sqrt(np.mean(frames**2, axis=1))
    if energy.max() < 1e-4:
        return metrics, ["silent"]
    level = db(active_rms(samples))
    # Noise floor: the quietest 100 ms (five 20 ms frames). One short pause anywhere is
    # enough; a percentile would mistake speech for noise in fluent, pause-light reading.
    noise = db(float(np.sqrt(np.mean(np.sort(energy**2)[:NOISE_FRAMES]))))
    clipped = float(np.mean(np.abs(samples) >= 0.999))
    metrics.update(level_dbfs=round(level, 1), noise_dbfs=round(noise, 1),
                   snr_db=round(level - noise, 1), clipped=round(clipped, 5))
    problems = []
    if level < MIN_LEVEL_DB:
        problems.append("too_quiet")
    if clipped > MAX_CLIPPED:
        problems.append("clipped")
    if level - noise < MIN_SNR_DB:
        problems.append("noisy")
    return metrics, problems
