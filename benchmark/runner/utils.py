"""Shared utilities for NepTTS-Bench benchmark runner."""

import json
import re
import unicodedata
from pathlib import Path

import numpy as np


def normalize_nepali_text(text: str) -> str:
    """Normalize Nepali text for CER/WER comparison.

    Steps (from asr_roundtrip_protocol.json):
    1. Unicode NFC normalization for Devanagari
    2. Strip all punctuation (Devanagari and Latin)
    3. Normalize whitespace
    4. Lowercase Latin characters
    5. Normalize Devanagari numerals to Arabic numerals
    """
    text = unicodedata.normalize("NFC", text)

    # Strip punctuation (Devanagari purna viram, deergha viram, Latin)
    text = re.sub(r"[।॥,;:!?\.\-\"'\(\)\[\]{}<>…–—/\\@#\$%\^&\*\+\=\|~`]", "", text)

    # Normalize Devanagari numerals to Arabic
    dev_digits = "०१२३४५६७८९"
    for i, d in enumerate(dev_digits):
        text = text.replace(d, str(i))

    # Lowercase Latin characters
    text = text.lower()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def load_audio(path: str | Path) -> tuple[np.ndarray, int]:
    """Load audio file, return (samples, sample_rate). Converts to mono float32."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def compute_snr(audio: np.ndarray, sr: int = 16000, frame_ms: int = 30) -> float:
    """Estimate signal-to-noise ratio in dB using frame energy VAD."""
    frame_len = int(sr * frame_ms / 1000)
    n_frames = len(audio) // frame_len
    if n_frames == 0:
        return 0.0

    frames = audio[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = np.mean(frames**2, axis=1)

    if len(energies) == 0:
        return 0.0

    # Use energy threshold to separate signal vs noise frames
    threshold = np.median(energies) * 0.1
    signal_frames = energies[energies > threshold]
    noise_frames = energies[energies <= threshold]

    if len(noise_frames) == 0 or len(signal_frames) == 0:
        return 40.0  # Very clean or all silence

    signal_power = np.mean(signal_frames)
    noise_power = np.mean(noise_frames)

    if noise_power < 1e-10:
        return 60.0

    return float(10 * np.log10(signal_power / noise_power))


def detect_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> dict:
    """Detect leading and trailing silence in milliseconds."""
    threshold = 10 ** (threshold_db / 20)
    abs_audio = np.abs(audio)

    # Leading silence
    leading_ms = 0.0
    for i, s in enumerate(abs_audio):
        if s > threshold:
            leading_ms = i / sr * 1000
            break
    else:
        leading_ms = len(audio) / sr * 1000

    # Trailing silence
    trailing_ms = 0.0
    for i in range(len(abs_audio) - 1, -1, -1):
        if abs_audio[i] > threshold:
            trailing_ms = (len(abs_audio) - 1 - i) / sr * 1000
            break

    return {
        "leading_silence_ms": round(leading_ms, 1),
        "trailing_silence_ms": round(trailing_ms, 1),
        "total_duration_ms": round(len(audio) / sr * 1000, 1),
    }


def load_json(path: str | Path) -> dict | list:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str | Path, data: dict | list) -> None:
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).resolve().parent.parent.parent
