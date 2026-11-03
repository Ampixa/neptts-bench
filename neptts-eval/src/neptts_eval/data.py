"""Sentence data loading — from HuggingFace or local cache."""

import json
import re
from pathlib import Path


CACHE_DIR = Path.home() / ".cache" / "neptts-bench"
SENTENCES_CACHE = CACHE_DIR / "sentences.json"
HF_DATASET = "ampixa/neptts-bench"


# Source installs keep the versioned prompt bank independent of the caller's cwd.
SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_SENTENCES = SOURCE_ROOT / "benchmark" / "sentences.json"
LEGACY_SOURCE_SENTENCES = SOURCE_ROOT / "benchmark" / "data" / "sentences_fixed.json"


def _read_sentences(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        sentences = json.load(f)
    return {sentence["sent_id"]: sentence for sentence in sentences}


def load_sentences() -> dict[str, dict]:
    """Load versioned source prompts first, then cache, then Hugging Face."""
    if SOURCE_SENTENCES.is_file():
        return _read_sentences(SOURCE_SENTENCES)
    if SENTENCES_CACHE.is_file():
        return _read_sentences(SENTENCES_CACHE)

    try:
        from datasets import load_dataset
        dataset = load_dataset(HF_DATASET, "sentences", split="train")
        sentences = [dict(row) for row in dataset]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with SENTENCES_CACHE.open("w", encoding="utf-8") as f:
            json.dump(sentences, f, ensure_ascii=False, indent=2)
        return {sentence["sent_id"]: sentence for sentence in sentences}
    except Exception as exc:
        # Preserve the network/dependency/decoding cause if no fallback exists.
        hf_error = exc

    if LEGACY_SOURCE_SENTENCES.is_file():
        return _read_sentences(LEGACY_SOURCE_SENTENCES)

    raise FileNotFoundError(
        f"Could not load benchmark sentences from the source checkout, {SENTENCES_CACHE}, "
        f"or Hugging Face dataset {HF_DATASET} ({type(hf_error).__name__}). "
        "Install from the source checkout with benchmark/sentences.json, "
        "or restore a verified sentences.json cache and check Hugging Face access."
    ) from hf_error


def discover_audio_files(wav_dir: str | Path) -> dict[str, Path]:
    """Discover audio files in a directory, mapping sent_id -> path."""
    wav_dir = Path(wav_dir)
    if not wav_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {wav_dir}")

    files = {}
    for ext in ("*.wav", "*.mp3", "*.flac", "*.webm"):
        for f in wav_dir.glob(ext):
            if re.match(r"sent_\d+", f.stem):
                files[f.stem] = f

    if not files:
        raise FileNotFoundError(
            f"No audio files matching sent_XXX.wav/mp3/flac found in {wav_dir}"
        )

    return dict(sorted(files.items()))
