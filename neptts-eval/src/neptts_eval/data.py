"""Sentence data loading — from HuggingFace or local cache."""

import json
import re
from pathlib import Path


CACHE_DIR = Path.home() / ".cache" / "neptts-bench"
SENTENCES_CACHE = CACHE_DIR / "sentences.json"
HF_DATASET = "ampixa/neptts-bench"


def load_sentences() -> dict[str, dict]:
    """Load benchmark sentences. Downloads from HF on first run."""
    # Try cache first
    if SENTENCES_CACHE.exists():
        with open(SENTENCES_CACHE) as f:
            sents = json.load(f)
        return {s["sent_id"]: s for s in sents}

    # Try HuggingFace
    try:
        from datasets import load_dataset
        ds = load_dataset(HF_DATASET, "sentences", split="train")
        sents = [dict(row) for row in ds]
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(SENTENCES_CACHE, "w", encoding="utf-8") as f:
            json.dump(sents, f, ensure_ascii=False, indent=2)
        return {s["sent_id"]: s for s in sents}
    except Exception:
        pass

    # Try local fallback (if run from repo)
    for p in [
        Path("benchmark/data/sentences_fixed.json"),
        Path(__file__).parent.parent.parent.parent / "benchmark" / "data" / "sentences_fixed.json",
    ]:
        if p.exists():
            with open(p) as f:
                sents = json.load(f)
            return {s["sent_id"]: s for s in sents}

    raise FileNotFoundError(
        "Could not load benchmark sentences. Run 'pip install datasets' and ensure internet access, "
        "or place sentences_fixed.json in ~/.cache/neptts-bench/"
    )


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
