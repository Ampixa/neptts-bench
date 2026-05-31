#!/usr/bin/env python3
"""Run the Nepali Conformer CTC-BPE ASR checkpoint on TTS outputs.

The checkpoint/assets are expected from the shared Google Drive folder:
Conformer_model_assets/{config.yaml, tokenizer.*, vocab.txt, *.ckpt}.

Output: benchmark/results/asr_roundtrip_conformer.json
"""

import argparse
import json
import os
import re
import shutil
import time
import unicodedata
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ASSETS = ROOT / "Conformer_model_assets"
DEFAULT_CKPT = DEFAULT_ASSETS / "Conformer-CTC-BPE--val_wer=0.2177-epoch=49.ckpt"
DEFAULT_SENTENCES = ROOT / "benchmark" / "sentences.json"
DEFAULT_OUTPUT = ROOT / "benchmark" / "results" / "asr_roundtrip_conformer.json"

BOLNE_AUDIO_ROOT = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs")
DEFAULT_AUDIO_ROOT = (
    BOLNE_AUDIO_ROOT
    if BOLNE_AUDIO_ROOT.exists()
    else ROOT / "benchmark" / "data" / "tts_outputs"
)


def patch_numpy_for_nemo():
    """NeMo 2.1 still references np.sctypes, removed in NumPy 2."""
    import numpy as np

    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "int": [np.int8, np.int16, np.int32, np.int64],
            "uint": [np.uint8, np.uint16, np.uint32, np.uint64],
            "float": [np.float16, np.float32, np.float64],
            "complex": [np.complex64, np.complex128],
            "others": [np.bool_, np.bytes_, np.str_, np.object_],
        }


def ensure_tokenizer_layout(assets_dir: Path) -> None:
    tok_dir = assets_dir / "tokens" / "tokenizer_spe_unigram_v128_max_200"
    tok_dir.mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.model", "tokenizer.vocab", "vocab.txt"):
        src = assets_dir / name
        dst = tok_dir / name
        if src.exists() and not dst.exists():
            shutil.copyfile(src, dst)


def load_sentences(path: Path) -> dict[str, dict]:
    with open(path) as f:
        rows = json.load(f)
    return {
        row["sent_id"]: row
        for row in rows
        if row.get("sent_id") and not row["sent_id"].startswith("chirp_")
    }


def discover_systems(audio_root: Path) -> list[str]:
    systems = []
    for d in sorted(p for p in audio_root.iterdir() if p.is_dir()):
        child_systems = []
        for child in sorted(p for p in d.iterdir() if p.is_dir()):
            if list_audio(child):
                child_systems.append(str(child.relative_to(audio_root)))
        if child_systems:
            systems.extend(child_systems)
        elif list_audio(d):
            systems.append(str(d.relative_to(audio_root)))
    return systems


def list_audio(path: Path) -> list[Path]:
    return sorted(
        p
        for p in list(path.glob("*.wav")) + list(path.glob("*.mp3")) + list(path.glob("*.flac"))
        if not p.name.startswith("._")
    )


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text or "")
    text = re.sub(r"[।!?,;:\.\-\(\)\[\]\"'॥]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (0 if r == h else 1),
            )
        prev = cur
    return prev[-1]


def char_error_rate(ref: str, hyp: str) -> float:
    ref_chars = list(ref.replace(" ", ""))
    hyp_chars = list(hyp.replace(" ", ""))
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return edit_distance(ref_chars, hyp_chars) / len(ref_chars)


def word_error_rate(ref: str, hyp: str) -> float:
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


def hyp_to_text(hyp) -> str:
    if isinstance(hyp, str):
        return hyp
    if hasattr(hyp, "text"):
        return hyp.text
    return str(hyp)


def load_model(ckpt: Path, assets_dir: Path, device: str):
    patch_numpy_for_nemo()

    import torch
    from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE

    ensure_tokenizer_layout(assets_dir)
    cwd = Path.cwd()
    try:
        os.chdir(assets_dir)
        model = EncDecCTCModelBPE.load_from_checkpoint(str(ckpt), map_location="cpu")
    finally:
        os.chdir(cwd)

    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
        model = model.to("cuda")
    model.eval()
    return model


def transcribe_batch(model, paths: list[Path], batch_size: int, num_workers: int) -> list[str]:
    raw = model.transcribe(
        [str(p) for p in paths],
        batch_size=batch_size,
        num_workers=num_workers,
        verbose=False,
    )
    return [hyp_to_text(h) for h in raw]


def evaluate_system(
    model,
    system: str,
    audio_root: Path,
    sentences: dict[str, dict],
    batch_size: int,
    num_workers: int,
) -> dict:
    system_dir = audio_root / system
    files = [p for p in list_audio(system_dir) if p.stem in sentences]
    print(f"\n=== {system} ({len(files)} matched files) ===")

    details = []
    errors = 0
    t0 = time.time()

    for start in range(0, len(files), batch_size):
        batch = files[start:start + batch_size]
        try:
            hyps = transcribe_batch(model, batch, batch_size, num_workers)
        except Exception as exc:
            print(f"  batch error at {start}: {exc}; retrying one by one")
            hyps = []
            for path in batch:
                try:
                    hyps.extend(transcribe_batch(model, [path], 1, num_workers))
                except Exception as item_exc:
                    errors += 1
                    print(f"  ERROR {path.stem}: {item_exc}")
                    hyps.append("")

        for path, hyp_raw in zip(batch, hyps):
            sent = sentences[path.stem]
            ref = normalize_text(sent.get("text_devanagari", ""))
            hyp = normalize_text(hyp_raw)
            cer = char_error_rate(ref, hyp)
            wer = word_error_rate(ref, hyp)
            details.append({
                "sent_id": path.stem,
                "ref": ref,
                "hyp": hyp,
                "cer": round(cer, 4),
                "wer": round(wer, 4),
                "category": sent.get("category", ""),
                "contrast_word": sent.get("contrast_word", ""),
            })

        done = min(start + batch_size, len(files))
        if done == len(files) or done % 40 == 0:
            avg = sum(r["cer"] for r in details) / len(details) if details else 0.0
            rate = done / max(time.time() - t0, 1e-6)
            print(f"  {done}/{len(files)}  avg CER={avg:.3f}  ({rate:.1f} files/s)")

    if not details:
        return {
            "summary": {
                "system": system,
                "num_files": 0,
                "n_errors": errors,
                "avg_cer": None,
                "avg_wer": None,
            },
            "details": [],
        }

    cers = [r["cer"] for r in details]
    wers = [r["wer"] for r in details]
    by_cat = defaultdict(list)
    for row in details:
        by_cat[row["category"] or "other"].append(row["cer"])

    summary = {
        "system": system,
        "num_files": len(details),
        "n_errors": errors,
        "avg_cer": round(sum(cers) / len(cers), 4),
        "median_cer": round(sorted(cers)[len(cers) // 2], 4),
        "max_cer": round(max(cers), 4),
        "pct_cer_gt_1": round(100 * sum(c > 1.0 for c in cers) / len(cers), 2),
        "avg_wer": round(sum(wers) / len(wers), 4),
        "per_category": {
            cat: round(sum(vals) / len(vals), 4)
            for cat, vals in sorted(by_cat.items())
        },
        "worst_5": sorted(details, key=lambda row: -row["cer"])[:5],
    }
    print(
        f"  -> avg CER={summary['avg_cer']:.3f}  "
        f"median={summary['median_cer']:.3f}  avg WER={summary['avg_wer']:.3f}"
    )
    return {"summary": summary, "details": details}


def main():
    parser = argparse.ArgumentParser(description="Evaluate Conformer ASR round-trip CER/WER")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--audio-root", type=Path, default=DEFAULT_AUDIO_ROOT)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--systems", nargs="*", default=None,
                        help="Systems relative to --audio-root. Defaults to all discovered systems.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()

    sentences = load_sentences(args.sentences)
    systems = args.systems or discover_systems(args.audio_root)

    print(f"Loaded {len(sentences)} sentence references from {args.sentences}")
    print(f"Evaluating {len(systems)} systems from {args.audio_root}")
    print(f"Loading Conformer checkpoint: {args.ckpt}")
    model = load_model(args.ckpt.resolve(), args.assets_dir.resolve(), args.device)

    all_results = {}
    grand_t0 = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for system in systems:
        all_results[system] = evaluate_system(
            model, system, args.audio_root, sentences, args.batch_size, args.num_workers
        )
        args.output.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))

    print(f"\nWrote {args.output}")
    print(f"Total time: {(time.time() - grand_t0) / 60:.1f} min")
    print(f"{'System':<28} {'n':>5} {'CER':>8} {'WER':>8} {'med CER':>8}")
    print("-" * 62)
    for system in systems:
        summary = all_results[system]["summary"]
        if summary["avg_cer"] is None:
            print(f"{system:<28} {0:>5} {'--':>8} {'--':>8} {'--':>8}")
        else:
            print(
                f"{system:<28} {summary['num_files']:>5} "
                f"{summary['avg_cer']:>8.3f} {summary['avg_wer']:>8.3f} "
                f"{summary['median_cer']:>8.3f}"
            )


if __name__ == "__main__":
    main()
