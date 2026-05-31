#!/usr/bin/env python3
"""Generate audio from a public Nepali VITS checkpoint for unseen-system checks."""

import argparse
import json
import re
from pathlib import Path

import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SENTENCES = ROOT / "benchmark" / "sentences.json"
DEFAULT_AUDIO_ROOT = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs")
DEFAULT_MODEL_ID = "atul10/nepali_male_v1"


def sent_sort_key(item):
    match = re.fullmatch(r"sent_(\d+)", item["sent_id"])
    return int(match.group(1)) if match else 10**9


def load_sentences(path: Path):
    data = json.loads(path.read_text())
    sentences = [
        row
        for row in data
        if re.fullmatch(r"sent_\d+", row["sent_id"]) and row.get("text_devanagari")
    ]
    return sorted(sentences, key=sent_sort_key)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--sentences", type=Path, default=DEFAULT_SENTENCES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_AUDIO_ROOT / "nepali_male_v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    sentences = load_sentences(args.sentences)
    if args.limit:
        sentences = sentences[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Model: {args.model_id}")
    print(f"Device: {args.device}")
    print(f"Sentences: {len(sentences)}")
    print(f"Output: {args.output_dir}")

    model = VitsModel.from_pretrained(args.model_id).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model.eval()

    written = 0
    skipped = 0
    with torch.no_grad():
        for idx, sent in enumerate(sentences, start=1):
            out_path = args.output_dir / f"{sent['sent_id']}.wav"
            if out_path.exists():
                skipped += 1
                continue

            inputs = tokenizer(sent["text_devanagari"], return_tensors="pt").to(args.device)
            output = model(**inputs).waveform
            waveform = output.squeeze().detach().cpu().numpy()
            sf.write(out_path, waveform, model.config.sampling_rate)
            written += 1

            if idx % 20 == 0 or idx == len(sentences):
                print(f"  {idx}/{len(sentences)} written={written} skipped={skipped}", flush=True)

    total = len(list(args.output_dir.glob("*.wav")))
    print(f"Done: wrote={written} skipped={skipped} total_wav={total}")


if __name__ == "__main__":
    main()
