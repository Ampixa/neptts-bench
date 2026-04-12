#!/usr/bin/env python3
"""Generate TTS audio using ElevenLabs API (multilingual v2)."""

import json
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SENTENCES_PATH = ROOT / "benchmark" / "data" / "sentences_fixed.json"
OUTPUT_DIR = ROOT / "benchmark" / "data" / "tts_outputs" / "elevenlabs"

API_KEY = "sk_062ff164c08032e68ee53cca613287f3afe2b8b0453a1e6c"
# Sarah - mature, clear voice. Good for benchmark.
VOICE_ID = "EXAVITQu4vr4xnSDxMaL"
MODEL_ID = "eleven_multilingual_v2"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(SENTENCES_PATH) as f:
        sentences = [s for s in json.load(f) if not s["sent_id"].startswith("chirp")]

    headers = {
        "xi-api-key": API_KEY,
        "Content-Type": "application/json",
    }

    total = len(sentences)
    success = 0
    errors = []

    for i, s in enumerate(sentences):
        sent_id = s["sent_id"]
        text = s["text_devanagari"]
        out_file = OUTPUT_DIR / f"{sent_id}.mp3"

        if out_file.exists() and out_file.stat().st_size > 100:
            success += 1
            continue

        try:
            resp = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
                headers=headers,
                json={"text": text, "model_id": MODEL_ID},
                timeout=30,
            )
            resp.raise_for_status()
            out_file.write_bytes(resp.content)

            if out_file.stat().st_size > 100:
                success += 1
            else:
                errors.append(sent_id)

        except Exception as e:
            errors.append(sent_id)
            print(f"  ERROR {sent_id}: {e}")

        # Rate limit for free tier
        time.sleep(1.0)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{total} ({success} ok, {len(errors)} errors)")

    print(f"\nDone: {success}/{total}, {len(errors)} errors")
    if errors:
        print(f"Failed: {errors[:10]}")


if __name__ == "__main__":
    main()
