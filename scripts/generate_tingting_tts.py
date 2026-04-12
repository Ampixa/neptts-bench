#!/usr/bin/env python3
"""Generate TTS audio using TingTing.io API (Nepali-specific TTS).

Two voices: np_ashmita (female), np_sambriddhi (female)
"""

import json
import os
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
SENTENCES_PATH = ROOT / "benchmark" / "data" / "sentences_fixed.json"
OUTPUT_BASE = ROOT / "benchmark" / "data" / "tts_outputs"

API_URL = "https://app.tingting.io/api/test-voice/"
AUTH_TOKEN = "Bearer WSpQIy3vL0yX1ngUS2iLUETS2IxoQcnAKRLdPg1ljLfkYXFPh4jsHosAxz0a5pWi"
HEADERS = {
    "accept": "application/json",
    "authorization": AUTH_TOKEN,
    "content-type": "application/json",
    "origin": "https://www.tingting.io",
    "referer": "https://www.tingting.io/",
}

VOICES = {
    "tingting_asmita": "np_ashmita",
    "tingting_sambriddhi": "np_sambriddhi",
    "tingting_subina": "np_subina",
}


def generate_for_voice(voice_key, voice_input, sentences):
    out_dir = OUTPUT_BASE / voice_key
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(sentences)
    success = 0
    errors = []

    for i, s in enumerate(sentences):
        sent_id = s["sent_id"]
        text = s["text_devanagari"]
        out_file = out_dir / f"{sent_id}.mp3"

        if out_file.exists() and out_file.stat().st_size > 100:
            success += 1
            continue

        try:
            resp = requests.post(
                API_URL,
                headers=HEADERS,
                json={"voice_input": voice_input, "message": text},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            audio_url = data["response"]["audio_url"]

            # Download audio
            audio_resp = requests.get(audio_url, timeout=30)
            audio_resp.raise_for_status()
            out_file.write_bytes(audio_resp.content)

            if out_file.stat().st_size > 100:
                success += 1
            else:
                errors.append(sent_id)
                print(f"  EMPTY: {sent_id}")

        except Exception as e:
            errors.append(sent_id)
            print(f"  ERROR {sent_id}: {e}")

        # Rate limit
        time.sleep(0.5)

        if (i + 1) % 20 == 0:
            print(f"  {voice_key}: {i+1}/{total} ({success} ok, {len(errors)} errors)")

    print(f"  {voice_key}: Done — {success}/{total}, {len(errors)} errors")
    return success, errors


def main():
    with open(SENTENCES_PATH) as f:
        all_sents = json.load(f)

    # Only benchmark sentences, not chirp
    sentences = [s for s in all_sents if not s["sent_id"].startswith("chirp")]
    print(f"Generating for {len(sentences)} sentences")

    for voice_key, voice_input in VOICES.items():
        print(f"\n=== {voice_key} ({voice_input}) ===")
        generate_for_voice(voice_key, voice_input, sentences)


if __name__ == "__main__":
    main()
