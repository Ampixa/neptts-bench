#!/usr/bin/env python3
"""Generate TTS audio for all benchmark sentences using multiple systems.

Systems:
  - edge_tts: Microsoft Edge TTS (Hemkala + Sagar voices)
  - gtts: Google Translate TTS
  - mms_tts: Meta MMS-TTS (facebook/mms-tts-npi)
  - gemini: Gemini 2.5 Flash TTS

Usage:
    python scripts/generate_tts.py --system edge_tts
    python scripts/generate_tts.py --system all
"""

import argparse
import asyncio
import json
import os
import sqlite3
import struct
import sys
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "benchmark" / "data" / "hetzner_recordings" / "recordings.db"
OUTPUT_BASE = ROOT / "benchmark" / "data" / "tts_outputs"


def load_sentences():
    """Load sentences from DB (only the original 164 that have the current/new text)."""
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    c = db.cursor()
    # Get sentences that are the "current" versions (sent_001 to sent_174)
    # Skip sent_175+ which are old-text copies for recording remapping
    c.execute("SELECT sent_id, text_dev, text_roman, category, contrast_word FROM sentences WHERE CAST(REPLACE(sent_id, 'sent_', '') AS INTEGER) <= 174 ORDER BY sent_id")
    sentences = [dict(r) for r in c.fetchall()]
    db.close()
    return sentences


def generate_edge_tts(sentences, output_dir):
    """Generate with Edge TTS (both voices)."""
    import edge_tts

    for voice_name, voice_id in [("hemkala", "ne-NP-HemkalaNeural"), ("sagar", "ne-NP-SagarNeural")]:
        voice_dir = output_dir / voice_name
        voice_dir.mkdir(parents=True, exist_ok=True)

        async def _generate():
            for i, sent in enumerate(sentences):
                out_path = voice_dir / f"{sent['sent_id']}.mp3"
                if out_path.exists():
                    continue
                try:
                    communicate = edge_tts.Communicate(sent["text_dev"], voice_id)
                    await communicate.save(str(out_path))
                except Exception as e:
                    print(f"  ERROR {sent['sent_id']}: {e}")
                if (i + 1) % 20 == 0:
                    print(f"  {voice_name}: {i+1}/{len(sentences)}")

        asyncio.run(_generate())
        count = len(list(voice_dir.glob("*.mp3")))
        print(f"  edge_tts/{voice_name}: {count} files")


def generate_gtts(sentences, output_dir):
    """Generate with Google Translate TTS."""
    from gtts import gTTS

    output_dir.mkdir(parents=True, exist_ok=True)
    for i, sent in enumerate(sentences):
        out_path = output_dir / f"{sent['sent_id']}.mp3"
        if out_path.exists():
            continue
        try:
            tts = gTTS(sent["text_dev"], lang="ne")
            tts.save(str(out_path))
        except Exception as e:
            print(f"  ERROR {sent['sent_id']}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  gtts: {i+1}/{len(sentences)}")
        time.sleep(0.3)  # Rate limiting
    count = len(list(output_dir.glob("*.mp3")))
    print(f"  gtts: {count} files")


def generate_mms_tts(sentences, output_dir):
    """Generate with Meta MMS-TTS."""
    import torch
    import scipy.io.wavfile

    output_dir.mkdir(parents=True, exist_ok=True)

    print("  Loading MMS-TTS model...")
    from transformers import VitsModel, AutoTokenizer
    model = VitsModel.from_pretrained("facebook/mms-tts-npi")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-npi")

    for i, sent in enumerate(sentences):
        out_path = output_dir / f"{sent['sent_id']}.wav"
        if out_path.exists():
            continue
        try:
            inputs = tokenizer(sent["text_dev"], return_tensors="pt")
            with torch.no_grad():
                output = model(**inputs).waveform
            waveform = output.squeeze().numpy()
            scipy.io.wavfile.write(str(out_path), rate=model.config.sampling_rate, data=waveform)
        except Exception as e:
            print(f"  ERROR {sent['sent_id']}: {e}")
        if (i + 1) % 20 == 0:
            print(f"  mms_tts: {i+1}/{len(sentences)}")
    count = len(list(output_dir.glob("*.wav")))
    print(f"  mms_tts: {count} files")


def generate_gemini(sentences, output_dir):
    """Generate with Gemini 2.5 Flash TTS."""
    from google import genai
    from google.genai import types

    # Load API key
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        fmw = Path.home() / ".fmw"
        if fmw.exists():
            for line in fmw.read_text().splitlines():
                if line.startswith("GEMINI_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        print("  ERROR: No GEMINI_API_KEY found")
        return

    client = genai.Client(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, sent in enumerate(sentences):
        out_path = output_dir / f"{sent['sent_id']}.wav"
        if out_path.exists():
            continue
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=sent["text_dev"],
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore",
                            )
                        )
                    ),
                ),
            )
            # Extract audio data
            audio_data = response.candidates[0].content.parts[0].inline_data.data
            # Write as WAV (PCM 16-bit 24kHz mono)
            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                wf.writeframes(audio_data)
        except Exception as e:
            print(f"  ERROR {sent['sent_id']}: {e}")
            time.sleep(1)
        if (i + 1) % 20 == 0:
            print(f"  gemini: {i+1}/{len(sentences)}")
        time.sleep(0.2)  # Rate limiting
    count = len(list(output_dir.glob("*.wav")))
    print(f"  gemini: {count} files")


SYSTEMS = {
    "edge_tts": generate_edge_tts,
    "gtts": generate_gtts,
    "mms_tts": generate_mms_tts,
    "gemini": generate_gemini,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default="all", choices=list(SYSTEMS.keys()) + ["all"])
    args = parser.parse_args()

    sentences = load_sentences()
    print(f"Loaded {len(sentences)} sentences\n")

    systems = list(SYSTEMS.keys()) if args.system == "all" else [args.system]

    for system in systems:
        print(f"=== Generating: {system} ===")
        output_dir = OUTPUT_BASE / system
        SYSTEMS[system](sentences, output_dir)
        print()


if __name__ == "__main__":
    main()
