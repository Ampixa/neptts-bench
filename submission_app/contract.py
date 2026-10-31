"""Versioned submission contract and bounded, audio-only ZIP validation."""

from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "benchmark/sentences.json"
SENTENCES = {
    s["sent_id"]: s
    for s in json.loads(SOURCE.read_text())
    if s.get("category") != "natural_speech"
    and re.fullmatch(r"sent_\d{3}", s["sent_id"])
}
CANONICAL = json.dumps(
    SENTENCES, ensure_ascii=False, sort_keys=True, separators=(",", ":")
).encode()
BENCHMARK = "neptts-synthesis-v1-" + hashlib.sha256(CANONICAL).hexdigest()[:12]
MAX_ARCHIVE = 150 * 1024 * 1024
MAX_EXPANDED = 300 * 1024 * 1024
MAX_AUDIO = 8 * 1024 * 1024


class Manifest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    benchmark: str
    system_name: str = Field(min_length=2, max_length=80)
    model_version: str = Field(min_length=1, max_length=100)
    organization: str = Field(min_length=1, max_length=100)
    model_url: str = Field(default="", max_length=300)
    voice: str = Field(min_length=1, max_length=120)
    generation_settings: str = Field(min_length=1, max_length=1500)
    post_processing: str = Field(min_length=1, max_length=500)
    selection_policy: str = Field(min_length=1, max_length=500)
    training_overlap: str = Field(min_length=1, max_length=500)
    permission_to_evaluate_and_publish: bool

    @field_validator("benchmark")
    @classmethod
    def current_benchmark(cls, v):
        if v != BENCHMARK:
            raise ValueError(
                f"Use benchmark {BENCHMARK}; download the current sentence pack."
            )
        return v

    @field_validator("model_url")
    @classmethod
    def safe_url(cls, v):
        if v and not v.startswith("https://"):
            raise ValueError("Model URL must start with https://")
        return v

    @field_validator("permission_to_evaluate_and_publish")
    @classmethod
    def consent(cls, v):
        if not v:
            raise ValueError("Permission to evaluate and publish scores is required.")
        return v


def template():
    return {
        "benchmark": BENCHMARK,
        "system_name": "Your TTS",
        "model_version": "checkpoint-or-api-version",
        "organization": "Your team",
        "model_url": "",
        "voice": "voice-name",
        "generation_settings": "seed, temperature and all inference settings",
        "post_processing": "none",
        "selection_policy": "One generation per sentence; no cherry-picking",
        "training_overlap": "Declare any training on these benchmark sentences, or unknown",
        "permission_to_evaluate_and_publish": False,
    }


def sentence_pack():
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(
            "sentences.json",
            json.dumps(list(SENTENCES.values()), ensure_ascii=False, indent=2),
        )
        z.writestr("submission.json", json.dumps(template(), indent=2))
        z.writestr(
            "INSTRUCTIONS.txt",
            f"Benchmark: {BENCHMARK}\nGenerate all {len(SENTENCES)} exact prompts.\n"
            "Submit only submission.json and audio/sent_NNN.wav in one ZIP.\n"
            "WAV: mono PCM16, 16–48 kHz, 0.2–30 seconds. Keep the original generation rate.\n"
            "No missing or extra IDs, silence, symlinks, executable files or directories inside audio.\n"
            "Do not include reference recordings, personal data, secrets or model weights.\n"
            "Declare your voice, generation settings, post-processing, retries and training overlap.\n"
            "Use audio you may submit for evaluation. Uploaded audio stays private to reviewers;\n"
            "approved model metadata, scores and report are public. Human MOS is a separate study.\n",
        )
    return out.getvalue()


def validate_archive(path: Path, destination: Path):
    """Never extract paths supplied by the archive. Persist only fully validated WAVs."""
    expected = {f"audio/{sid}.wav" for sid in SENTENCES} | {"submission.json"}
    if path.stat().st_size > MAX_ARCHIVE:
        raise ValueError("Archive exceeds 150 MiB.")
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        if len(members) > len(expected) + 1:
            raise ValueError(
                "Too many archive entries. Include only the manifest and required audio."
            )
        names = [x.filename for x in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate ZIP paths are not allowed.")
        if sum(x.file_size for x in members) > MAX_EXPANDED:
            raise ValueError("Expanded archive exceeds 300 MiB.")
        for x in members:
            if x.filename == "audio/" and x.is_dir():
                continue
            if x.filename not in expected:
                raise ValueError(f"Unexpected archive entry: {x.filename[:120]}")
            mode = x.external_attr >> 16
            if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG)):
                raise ValueError("Only regular files are accepted.")
            if x.flag_bits & 1:
                raise ValueError("Encrypted ZIP files are not supported.")
            limit = 16384 if x.filename == "submission.json" else MAX_AUDIO
            if x.file_size > limit or x.compress_type not in (
                zipfile.ZIP_STORED,
                zipfile.ZIP_DEFLATED,
            ):
                raise ValueError("Unsupported compression or oversized member.")
        missing = sorted(expected - set(names))
        if missing:
            raise ValueError(
                f"Missing {len(missing)} required files: {', '.join(missing[:5])}"
            )
        manifest = Manifest.model_validate_json(z.read("submission.json"))
        digest = hashlib.sha256()
        total_seconds = 0.0
        details = {}
        destination.mkdir(parents=True, exist_ok=False)
        for sid in sorted(SENTENCES):
            raw = z.read(f"audio/{sid}.wav")
            with sf.SoundFile(io.BytesIO(raw)) as audio:
                duration = audio.frames / audio.samplerate
                if (
                    audio.format != "WAV"
                    or audio.subtype != "PCM_16"
                    or audio.channels != 1
                    or not 16000 <= audio.samplerate <= 48000
                    or not 0.2 <= duration <= 30
                ):
                    raise ValueError(
                        f"{sid}: use mono PCM16 WAV, 16–48 kHz and 0.2–30 seconds."
                    )
                samples = audio.read(dtype="float32")
                if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-4:
                    raise ValueError(f"{sid}: audio is silent or invalid.")
                sha = hashlib.sha256(raw).hexdigest()
                details[sid] = {
                    "sha256": sha,
                    "seconds": round(duration, 4),
                    "sample_rate": audio.samplerate,
                }
                total_seconds += duration
                digest.update(sid.encode() + bytes.fromhex(sha))
            (destination / f"{sid}.wav").write_bytes(raw)
    return manifest.model_dump(), {
        "files": len(details),
        "seconds": round(total_seconds, 3),
        "audio_sha256": digest.hexdigest(),
        "per_file": details,
    }
