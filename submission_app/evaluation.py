"""Server-owned ASR evaluation. Never trust scores or commands in an upload."""

import hashlib
import importlib.metadata
import json
import math
import os
import sys
from pathlib import Path

from .contract import BENCHMARK, ROOT, SENTENCES

sys.path.insert(0, str(ROOT / "neptts-eval/src"))
from neptts_eval.metrics import char_error_rate, word_error_rate
from neptts_eval.normalize import normalize_nepali_text


def profile(model="small"):
    if model not in ("tiny", "small"):
        raise ValueError("Supported evaluator models: tiny (development) or small.")
    sources = [
        Path(__file__),
        ROOT / "neptts-eval/src/neptts_eval/metrics.py",
        ROOT / "neptts-eval/src/neptts_eval/normalize.py",
    ]
    code_sha = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    import whisper

    weights_sha = whisper._MODELS[model].split("/")[-2]
    versions = {
        n: importlib.metadata.version(n) for n in ("openai-whisper", "torch", "numpy")
    }
    spec = {
        "benchmark": BENCHMARK,
        "model": model,
        "weights_sha256": weights_sha,
        "code_sha256": code_sha,
        "packages": versions,
        "language": "ne",
        "device": "cpu",
        "temperature": 0,
        "beam_size": 5,
        "aggregation": "macro-average sentence CER/WER",
        "development_only": model == "tiny",
    }
    spec["id"] = (
        "whisper-"
        + model
        + "-ne-"
        + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:12]
    )
    return spec


def validate_report(report, expected_profile):
    if report.get("profile", {}).get("id") != expected_profile:
        raise ValueError("Evaluator profile mismatch.")
    results = report.get("per_sentence", {})
    if set(results) != set(SENTENCES):
        raise ValueError("Evaluation must cover every required sentence exactly once.")
    for row in results.values():
        for metric in ("cer", "wer"):
            x = row.get(metric)
            if (
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                or x < 0
            ):
                raise ValueError("Invalid evaluator metric.")
    for metric in ("cer", "wer"):
        mean = sum(v[metric] for v in results.values()) / len(results)
        if abs(mean - report.get(metric, -1)) > 1e-8:
            raise ValueError("Report aggregate does not match per-sentence results.")


def evaluate(audio_dir, model_name, progress=lambda n: None):
    import torch
    import whisper

    torch.set_num_threads(2)
    spec = profile(model_name)
    model = whisper.load_model(
        model_name, device="cpu", download_root=os.environ.get("NEPTTS_MODEL_CACHE")
    )
    results = {}
    for i, (sid, sentence) in enumerate(sorted(SENTENCES.items()), 1):
        result = model.transcribe(
            str(audio_dir / f"{sid}.wav"),
            language="ne",
            task="transcribe",
            fp16=False,
            temperature=0,
            beam_size=5,
            condition_on_previous_text=False,
        )
        ref = normalize_nepali_text(sentence["text_devanagari"])
        hyp = normalize_nepali_text(result.get("text", ""))
        results[sid] = {
            "cer": char_error_rate(ref, hyp),
            "wer": word_error_rate(ref, hyp),
        }
        progress(i)
    return {
        "profile": spec,
        "per_sentence": results,
        "cer": sum(x["cer"] for x in results.values()) / len(results),
        "wer": sum(x["wer"] for x in results.values()) / len(results),
        "human_mos": None,
        "predicted_mos": None,
    }
