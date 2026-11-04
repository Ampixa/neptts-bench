"""Private, CPU-only DISTILL-32M v2 full-context calibrated judge.

Importing this module performs no model loading or network access. Reviewers
provide the existing owner-controlled model, recipe and normalizer snapshot.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import re
import subprocess
import sys

PROFILE_NAME = "d32v2-s60000-full-translit"
CONTRACT_PATH = Path(__file__).with_name("d32_contract.json")
PINS = json.loads(CONTRACT_PATH.read_text())
LATIN = re.compile(r"[A-Za-z]")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256():
    return hashlib.sha256(Path(__file__).read_bytes() + CONTRACT_PATH.read_bytes()).hexdigest()


def profile(benchmark):
    """Public, path-free instrument identity; no private inputs are read."""
    spec = {
        "benchmark": benchmark,
        "model": PROFILE_NAME,
        "weights_sha256": PINS["checkpoint_sha256"],
        "bpe_sha256": PINS["bpe_sha256"],
        "code_sha256": code_sha256(),
        "packages": PINS["packages"],
        "recipe_hashes": PINS["recipe_hashes"],
        "icefall_commit": PINS["icefall_commit"],
        "normalizer_hashes": PINS["normalizer_hashes"],
        "normalization_manifest_sha256": PINS["normalization_manifest_sha256"],
        "language": "ne",
        "device": "cpu",
        "threads": 1,
        "parameters": 33447414,
        "step": 60000,
        "chunk_size": -1,
        "left_context_frames": -1,
        "decoding": "greedy_search_batch; full-context FP32",
        "audio": "soundfile float32; mono channel mean; scipy resample_poly to 16000 Hz",
        "features": "lhotse Fbank(FbankConfig(num_filters=80)); 30 log-eps right-pad frames",
        "normalization": "legacy score2 conditional Latin-hypothesis transliteration; frozen suite CER includes spaces",
        "aggregation": "macro-average sentence CER/WER",
        "development_only": False,
        "availability": "private owner-operated reviewer profile; call-trained weights not released",
    }
    spec["id"] = PROFILE_NAME + "-" + hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:12]
    return spec


def validate_profile(spec, benchmark):
    """Reject every settings/source/runtime deviation without loading models."""
    if spec != profile(benchmark):
        raise ValueError("D32 evaluator identity/settings do not match the pinned profile.")


def _verified(path, expected, label):
    if sha256(path) != expected:
        raise ValueError("D32 source/model hash mismatch: " + label)


def _load_module(name, path, package=False):
    options = {"submodule_search_locations": [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load pinned D32 source module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_normalizer(normalizer_root):
    """Use only the calibrated snapshot, including its lexical resources."""
    root = Path(normalizer_root)
    _verified(root / "normalization_manifest.json", PINS["normalization_manifest_sha256"],
              "normalization_manifest.json")
    for name, expected in PINS["normalizer_hashes"].items():
        _verified(root / name, expected, name)
    snapshot = root / "normalization_snapshot"
    frontend = _load_module("_d32_frozen_frontend", snapshot / "frontend/__init__.py", True)
    nepali = _load_module("_d32_frozen_nepali_norm", snapshot / "nepali_norm.py")
    suite = _load_module("_d32_frozen_suite", snapshot / "suite_common.py")
    # The original suite discovers this exact function through an Ubuntu-only
    # path. Inject the verified snapshot function for portable offline use.
    suite._bench_normalize = nepali.normalize

    def counts(ref, hyp):
        if LATIN.search(hyp) and not LATIN.search(ref):
            hyp = frontend.normalize(hyp, english=True, nepal=False)
        result = suite.cer_wer(ref, hyp)
        if result["chars"] <= 0 or result["words"] <= 0:
            raise ValueError("D32 reference normalizes to empty text.")
        return {key: result[key] for key in
                ("cer", "wer", "char_errors", "chars", "word_errors", "words")}
    return counts


def build_decoder(config):
    """Hash-check all inputs before loading the calibrated CPU model."""
    threads = config.get("threads", 1)
    if type(threads) is not int or threads != 1:
        raise ValueError("The calibrated D32 profile requires one CPU thread.")
    for name, expected in PINS["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("D32 runtime version mismatch: " + name)
    checkpoint_path, bpe_path = Path(config["checkpoint"]), Path(config["bpe"])
    recipe, icefall = Path(config["recipe"]), Path(config["icefall"])
    _verified(checkpoint_path, PINS["checkpoint_sha256"], "checkpoint")
    _verified(bpe_path, PINS["bpe_sha256"], "tokenizer")
    for name, expected in PINS["recipe_hashes"].items():
        _verified(recipe / name, expected, name)
    revision = subprocess.run(["git", "-C", str(icefall), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(["git", "-C", str(icefall), "status", "--porcelain"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if revision != PINS["icefall_commit"] or status:
        raise ValueError("D32 icefall revision/worktree mismatch.")

    import sentencepiece as spm
    import torch
    from lhotse import Fbank, FbankConfig

    torch.set_num_threads(1)
    # These are verified owner-controlled sources, never submission contents.
    sys.path[:0] = [str(recipe), str(icefall)]
    from train import add_model_arguments, get_model, get_params
    from beam_search import greedy_search_batch

    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    params = get_params()
    parser = argparse.ArgumentParser()
    add_model_arguments(parser)
    parser.add_argument("--context-size", type=int, default=2)
    params.update(vars(parser.parse_args([])))
    for action in parser._actions:
        if action.dest in checkpoint["args"]:
            value = checkpoint["args"][action.dest]
            params[action.dest] = action.type(value) if action.type and isinstance(value, str) else value
    if "-1" not in str(params.chunk_size).split(",") or "-1" not in str(params.left_context_frames).split(","):
        raise ValueError("D32 checkpoint does not support calibrated full context.")
    params.chunk_size = params.left_context_frames = "-1"
    tokenizer = spm.SentencePieceProcessor(model_file=str(bpe_path))
    params.vocab_size = tokenizer.get_piece_size()
    params.blank_id = tokenizer.piece_to_id("<blk>")
    params.unk_id = tokenizer.piece_to_id("<unk>")
    params.sos_id = params.eos_id = tokenizer.piece_to_id("<sos/eos>")
    if params.blank_id != 0:
        raise ValueError("D32 blank token identity mismatch.")
    model = get_model(params).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if sum(p.numel() for p in model.parameters()) != 33447414:
        raise ValueError("D32 parameter count mismatch.")
    features = Fbank(FbankConfig(num_filters=80))

    @torch.no_grad()
    def decode(samples):
        f = torch.from_numpy(features.extract(samples.reshape(1, -1), 16000)).unsqueeze(0)
        lens = torch.tensor([f.shape[1] + 30], dtype=torch.int64)
        f = torch.nn.functional.pad(f, (0, 0, 0, 30), value=math.log(1e-10))
        out, out_lens = model.forward_encoder(f, lens)
        tokens = greedy_search_batch(model=model, encoder_out=out, encoder_out_lens=out_lens)
        return " ".join(tokenizer.decode(tokens)[0].split())
    return decode


def read_audio(path):
    # Call audio and transcripts are categorically outside this instrument.
    if {"eval_private", "khalti", "calls", "orbit"} & {p.lower() for p in Path(path).resolve().parts}:
        raise ValueError("Forbidden D32 input.")
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        # The caller can log the benchmark ID; audio filesystem paths stay private.
        raise RuntimeError("D32 audio read failed (" + type(exc).__name__ + ").") from None
    samples = samples.mean(axis=1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("D32 audio is empty or nonfinite.")
    if sample_rate != 16000:
        divisor = math.gcd(sample_rate, 16000)
        samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor).astype(np.float32)
    return samples


def evaluate(audio_dir, sentences, benchmark, config, progress=lambda n: None):
    """Score every requested ID exactly once; never return speech or paths."""
    counts = load_normalizer(config["normalizer_root"])
    decode = build_decoder(config)
    results = {}
    for i, (sid, sentence) in enumerate(sorted(sentences.items()), 1):
        samples = read_audio(Path(audio_dir) / (sid + ".wav"))
        results[sid] = counts(sentence["text_devanagari"], decode(samples))
        progress(i)
    if not results:
        raise ValueError("D32 evaluation needs a nonempty prompt set.")
    return {
        "profile": profile(benchmark), "per_sentence": results,
        "cer": sum(r["cer"] for r in results.values()) / len(results),
        "wer": sum(r["wer"] for r in results.values()) / len(results),
        "corpus_cer": sum(r["char_errors"] for r in results.values()) / sum(r["chars"] for r in results.values()),
        "corpus_wer": sum(r["word_errors"] for r in results.values()) / sum(r["words"] for r in results.values()),
        "human_mos": None, "predicted_mos": None,
    }


def main():
    """Standalone local-only scorer; arena package binding uses submission_app.offline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, required=True,
                        help="Local versioned sentences.json; no network fallback")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", type=Path, required=True)
    for name in ("checkpoint", "bpe", "recipe", "icefall", "normalizer-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output already exists.")
    rows = json.loads(args.sentences.read_text())
    if not isinstance(rows, list) or not rows:
        parser.error("Expected a nonempty sentence list.")
    print("SCHEMA sentences", sorted(rows[0]), flush=True)
    sentences = {row["sent_id"]: row for row in rows if row["sent_id"].startswith("sent_")}
    if len(sentences) != sum(row["sent_id"].startswith("sent_") for row in rows):
        parser.error("Duplicate synthesis sentence IDs.")
    config = {name: getattr(args, name) for name in
              ("checkpoint", "bpe", "recipe", "icefall", "normalizer_root")}
    report = evaluate(args.wav_dir, sentences, args.benchmark, config,
                      progress=lambda n: print(f"Evaluated {n}/{len(sentences)}", flush=True))
    with args.output.open("x") as output:
        json.dump(report, output, allow_nan=False, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
