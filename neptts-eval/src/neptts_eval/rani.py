"""Rani ASR (Ampixa Labs) intelligibility judge for NepTTS-Bench.

Rani is a NeMo hybrid TDT/CTC Conformer trained mostly on conversational and
telephone Nepali. Weights are public but gated (huggingface.co/ampixa/rani-asr).
Licence: evaluating other models with Rani outputs requires the attribution
"Rani (Ampixa Labs)" with a link, which this profile carries.

Importing this module loads no model and touches no network. The evaluator
identity comes from `rani_contract.json`; the checkpoint hash and runtime
versions are recorded there once with `python -m neptts_eval.rani pin`. Until
pinned, every scoring and import path fails closed.

    python -m neptts_eval.rani pin --checkpoint ~/models/rani-asr/rani/rani.nemo
    python -m neptts_eval.rani score --wav-dir DIR --sentences benchmark/sentences.json \
        --benchmark ID --checkpoint ... --output report.json
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import tempfile
from pathlib import Path

from .metrics import _edit_distance
from .rani_text import normalize

PROFILE_NAME = "rani"
HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / "rani_contract.json"
PINS = json.loads(CONTRACT_PATH.read_text())
RUNTIME_PACKAGES = ("nemo_toolkit", "torch", "numpy", "scipy", "soundfile")
THREADS = 2
SAMPLE_RATE = 16000
DROP_TOKENS = {"<breath>"}
IDENTITY_SOURCES = ("rani.py", "rani_text.py", "rani_numbers.py",
                    "rani_latin_map.json", "metrics.py")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def code_sha256():
    """Scorer, normalizer and resources; the contract's pins are in the spec itself."""
    return hashlib.sha256(b"".join((HERE / n).read_bytes() for n in IDENTITY_SOURCES)).hexdigest()


def is_pinned(pins=None):
    pins = PINS if pins is None else pins
    sha, packages = pins.get("checkpoint_sha256"), pins.get("packages")
    return (isinstance(sha, str) and len(sha) == 64
            and isinstance(packages, dict) and set(packages) == set(RUNTIME_PACKAGES))


def profile(benchmark):
    """Public, path-free evaluator identity. Refuses until the contract is pinned."""
    if not is_pinned():
        raise ValueError("Rani contract is not pinned; run `python -m neptts_eval.rani pin`.")
    spec = {
        "benchmark": benchmark,
        "model": PROFILE_NAME,
        "weights_sha256": PINS["checkpoint_sha256"],
        "weights_source": PINS["source"]["weights"] + " " + PINS["source"]["checkpoint_file"],
        "upstream_commit": PINS["source"]["commit"],
        "code_sha256": code_sha256(),
        "packages": PINS["packages"],
        "language": "ne",
        "device": "cpu",
        "threads": THREADS,
        "batch_size": 1,
        "decoding": "rnnt (TDT) head, greedy_batch, restore_from strict=False; no language model",
        "audio": "soundfile float32; mono channel mean; scipy resample_poly to 16000 Hz; float WAV",
        "postprocess": "drop <breath> tokens",
        "normalization": "NepTel 0.2 official scorer (score_v02) on reference and hypothesis",
        "cer": "character edits excluding spaces",
        "aggregation": "macro-average sentence CER/WER; corpus CER/WER from edit counts",
        "development_only": False,
        "attribution": PINS["attribution"],
        "availability": "public gated weights; access approval at huggingface.co/ampixa/rani-asr",
    }
    spec["id"] = PROFILE_NAME + "-" + hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()[:12]
    return spec


def validate_profile(spec, benchmark):
    """Reject any drift in settings, code, weights or runtime without loading models."""
    if spec != profile(benchmark):
        raise ValueError("Rani evaluator identity/settings do not match the pinned profile.")


def counts(reference, hypothesis):
    ref, hyp = normalize(reference), normalize(hypothesis)
    ref_chars, hyp_chars = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    ref_words, hyp_words = ref.split(), hyp.split()
    if not ref_chars or not ref_words:
        raise ValueError("Rani reference normalizes to empty text.")
    char_errors = _edit_distance(ref_chars, hyp_chars)
    word_errors = _edit_distance(ref_words, hyp_words)
    return {
        "cer": char_errors / len(ref_chars), "wer": word_errors / len(ref_words),
        "char_errors": char_errors, "chars": len(ref_chars),
        "word_errors": word_errors, "words": len(ref_words),
    }


def clean_transcript(text):
    return " ".join(t for t in str(text).split() if t not in DROP_TOKENS)


def read_audio(path):
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    try:
        samples, rate = sf.read(path, dtype="float32", always_2d=True)
    except (RuntimeError, OSError, ValueError) as exc:
        raise RuntimeError("Rani audio read failed (" + type(exc).__name__ + ").") from None
    samples = samples.mean(axis=1)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError("Rani audio is empty or nonfinite.")
    if rate != SAMPLE_RATE:
        g = math.gcd(rate, SAMPLE_RATE)
        samples = resample_poly(samples, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return samples


def _check_runtime(checkpoint):
    if not is_pinned():
        raise ValueError("Rani contract is not pinned; run `python -m neptts_eval.rani pin`.")
    for name, expected in PINS["packages"].items():
        if importlib.metadata.version(name) != expected:
            raise ValueError("Rani runtime version mismatch: " + name)
    if sha256(checkpoint) != PINS["checkpoint_sha256"]:
        raise ValueError("Rani checkpoint hash mismatch.")


def build_decoder(config):
    """Verify weights and runtime, then load Rani exactly as its README documents."""
    checkpoint = Path(config["checkpoint"])
    _check_runtime(checkpoint)
    import torch
    from nemo.collections.asr.models import EncDecHybridRNNTCTCBPEModel
    from omegaconf import OmegaConf, open_dict

    torch.set_num_threads(THREADS)
    model = EncDecHybridRNNTCTCBPEModel.restore_from(
        str(checkpoint), map_location="cpu", strict=False).eval()
    with open_dict(model.cfg):
        model.cfg.validation_ds = OmegaConf.create({"use_start_end_token": False})
    decoding = model.cfg.decoding
    with open_dict(decoding):
        decoding.strategy = "greedy_batch"
    model.change_decoding_strategy(decoding, decoder_type="rnnt")

    @torch.no_grad()
    def decode(wav_path):
        out = model.transcribe([str(wav_path)], batch_size=1, verbose=False)[0]
        return getattr(out, "text", out)
    return decode


def score_file(decode, path, text):
    """Counts for one recording (e.g. a donated take) against its prompt text."""
    import soundfile as sf

    with tempfile.TemporaryDirectory() as temp:
        wav = Path(temp) / "clip.wav"
        sf.write(wav, read_audio(path), SAMPLE_RATE, subtype="FLOAT")
        return counts(text, clean_transcript(decode(wav)))


def evaluate(audio_dir, sentences, benchmark, config, progress=lambda n: None, decoder=None):
    """Score every requested ID exactly once; return numbers only, never transcripts."""
    import soundfile as sf

    spec = profile(benchmark)
    decode = decoder or build_decoder(config)
    results = {}
    with tempfile.TemporaryDirectory() as temp:
        for i, (sid, sentence) in enumerate(sorted(sentences.items()), 1):
            wav = Path(temp) / (sid + ".wav")
            sf.write(wav, read_audio(Path(audio_dir) / (sid + ".wav")), SAMPLE_RATE, subtype="FLOAT")
            results[sid] = counts(sentence["text_devanagari"], clean_transcript(decode(wav)))
            wav.unlink()
            progress(i)
    if not results:
        raise ValueError("Rani evaluation needs a nonempty prompt set.")
    total = {k: sum(r[k] for r in results.values())
             for k in ("char_errors", "chars", "word_errors", "words")}
    return {
        "profile": spec, "per_sentence": results,
        "cer": sum(r["cer"] for r in results.values()) / len(results),
        "wer": sum(r["wer"] for r in results.values()) / len(results),
        "corpus_cer": total["char_errors"] / total["chars"],
        "corpus_wer": total["word_errors"] / total["words"],
        "human_mos": None, "predicted_mos": None,
    }


def pin(checkpoint, force=False):
    """Record the checkpoint hash and installed runtime versions in the contract."""
    if is_pinned() and not force:
        raise SystemExit("Contract already pinned; pass --force to create a new evaluator identity.")
    pins = dict(PINS)
    pins["checkpoint_sha256"] = sha256(checkpoint)
    pins["packages"] = {n: importlib.metadata.version(n) for n in RUNTIME_PACKAGES}
    CONTRACT_PATH.write_text(json.dumps(pins, indent=2, ensure_ascii=False) + "\n")
    return pins


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pin", help="record checkpoint SHA256 and runtime versions")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--force", action="store_true")
    s = sub.add_parser("score", help="score a directory of <sent_id>.wav files")
    s.add_argument("--wav-dir", type=Path, required=True)
    s.add_argument("--sentences", type=Path, required=True)
    s.add_argument("--benchmark", required=True)
    s.add_argument("--checkpoint", type=Path, required=True)
    s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "pin":
        pins = pin(args.checkpoint, args.force)
        print("Pinned", pins["checkpoint_sha256"], pins["packages"])
        return
    if args.output.exists():
        parser.error("Output already exists.")
    rows = json.loads(args.sentences.read_text(encoding="utf-8"))
    sentences = {r["sent_id"]: r for r in rows if r["sent_id"].startswith("sent_")}
    report = evaluate(args.wav_dir, sentences, args.benchmark, {"checkpoint": args.checkpoint},
                      progress=lambda n: print(f"Evaluated {n}/{len(sentences)}", flush=True))
    with args.output.open("x") as out:
        json.dump(report, out, allow_nan=False, indent=2)
        out.write("\n")


if __name__ == "__main__":
    main()
