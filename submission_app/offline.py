"""Evaluate a reviewer-downloaded package locally; output a bound JSON report."""

import argparse
import hashlib
import json
import math
import tempfile
import zipfile
from pathlib import Path

from .contract import BENCHMARK, MAX_ARCHIVE, MAX_EXPANDED, SENTENCES, validate_archive
from .evaluation import evaluate, validate_report


def validate_d32_report(report):
    """Accept numeric evidence only; reject transcripts, paths and unbound counts."""
    if set(report) != {
        "profile",
        "per_sentence",
        "cer",
        "wer",
        "corpus_cer",
        "corpus_wer",
        "human_mos",
        "predicted_mos",
    }:
        raise ValueError("Unsupported D32 report fields.")
    if report["human_mos"] is not None or report["predicted_mos"] is not None:
        raise ValueError("D32 reports accept CER/WER only.")
    totals = dict(char_errors=0, chars=0, word_errors=0, words=0)
    for row in report["per_sentence"].values():
        if set(row) != {"cer", "wer", *totals}:
            raise ValueError("Unsupported D32 sentence fields.")
        for field in totals:
            if type(row[field]) is not int or row[field] < (
                1 if field in ("chars", "words") else 0
            ):
                raise ValueError("Invalid D32 edit counts.")
            totals[field] += row[field]
        for metric, errors, units in (
            ("cer", "char_errors", "chars"),
            ("wer", "word_errors", "words"),
        ):
            if abs(row[metric] - row[errors] / row[units]) > 1e-8:
                raise ValueError("D32 sentence metric does not match edit counts.")
    for metric, errors, units in (
        ("corpus_cer", "char_errors", "chars"),
        ("corpus_wer", "word_errors", "words"),
    ):
        value = report[metric]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or abs(value - totals[errors] / totals[units]) > 1e-8
        ):
            raise ValueError("D32 corpus metric does not match edit counts.")


def validate_import(payload, sid, audio_sha):
    if (
        not isinstance(payload, dict)
        or payload.get("submission_id") != sid
        or payload.get("benchmark") != BENCHMARK
        or payload.get("audio_sha256") != audio_sha
    ):
        raise ValueError("Results do not match this submission, benchmark and audio.")
    report = payload.get("report")
    if not isinstance(report, dict) or not isinstance(report.get("profile"), dict):
        raise TypeError("Missing evaluator report/profile.")
    spec = report["profile"]
    if spec.get("model") == "d32v2-s60000-full-translit":
        from neptts_eval import d32

        d32.validate_profile(spec, BENCHMARK)
        validate_report(report, spec["id"])
        validate_d32_report(report)
        return report
    if (
        spec.get("benchmark") != BENCHMARK
        or spec.get("model") != "small"
        or spec.get("development_only") is not False
    ):
        raise ValueError("Use the publishable Whisper small evaluator profile.")
    identity = dict(spec)
    profile_id = identity.pop("id", None)
    expected = (
        "whisper-small-ne-"
        + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    )
    if profile_id != expected:
        raise ValueError("Evaluator profile identity does not match its settings.")
    for field in ("weights_sha256", "code_sha256"):
        value = spec.get(field, "")
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("Missing evaluator provenance.")
    if not isinstance(spec.get("packages"), dict) or not spec["packages"]:
        raise ValueError("Missing evaluator package versions.")
    if report.get("human_mos") is not None or report.get("predicted_mos") is not None:
        raise ValueError("This import accepts CER/WER only, not MOS claims.")
    validate_report(report, profile_id)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["whisper-small", "d32v2-s60000-full-translit"],
        default="whisper-small",
    )
    for name in ("checkpoint", "bpe", "recipe", "icefall", "normalizer-root"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    if args.profile == "d32v2-s60000-full-translit" and not all(
        getattr(args, key)
        for key in ("checkpoint", "bpe", "recipe", "icefall", "normalizer_root")
    ):
        parser.error(
            "D32 requires checkpoint, bpe, recipe, icefall and normalizer-root."
        )
    if args.output.exists():
        parser.error("Output already exists; choose a new path.")
    if args.package.stat().st_size > MAX_ARCHIVE + 16384:
        raise ValueError("Package exceeds size limit.")
    temporary_root = (
        args.package.resolve().parent
        if args.profile == "d32v2-s60000-full-translit"
        else None
    )
    with tempfile.TemporaryDirectory(dir=temporary_root) as temp:
        temp = Path(temp)
        with zipfile.ZipFile(args.package) as source:
            # Never extract arbitrary paths or execute files supplied in a package.
            if (
                len(source.infolist()) > len(SENTENCES) + 2
                or sum(m.file_size for m in source.infolist()) > MAX_EXPANDED + 16384
            ):
                raise ValueError("Package exceeds expanded size/member limit.")
            if source.getinfo("evaluation-job.json").file_size > 16384:
                raise ValueError("Oversized job metadata.")
            job = json.loads(source.read("evaluation-job.json"))
            bundle = temp / "submission.zip"
            with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as target:
                for member in source.infolist():
                    if member.filename == "evaluation-job.json":
                        continue
                    if member.file_size > 8 * 1024**2:
                        raise ValueError("Oversized package member.")
                    target.writestr(member, source.read(member))
        _, validation = validate_archive(bundle, temp / "audio", reviewer_package=True)
        if (
            job.get("audio_sha256") != validation["audio_sha256"]
            or job.get("benchmark") != BENCHMARK
        ):
            raise ValueError("Package integrity/benchmark mismatch.")

        def progress(n):
            print(f"Evaluated {n}/{len(SENTENCES)}", flush=True)

        if args.profile == "whisper-small":
            report = evaluate(temp / "audio", "small", progress)
        else:
            from neptts_eval import d32

            config = {
                key: getattr(args, key)
                for key in ("checkpoint", "bpe", "recipe", "icefall", "normalizer_root")
            }
            config["threads"] = 1
            report = d32.evaluate(
                temp / "audio", SENTENCES, BENCHMARK, config, progress
            )
        payload = {**job, "report": report}
        validate_import(payload, job["submission_id"], validation["audio_sha256"])
        args.output.write_text(json.dumps(payload, allow_nan=False, indent=2) + "\n")
        print(f"Results written to {args.output}. Upload in the reviewer screen.")


if __name__ == "__main__":
    main()
