"""Instrument gates exercise fixed decoder identity without models or real audio."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def analysis(monkeypatch):
    source = ROOT / "benchmark/judge_calibration_v1/calibration/analyze.py"
    monkeypatch.syspath_prepend(str(source.parent))
    spec = importlib.util.spec_from_file_location("calibration_analysis", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arena_counts_match_immutable_scorer_spaces_digits_and_script(analysis):
    from submission_app.evaluation import (
        char_error_rate,
        normalize_nepali_text,
        word_error_rate,
    )

    # Deliberately artificial inputs: no recognition or calibration claim.
    ref, hyp = "क ख १२ hello।", "कख 12 HELLO!"
    got = analysis.arena_counts(ref, hyp)
    r, h = normalize_nepali_text(ref), normalize_nepali_text(hyp)
    assert got["cer"] == char_error_rate(r, h) == 0
    assert got["wer"] == word_error_rate(r, h)
    assert got["chars"] == len(r.replace(" ", ""))
    assert analysis.arena_counts("हेलो", "hello")["cer"] > 0


def streaming_receipt(analysis):
    # Identity-only fixture uses actual verified pins, never invented CER values.
    return {
        "mode": "online",
        "threads": 1,
        "provider": "cpu",
        "worker_sha256": hashlib.sha256(
            (
                ROOT / "benchmark/judge_calibration_v1/calibration/run_zipformer.py"
            ).read_bytes()
        ).hexdigest(),
        "sherpa_onnx": "identity-fixture",
        "padding_samples": 12800,
        "hashes": {**analysis.D32_STREAMING_PINS, "tokens.txt": analysis.D32_TOKEN_SHA},
        "export_metadata": {
            **analysis.D32_STREAMING_SETTINGS,
            "checkpoint": "/restricted/model.pt",
            "files": {
                name: {"sha256": value}
                for name, value in analysis.D32_STREAMING_PINS.items()
            },
        },
    }


def test_streaming_identity_retains_settings_without_paths(analysis, tmp_path):
    receipt = streaming_receipt(analysis)
    path = tmp_path / "observation.meta.json"
    path.write_text(json.dumps(receipt))
    result, _ = analysis.instrument_metadata(
        "d32_streaming", [path.with_name("observation.jsonl")]
    )
    assert result["export_settings"] == analysis.D32_STREAMING_SETTINGS
    assert "/restricted" not in json.dumps(result)
    assert "checkpoint" not in result


@pytest.mark.parametrize(
    "field",
    [
        "chunk_size",
        "left_context_frames",
        "checkpoint_sha256",
        "bpe_sha256",
        "n_params",
    ],
)
def test_streaming_wrong_export_settings_cannot_claim_320ms(analysis, tmp_path, field):
    receipt = streaming_receipt(analysis)
    receipt["export_metadata"][field] = "wrong-identity"
    path = tmp_path / "observation.meta.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="D32 streaming"):
        analysis.instrument_metadata(
            "d32_streaming", [path.with_name("observation.jsonl")]
        )


def test_streaming_actual_model_hash_must_match_declared_export(analysis, tmp_path):
    receipt = streaming_receipt(analysis)
    receipt["hashes"]["encoder.int8.onnx"] = "0" * 64
    path = tmp_path / "observation.meta.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="actual model hash mismatch"):
        analysis.instrument_metadata(
            "d32_streaming", [path.with_name("observation.jsonl")]
        )


def test_streaming_token_file_must_match_verified_bpe_mapping(analysis, tmp_path):
    receipt = streaming_receipt(analysis)
    receipt["hashes"]["tokens.txt"] = "0" * 64
    path = tmp_path / "observation.meta.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="tokens.txt identity mismatch"):
        analysis.instrument_metadata(
            "d32_streaming", [path.with_name("observation.jsonl")]
        )
