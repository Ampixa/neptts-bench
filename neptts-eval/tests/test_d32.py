"""Private-judge identity and fail-closed coverage without model/audio access."""

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def d32():
    source = Path(__file__).resolve().parents[1] / "src/neptts_eval/d32.py"
    spec = importlib.util.spec_from_file_location("test_d32_module", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("field", ["weights_sha256", "bpe_sha256", "code_sha256", "device",
                                  "threads", "chunk_size", "normalization", "packages"])
def test_import_rejects_instrument_drift(d32, field):
    profile = copy.deepcopy(d32.profile("synthetic-benchmark"))
    profile[field] = "tampered"
    with pytest.raises(ValueError, match="identity/settings"):
        d32.validate_profile(profile, "synthetic-benchmark")


def test_profile_is_path_free_and_recomputable_without_private_runtime(d32):
    spec = d32.profile("synthetic-benchmark")
    d32.validate_profile(spec, "synthetic-benchmark")
    assert spec["parameters"] == 33447414
    assert spec["chunk_size"] == spec["left_context_frames"] == -1
    assert "private" in spec["availability"] and "call-trained" in spec["availability"]
    assert "/Users/" not in json.dumps(spec) and "/mnt/" not in json.dumps(spec)
    with pytest.raises(ValueError):
        d32.validate_profile(spec, "other-benchmark")


def test_snapshot_drift_rejected_before_any_code_import(d32, tmp_path, monkeypatch):
    (tmp_path / "normalization_manifest.json").write_text("tampered")
    monkeypatch.setattr(d32, "_load_module", lambda *a, **k: pytest.fail("Imported unverified code"))
    with pytest.raises(ValueError, match="hash mismatch"):
        d32.load_normalizer(tmp_path)


def test_legacy_latin_condition_preserves_latin_references(d32, monkeypatch):
    calls = []
    def normalize(text, **kwargs):
        calls.append((text, kwargs))
        return "हेलो"
    def suite_counts(ref, hyp):
        return {"cer": 0.0, "wer": 0.0, "char_errors": 0, "chars": 1,
                "word_errors": 0, "words": 1, "ref_norm": ref, "hyp_norm": hyp}
    def load(name, *args):
        if name == "_d32_frozen_suite":
            return SimpleNamespace(cer_wer=suite_counts)
        return SimpleNamespace(normalize=normalize)
    monkeypatch.setattr(d32, "_verified", lambda *a: None)
    monkeypatch.setattr(d32, "_load_module", load)
    counts = d32.load_normalizer("synthetic-unused-root")
    counts("हेलो", "hello")
    counts("hello नेपाल", "hello")
    counts("हेलो", "हेलो")
    assert calls == [("hello", {"english": True, "nepal": False})]


def test_cpu_thread_deviation_fails_before_private_model_loading(d32):
    with pytest.raises(ValueError, match="one CPU thread"):
        d32.build_decoder({"threads": 2})


@pytest.mark.parametrize("revision,status", [("wrong-revision", ""),
                                            ("pinned", " M icefall/utils.py")])
def test_icefall_deviation_fails_before_private_model_loading(d32, monkeypatch, revision, status):
    monkeypatch.setattr(d32.importlib.metadata, "version", lambda name: d32.PINS["packages"][name])
    monkeypatch.setattr(d32, "_verified", lambda *a: None)
    def run(command, **kwargs):
        value = (d32.PINS["icefall_commit"] if revision == "pinned" else revision)
        return SimpleNamespace(stdout=status if "status" in command else value)
    monkeypatch.setattr(d32.subprocess, "run", run)
    with pytest.raises(ValueError, match="icefall revision/worktree"):
        d32.build_decoder({key: "synthetic-unused" for key in ("checkpoint", "bpe", "recipe", "icefall")})


@pytest.mark.parametrize("component", ["eval_private", "khalti", "calls", "orbit"])
def test_call_paths_never_reach_audio_reader(d32, component):
    with pytest.raises(ValueError, match="Forbidden"):
        d32.read_audio(Path("/") / component / "synthetic.wav")


def test_every_prompt_scored_and_public_report_omits_hypotheses(d32, monkeypatch):
    def counts(ref, hyp):
        assert hyp == "synthetic private hypothesis"
        return {"cer": 0.5, "wer": 1.0, "char_errors": 1, "chars": 2,
                "word_errors": 1, "words": 1}
    monkeypatch.setattr(d32, "load_normalizer", lambda config: counts)
    monkeypatch.setattr(d32, "build_decoder", lambda config: lambda _: "synthetic private hypothesis")
    read_ids = []
    monkeypatch.setattr(d32, "read_audio", lambda path: read_ids.append(path.stem))
    progress = []
    sentences = {sid: {"text_devanagari": "synthetic reference"} for sid in ("sent_002", "sent_001")}
    report = d32.evaluate("synthetic-private-root", sentences, "synthetic-benchmark",
                          {"normalizer_root": "unused"}, progress.append)
    assert read_ids == ["sent_001", "sent_002"] and progress == [1, 2]
    assert set(report["per_sentence"]) == set(sentences)
    assert report["cer"] == report["corpus_cer"] == 0.5
    encoded = json.dumps(report)
    for private in ("private hypothesis", "synthetic reference", "synthetic-private-root"):
        assert private not in encoded


def test_decode_failure_aborts_complete_report(d32, monkeypatch):
    def decode(_):
        raise RuntimeError("Synthetic decode failure")
    monkeypatch.setattr(d32, "load_normalizer", lambda config: None)
    monkeypatch.setattr(d32, "build_decoder", lambda config: decode)
    monkeypatch.setattr(d32, "read_audio", lambda path: None)
    with pytest.raises(RuntimeError, match="Synthetic decode failure"):
        d32.evaluate("unused", {"sent_001": {"text_devanagari": "synthetic"}},
                     "synthetic", {"normalizer_root": "unused"})
