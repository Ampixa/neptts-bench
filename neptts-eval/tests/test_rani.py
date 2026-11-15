"""Rani judge identity, normalization and scoring without NeMo, weights or network."""

import copy
import json

import numpy as np
import pytest
import soundfile as sf

from neptts_eval import rani, rani_text

FAKE_PINS = {
    **rani.PINS,
    "checkpoint_sha256": "a" * 64,
    "packages": {n: "0.0.0" for n in rani.RUNTIME_PACKAGES},
}


@pytest.fixture
def pinned(monkeypatch):
    monkeypatch.setattr(rani, "PINS", copy.deepcopy(FAKE_PINS))
    return rani


def test_unpinned_contract_fails_closed(monkeypatch):
    monkeypatch.setattr(rani, "PINS", {**FAKE_PINS, "checkpoint_sha256": None})
    assert not rani.is_pinned()
    with pytest.raises(ValueError, match="not pinned"):
        rani.profile("synthetic-benchmark")
    with pytest.raises(ValueError, match="not pinned"):
        rani._check_runtime("missing.nemo")
    assert not rani.is_pinned({**FAKE_PINS, "packages": {"torch": "1"}})


def test_shipped_contract_is_well_formed():
    pins = json.loads(rani.CONTRACT_PATH.read_text())
    assert pins["model"] == "rani"
    assert "Ampixa Labs" in pins["attribution"]
    assert set(pins["source"]["upstream_sha256"]) == {
        "eval/score_v02.py", "eval/nepali_numbers.py", "benchmark/latin_map.json"}


@pytest.mark.parametrize("field", ["weights_sha256", "code_sha256", "packages", "threads",
                                   "decoding", "normalization", "attribution", "benchmark"])
def test_profile_rejects_drift(pinned, field):
    spec = copy.deepcopy(pinned.profile("synthetic-benchmark"))
    spec[field] = "tampered"
    with pytest.raises(ValueError, match="identity/settings"):
        pinned.validate_profile(spec, "synthetic-benchmark")


def test_profile_is_path_free_and_attributed(pinned):
    spec = pinned.profile("synthetic-benchmark")
    pinned.validate_profile(spec, "synthetic-benchmark")
    assert spec["id"].startswith("rani-")
    assert spec["development_only"] is False
    assert "Rani (Ampixa Labs)" in spec["attribution"]
    text = json.dumps(spec)
    assert "/home/" not in text and "/mnt/" not in text and "/Users/" not in text


def test_code_identity_covers_normalizer_resources(pinned, monkeypatch, tmp_path):
    before = pinned.code_sha256()
    for name in pinned.IDENTITY_SOURCES:
        (tmp_path / name).write_bytes((pinned.HERE / name).read_bytes())
    (tmp_path / "rani_latin_map.json").write_text("{}")
    monkeypatch.setattr(pinned, "HERE", tmp_path)
    assert pinned.code_sha256() != before


@pytest.mark.parametrize("ref,hyp", [
    ("२० वटा", "बीस वटा"),                      # digits vs spoken words
    ("Facebook मा post गर्नुहोस्", "फेसबुक मा पोस्ट गर्नुहोस्"),  # Latin vs Devanagari
    ("नमस्ते, कस्तो छ?", "नमस्ते कस्तो छ"),      # punctuation
])
def test_equivalent_spellings_cost_nothing(ref, hyp):
    result = rani.counts(ref, hyp)
    assert result["char_errors"] == result["word_errors"] == 0


def test_counts_are_consistent_and_exclude_spaces():
    result = rani.counts("कलम किन", "कमल किन")
    assert result["words"] == 2 and result["word_errors"] == 1
    assert result["chars"] == len("कलमकिन")
    assert result["cer"] == result["char_errors"] / result["chars"]
    with pytest.raises(ValueError, match="empty"):
        rani.counts("।", "")


def test_breath_tokens_dropped():
    assert rani.clean_transcript("<breath> नमस्ते <breath> हजुर") == "नमस्ते हजुर"


def test_known_upstream_number_spelling_quirk_is_preserved():
    # Kept so scores match Rani's published NepTel numbers; see rani_text docstring.
    assert rani_text.normalize("२९") == "उनन्तिस"


def test_evaluate_resamples_and_scores_every_prompt(pinned, tmp_path):
    sentences = {"sent_001": {"text_devanagari": "नमस्ते हजुर"},
                 "sent_002": {"text_devanagari": "२० वटा"}}
    for sid in sentences:
        stereo = np.column_stack([np.zeros(48000), np.zeros(48000)]) + 0.1
        sf.write(tmp_path / f"{sid}.wav", stereo, 48000, subtype="PCM_16")
    seen, done = [], []

    def decoder(path):
        info = sf.info(path)
        seen.append((info.samplerate, info.channels))
        return {"sent_001": "<breath> नमस्ते हजुर", "sent_002": "बीस वटा"}[path.stem]

    report = pinned.evaluate(tmp_path, sentences, "synthetic-benchmark", {}, done.append, decoder)
    assert seen == [(16000, 1), (16000, 1)]
    assert done == [1, 2]
    assert report["cer"] == report["wer"] == report["corpus_cer"] == 0
    assert report["profile"] == pinned.profile("synthetic-benchmark")
    assert report["human_mos"] is None and report["predicted_mos"] is None
    assert "नमस्ते" not in json.dumps(report["per_sentence"])  # numbers only, no transcripts


def test_evaluate_fails_on_missing_audio(pinned, tmp_path):
    with pytest.raises(RuntimeError, match="audio read failed"):
        pinned.evaluate(tmp_path, {"sent_001": {"text_devanagari": "क"}}, "b", {}, decoder=str)


def test_checkpoint_hash_checked_before_model_import(pinned, tmp_path, monkeypatch):
    monkeypatch.setattr(rani.importlib.metadata, "version", lambda name: "0.0.0")
    checkpoint = tmp_path / "rani.nemo"
    checkpoint.write_bytes(b"not the pinned weights")
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        pinned.build_decoder({"checkpoint": checkpoint})


def test_pin_records_hash_and_refuses_silent_repin(monkeypatch, tmp_path):
    contract = tmp_path / "rani_contract.json"
    monkeypatch.setattr(rani, "CONTRACT_PATH", contract)
    monkeypatch.setattr(rani, "PINS", {**FAKE_PINS, "checkpoint_sha256": None, "packages": None})
    monkeypatch.setattr(rani.importlib.metadata, "version", lambda name: "9.9")
    checkpoint = tmp_path / "rani.nemo"
    checkpoint.write_bytes(b"weights")
    pins = rani.pin(checkpoint)
    assert pins["checkpoint_sha256"] == rani.sha256(checkpoint)
    assert json.loads(contract.read_text())["packages"]["nemo_toolkit"] == "9.9"
    monkeypatch.setattr(rani, "PINS", pins)
    with pytest.raises(SystemExit, match="already pinned"):
        rani.pin(checkpoint)
