"""Source-install prompt discovery without network access or user caches."""

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


@pytest.fixture
def data_module():
    source = Path(__file__).resolve().parents[1] / "src/neptts_eval/data.py"
    spec = importlib.util.spec_from_file_location("source_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_current_inventory(sentences):
    assert len(sentences) == 365
    assert sum(sid.startswith("sent_") for sid in sentences) == 205
    assert sum(sid.startswith("chirp_") for sid in sentences) == 160


def test_source_checkout_prompts_without_cache_or_network(data_module, tmp_path, monkeypatch):
    monkeypatch.setattr(data_module, "SENTENCES_CACHE", tmp_path / "missing.json")
    monkeypatch.chdir(tmp_path)
    assert_current_inventory(data_module.load_sentences())


def test_source_prompts_override_stale_cache_and_unrelated_cwd(data_module, tmp_path, monkeypatch):
    stale = [{"sent_id": f"sent_{i:03}", "text_devanagari": "stale"} for i in range(193)]
    cache = tmp_path / "sentences.json"
    cache.write_text(json.dumps(stale), encoding="utf-8")
    relative_prompts = tmp_path / "benchmark" / "sentences.json"
    relative_prompts.parent.mkdir()
    relative_prompts.write_text(json.dumps(stale[:1]), encoding="utf-8")
    monkeypatch.setattr(data_module, "SENTENCES_CACHE", cache)
    monkeypatch.chdir(tmp_path)
    assert_current_inventory(data_module.load_sentences())
    assert len(json.loads(cache.read_text())) == 193


def test_failed_download_retains_cause_and_recovery_path(data_module, tmp_path, monkeypatch):
    cause = ConnectionError("offline fixture")

    def unavailable(*args, **kwargs):
        raise cause

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=unavailable))
    monkeypatch.setattr(data_module, "SOURCE_SENTENCES", tmp_path / "absent_source.json")
    monkeypatch.setattr(data_module, "LEGACY_SOURCE_SENTENCES", tmp_path / "absent_legacy.json")
    monkeypatch.setattr(data_module, "SENTENCES_CACHE", tmp_path / "sentences.json")
    with pytest.raises(FileNotFoundError, match="verified sentences.json cache") as error:
        data_module.load_sentences()
    assert error.value.__cause__ is cause
    assert str(tmp_path / "sentences.json") in str(error.value)
