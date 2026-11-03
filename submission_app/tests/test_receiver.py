"""Exercise the receiver with real local symlinks and mocked installation/service I/O."""

import importlib.util
import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

RECEIVER = Path(__file__).resolve().parents[2] / "deploy/arena/receive.py"
SHA = "a" * 40


@pytest.mark.parametrize(
    "failure,has_previous,rollback_fails",
    [
        (None, True, False),
        ("restart", True, False),
        ("health", True, False),
        ("restart", False, False),
        ("health", False, False),
        ("restart", True, True),
    ],
)
def test_receiver_rolls_back_restart_and_health_failures(
    tmp_path, monkeypatch, failure, has_previous, rollback_fails
):
    releases = tmp_path / "releases"
    releases.mkdir()
    previous = releases / ("b" * 40)
    if has_previous:
        previous.mkdir()
        (tmp_path / "current").symlink_to(previous)
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        data = (SHA + "\n").encode()
        entry = tarfile.TarInfo("SOURCE_COMMIT")
        entry.size = len(data)
        bundle.addfile(entry, io.BytesIO(data))
    archive.seek(0)
    monkeypatch.setenv("SSH_ORIGINAL_COMMAND", "deploy " + SHA)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=archive))
    monkeypatch.setattr(
        shutil, "disk_usage", lambda _: SimpleNamespace(free=4 * 1024**3)
    )
    restarts = []

    def run(command, **kwargs):
        assert kwargs["check"] is True
        if command[0] == "sudo":
            restarts.append((tmp_path / "current").resolve())
            if failure == "restart" and (len(restarts) == 1 or rollback_fails):
                raise subprocess.CalledProcessError(1, command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", run)
    health = Mock(
        side_effect=OSError("fixture unhealthy") if failure == "health" else None,
        return_value=nullcontext(SimpleNamespace(status=200)),
    )
    monkeypatch.setattr(urllib.request, "urlopen", health)
    monkeypatch.setattr("time.sleep", lambda _: None)
    # Substitute only the installation root; execute the actual receiver control flow.
    source = RECEIVER.read_text()
    root_assignment = 'ROOT = Path("/opt/ampixa-arena")'
    assert source.count(root_assignment) == 1
    source = source.replace(root_assignment, f"ROOT = Path({str(tmp_path)!r})")
    script = tmp_path / "receive.py"
    script.write_text(source)
    spec = importlib.util.spec_from_file_location("receiver_probe", script)
    module = importlib.util.module_from_spec(spec)
    try:
        if failure:
            with pytest.raises(SystemExit) as error:
                spec.loader.exec_module(module)
            message = str(error.value)
            assert (
                "restart failed" if failure == "restart" else "Health check failed"
            ) in message
            if rollback_fails:
                assert "rollback failed" in message
                assert "previous application restored" not in message
            elif has_previous:
                assert "previous application restored" in message
            else:
                assert "no previous application to restore" in message
        else:
            spec.loader.exec_module(module)
        release = releases / SHA
        expected = previous if failure and has_previous else release
        assert (tmp_path / "current").resolve() == expected
        assert restarts == (
            [release, previous] if failure and has_previous else [release]
        )
        assert health.call_count == (
            0 if failure == "restart" else 30 if failure == "health" else 1
        )
    finally:
        if hasattr(module, "lock"):
            module.lock.close()
