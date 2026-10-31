#!/usr/bin/python3
"""Restricted SSH deployment entry point. Run as the Arena service account."""

import fcntl
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path("/opt/ampixa-arena")
command = os.environ.get("SSH_ORIGINAL_COMMAND", "")
if not re.fullmatch(r"deploy [0-9a-f]{40}", command):
    sys.exit("Only deploy <commit SHA> is accepted.")
sha = command.split()[1]
lock = (ROOT / ".deploy.lock").open("w")
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
release = ROOT / "releases" / sha
if release.exists():
    sys.exit("Release already exists; use a fresh commit or operator rollback.")
with tempfile.TemporaryDirectory(dir=ROOT) as temp:
    archive = Path(temp) / "release.tar"
    size = 0
    with archive.open("wb") as target:
        while chunk := sys.stdin.buffer.read(1024 * 1024):
            size += len(chunk)
            if size > 20 * 1024 * 1024:
                sys.exit("Archive too large.")
            target.write(chunk)
    if shutil.disk_usage(ROOT).free < 3 * 1024**3:
        sys.exit("At least 3 GiB free space is required.")
    release.mkdir()
    try:
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            if len(members) > 1000 or sum(m.size for m in members) > 30 * 1024**2:
                raise ValueError("Release exceeds limits")
            seen = set()
            for member in members:
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts or member.name in seen:
                    raise ValueError("Invalid release path")
                seen.add(member.name)
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Links and special files are forbidden")
                if path.parts[0] not in (
                    "submission_app",
                    "benchmark",
                    "neptts-eval",
                    "SOURCE_COMMIT",
                ):
                    raise ValueError("Unexpected release content")
                dest = release / path
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tar.extractfile(member) as src, dest.open("wb") as out:
                        shutil.copyfileobj(src, out)
        if (release / "SOURCE_COMMIT").read_text().strip() != sha:
            raise ValueError("Source identity mismatch")
        subprocess.run(
            ["/usr/bin/python3", "-m", "venv", str(release / ".venv")], check=True
        )
        subprocess.run(
            [
                str(release / ".venv/bin/pip"),
                "install",
                "--only-binary=:all:",
                "--disable-pip-version-check",
                "-r",
                str(release / "submission_app/requirements.txt"),
            ],
            check=True,
        )
        subprocess.run(
            [
                str(release / ".venv/bin/python"),
                "-c",
                "from submission_app.server import create_app",
            ],
            cwd=release,
            check=True,
        )
    except BaseException:
        shutil.rmtree(release)
        raise
previous = (ROOT / "current").resolve() if (ROOT / "current").exists() else None


def switch(path):
    link = ROOT / "next"
    link.unlink(missing_ok=True)
    link.symlink_to(path)
    link.replace(ROOT / "current")
    subprocess.run(
        ["sudo", "-n", "/usr/bin/systemctl", "restart", "ampixa-arena.service"],
        check=True,
    )


switch(release)
for _ in range(30):
    try:
        with urllib.request.urlopen(
            "http://127.0.0.1:5193/api/benchmark", timeout=2
        ) as response:
            if response.status == 200:
                print("Deployed " + sha)
                break
    except OSError:
        pass
    time.sleep(2)
else:
    if previous:
        switch(previous)
    sys.exit("Health check failed; previous application restored if present.")
# Keep current and previous plus recent releases; never remove persistent data.
releases = sorted(
    (ROOT / "releases").iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
)
for old in releases[5:]:
    if old not in (release, previous):
        shutil.rmtree(old)
