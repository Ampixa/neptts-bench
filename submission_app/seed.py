"""Add an Ampixa first-party system (a baseline or a human reading) to the arena.

First-party entries go through exactly the same path as public submissions: the
same audio checks, Rani scoring and automatic review. The only differences are
that Ampixa owns them, listener playback is allowed, and the arena labels them.

    python -m submission_app.seed --data /var/lib/ampixa-arena \\
        --wav-dir outputs/tingting_asmita --kind baseline \\
        --manifest seeds/tingting_asmita.json

The WAV directory must hold `<sent_id>.wav` for every prompt in the benchmark
(mono PCM16, 16–48 kHz). The manifest JSON holds the submission fields; see
`seeds/example_baseline.json`. Generation scripts live in `scripts/generate_*.py`.

Use `--kind human_reference` for native speakers reading the prompts. It anchors
the scale: a system's Elo becomes "this far from a real person".
"""

import argparse
import hashlib
import json
import secrets
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from . import autoreview
from .contract import SENTENCES, template, validate_archive
from .storage import db, event, initialize

OWNER = "ampixa-first-party"


def seed(root, wav_dir, fields, kind, note=""):
    """Validate and queue one first-party system. Returns (submission id, receipt token)."""
    root, wav_dir = Path(root), Path(wav_dir)
    initialize(root)
    manifest = {**template(), **fields,
                "permission_to_evaluate_and_publish": True,
                "permission_to_play_audio_to_listeners": True}
    missing = [s for s in SENTENCES if not (wav_dir / f"{s}.wav").is_file()]
    if missing:
        raise ValueError(f"{len(missing)} prompts have no WAV in {wav_dir}, e.g. {', '.join(missing[:5])}.")
    work = Path(tempfile.mkdtemp(prefix="seed-", dir=root))
    sid = uuid.uuid4().hex
    target = root / "uploads" / sid
    try:
        archive = work / "seed.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("submission.json", json.dumps(manifest))
            for s in SENTENCES:
                z.write(wav_dir / f"{s}.wav", f"audio/{s}.wav")
        manifest, validation = validate_archive(archive, work / "audio")
        token = secrets.token_urlsafe(32)
        now = time.time()
        auto = autoreview.enabled()
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            if c.execute("SELECT 1 FROM submissions WHERE audio_sha256=?",
                         (validation["audio_sha256"],)).fetchone():
                raise ValueError("This exact audio is already in the arena.")
            shutil.move(str(work / "audio"), target)
            c.execute(
                "INSERT INTO submissions(id,token_hash,created,updated,state,manifest,validation,"
                "audio_sha256,message,owner_id) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (sid, hashlib.sha256(token.encode()).hexdigest(), now, now,
                 "queued" if auto else "validated", json.dumps(manifest), json.dumps(validation),
                 validation["audio_sha256"],
                 "First-party entry. Queued for evaluation." if auto
                 else "First-party entry. Waiting for a reviewer to queue it.", OWNER),
            )
            c.execute("INSERT INTO system_kinds VALUES(?,?,?,?)", (sid, kind, note, now))
            c.execute("INSERT OR REPLACE INTO arena_consent VALUES(?,?,?,?)", (sid, 1, now, "first_party"))
            event(c, sid, "validated", f"First-party {kind} added by an operator; "
                                       f"all {len(SENTENCES)} prompts passed audio checks.")
            if auto:
                event(c, sid, "queued", f"Queued automatically by {autoreview.VERSION}.")
        return sid, token
    except Exception:
        shutil.rmtree(target, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(work, ignore_errors=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="JSON with the submission fields.")
    parser.add_argument("--kind", choices=["baseline", "human_reference"], required=True)
    parser.add_argument("--note", default="", help="Internal note, e.g. how and when audio was generated.")
    args = parser.parse_args(argv)
    try:
        sid, token = seed(args.data, args.wav_dir, json.loads(args.manifest.read_text()), args.kind, args.note)
    except (ValueError, OSError) as error:
        print(f"Not added: {error}", file=sys.stderr)
        return 1
    print(f"Added {args.kind} {sid}. Receipt token (keep it private): {token}")
    print("The Rani worker scores it next; it appears in the arena once published.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
