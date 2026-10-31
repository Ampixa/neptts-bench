"""SQLite state transitions: accepted audio -> evaluation -> independent approval."""

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def db(root: Path):
    conn = sqlite3.connect(root / "submissions.db", timeout=15)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def initialize(root):
    root.mkdir(parents=True, exist_ok=True)
    root.chmod(0o700)
    (root / "uploads").mkdir(exist_ok=True)
    with db(root) as c:
        c.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS submissions (
          id TEXT PRIMARY KEY, token_hash TEXT NOT NULL, created REAL NOT NULL,
          updated REAL NOT NULL, state TEXT NOT NULL, manifest TEXT NOT NULL,
          validation TEXT NOT NULL, audio_sha256 TEXT NOT NULL UNIQUE,
          report TEXT, message TEXT NOT NULL DEFAULT '', reviewer_note TEXT NOT NULL DEFAULT '',
          profile TEXT, progress INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS events (
          id INTEGER PRIMARY KEY, submission_id TEXT NOT NULL,
          time REAL NOT NULL, state TEXT NOT NULL, message TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rate_limits (key TEXT PRIMARY KEY, start REAL NOT NULL, count INTEGER NOT NULL);
        """)


def event(c, sid, state, message):
    c.execute(
        "INSERT INTO events(submission_id,time,state,message) VALUES(?,?,?,?)",
        (sid, time.time(), state, message),
    )


def unpack(row):
    d = dict(row)
    for k in ("manifest", "validation", "report"):
        d[k] = json.loads(d[k]) if d[k] else None
    d.pop("token_hash", None)
    return d
