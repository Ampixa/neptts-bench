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
        CREATE TABLE IF NOT EXISTS user_sessions (token_hash TEXT PRIMARY KEY, github_id TEXT NOT NULL, login TEXT NOT NULL, expires REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS oauth_states (state_hash TEXT PRIMARY KEY, binding_hash TEXT NOT NULL, verifier TEXT NOT NULL, expires REAL NOT NULL, callback TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS rate_limits (key TEXT PRIMARY KEY, start REAL NOT NULL, count INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS profile_evaluations (
          submission_id TEXT NOT NULL, profile TEXT NOT NULL,
          created REAL NOT NULL, updated REAL NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('awaiting_review','published','rejected')),
          report TEXT NOT NULL, reviewer_note TEXT NOT NULL DEFAULT '',
          PRIMARY KEY(submission_id, profile));
        CREATE TABLE IF NOT EXISTS arena_consent (
          submission_id TEXT PRIMARY KEY, granted INTEGER NOT NULL,
          updated REAL NOT NULL, via TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS arena_listeners (
          id TEXT PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE, created REAL NOT NULL,
          native_declared INTEGER NOT NULL, screened REAL, screen_attempts INTEGER NOT NULL DEFAULT 0);
        CREATE TABLE IF NOT EXISTS arena_screen_items (
          token_hash TEXT PRIMARY KEY, listener_id TEXT NOT NULL, attempt INTEGER NOT NULL,
          submission_id TEXT NOT NULL, sent_id TEXT NOT NULL, options TEXT NOT NULL,
          answer INTEGER NOT NULL, created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS arena_battles (
          id INTEGER PRIMARY KEY, token_hash TEXT NOT NULL UNIQUE, listener_id TEXT NOT NULL,
          sent_id TEXT NOT NULL, sub_a TEXT NOT NULL, sub_b TEXT NOT NULL,
          kind TEXT NOT NULL CHECK(kind IN ('compare','null','degraded')),
          degraded_side TEXT CHECK(degraded_side IN ('a','b')),
          served REAL NOT NULL, fetched_a REAL, fetched_b REAL,
          duration_a REAL, duration_b REAL);
        CREATE INDEX IF NOT EXISTS arena_battles_listener ON arena_battles(listener_id);
        CREATE TABLE IF NOT EXISTS arena_votes (
          battle_id INTEGER PRIMARY KEY REFERENCES arena_battles(id), listener_id TEXT NOT NULL,
          choice TEXT NOT NULL CHECK(choice IN ('a','b','both_good','both_bad')),
          tags TEXT NOT NULL, created REAL NOT NULL, elapsed REAL NOT NULL,
          valid INTEGER NOT NULL, reason TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS donors (
          id TEXT PRIMARY KEY, session_hash TEXT UNIQUE, recovery_hash TEXT NOT NULL UNIQUE,
          created REAL NOT NULL, consent_version TEXT NOT NULL, consent_sha256 TEXT NOT NULL,
          purposes TEXT NOT NULL, profile TEXT NOT NULL, mic_check TEXT,
          paid_status TEXT NOT NULL DEFAULT 'volunteer'
            CHECK(paid_status IN ('volunteer','applied','approved','rejected')),
          payout TEXT, rate_npr_per_minute REAL, withdrawn REAL);
        CREATE TABLE IF NOT EXISTS donor_recordings (
          id TEXT PRIMARY KEY, donor_id TEXT NOT NULL REFERENCES donors(id),
          prompt_set TEXT NOT NULL, prompt_id TEXT NOT NULL, created REAL NOT NULL,
          seconds REAL NOT NULL, sha256 TEXT NOT NULL, signal TEXT NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('asr_pending','accepted','needs_review','rejected')),
          asr_profile TEXT, cer REAL, reviewer_note TEXT NOT NULL DEFAULT '', reviewed REAL,
          UNIQUE(donor_id, prompt_set, prompt_id));
        CREATE INDEX IF NOT EXISTS donor_recordings_state ON donor_recordings(state);
        CREATE TABLE IF NOT EXISTS donor_hub_deletions (
          id INTEGER PRIMARY KEY, repo TEXT NOT NULL, path TEXT NOT NULL, sha256 TEXT NOT NULL,
          created REAL NOT NULL, committed REAL);
        CREATE TABLE IF NOT EXISTS donor_hub_sync (
          repo TEXT PRIMARY KEY, metadata_sha TEXT, last_sync REAL, last_error TEXT);
        CREATE TABLE IF NOT EXISTS worker_heartbeats (
          engine TEXT PRIMARY KEY, role TEXT NOT NULL, profile TEXT NOT NULL, pid INTEGER NOT NULL,
          started REAL NOT NULL, last_seen REAL NOT NULL, last_work REAL);
        CREATE TABLE IF NOT EXISTS system_kinds (
          submission_id TEXT PRIMARY KEY REFERENCES submissions(id),
          kind TEXT NOT NULL CHECK(kind IN ('baseline','human_reference')),
          note TEXT NOT NULL DEFAULT '', created REAL NOT NULL);
        CREATE TABLE IF NOT EXISTS donor_payouts (
          id INTEGER PRIMARY KEY, donor_id TEXT NOT NULL REFERENCES donors(id),
          amount_npr REAL NOT NULL CHECK(amount_npr > 0), reference TEXT NOT NULL,
          note TEXT NOT NULL DEFAULT '', created REAL NOT NULL);
        """)

        columns = {r[1] for r in c.execute("PRAGMA table_info(submissions)")}
        for name, kind in (("owner_id", "TEXT"),
                           ("attempts", "INTEGER NOT NULL DEFAULT 0"),
                           ("auto_review", "TEXT")):
            if name not in columns:
                c.execute(f"ALTER TABLE submissions ADD COLUMN {name} {kind}")
        c.execute(
            "CREATE INDEX IF NOT EXISTS submissions_owner ON submissions(owner_id)"
        )
        if "region" not in {r[1] for r in c.execute("PRAGMA table_info(arena_listeners)")}:
            c.execute("ALTER TABLE arena_listeners ADD COLUMN region TEXT")
        # Where a donated take's audio is: local (dev), spool (waiting for Hugging Face) or hub.
        if "stored" not in {r[1] for r in c.execute("PRAGMA table_info(donor_recordings)")}:
            c.execute("ALTER TABLE donor_recordings ADD COLUMN stored TEXT NOT NULL DEFAULT 'local'")


def event(c, sid, state, message):
    c.execute(
        "INSERT INTO events(submission_id,time,state,message) VALUES(?,?,?,?)",
        (sid, time.time(), state, message),
    )


def unpack(row):
    d = dict(row)
    for k in ("manifest", "validation", "report", "auto_review"):
        d[k] = json.loads(d[k]) if d.get(k) else None
    d.pop("token_hash", None)
    return d
