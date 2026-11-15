"""Where donated audio lives: private Hugging Face datasets, with a short local spool.

Without configuration, takes stay on local disk (development and tests). With
NEPTTS_HF_DONATIONS_BENCHMARK_REPO set, the web server writes each take to a spool
folder and `sync` commits spooled takes to private Hugging Face dataset repos in
batches, then deletes the local copy. Playback and the ASR worker read from the
spool while a take is there, otherwise from Hugging Face.

Two repos keep evaluation-only recordings apart from training material:
- benchmark repo: takes of the anchor (benchmark) prompts; a small group only.
- open repo: takes of licensed open prompt sets; needed only once those exist.

Deleting a take that already reached Hugging Face queues a delete commit. After
that commit, the audio object is permanently erased from the repo's storage and
history (`permanently_delete_lfs_files`, matched by the SHA-256 we keep for every
take), so a withdrawal does not survive in old revisions.
"""

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from .storage import db

ANCHOR = "anchor-v1"
SPOOL_CAP = 2 * 1024**3  # stop accepting takes once this much is waiting for Hugging Face
MIN_FREE = 2 * 1024**3  # same free-disk floor as submission uploads
BATCH = 200  # takes per commit
SYNC_SECONDS = 300
log = logging.getLogger(__name__)


class StorageUnavailable(RuntimeError):
    pass


class LocalStore:
    """Takes stay on this machine. Also reads spooled takes left by a HubStore."""

    remote = False

    def __init__(self, root):
        self.root = Path(root)
        for name in ("donations", "donation_spool"):
            folder = self.root / name
            folder.mkdir(parents=True, exist_ok=True)
            folder.chmod(0o700)

    def local_path(self, rec):
        if rec["stored"] == "local":
            return self.root / "donations" / rec["donor_id"] / f"{rec['id']}.wav"
        if rec["stored"] == "spool":
            return self.root / "donation_spool" / rec["donor_id"] / f"{rec['id']}.wav"
        return None

    def save(self, donor_id, rec_id, body):
        """Write a new take; returns the `stored` value for its database row."""
        folder = self.root / "donations" / donor_id
        folder.mkdir(exist_ok=True)
        (folder / f"{rec_id}.wav").write_bytes(body)
        return "local"

    def check_capacity(self):
        if _free(self.root) < MIN_FREE:
            raise StorageUnavailable("Recording storage is temporarily full.")

    def read(self, rec):
        path = self.local_path(rec)
        if path is not None and path.exists():
            return path.read_bytes()
        return self.read_remote(rec)

    def read_remote(self, rec):
        raise FileNotFoundError(rec["id"])

    @contextmanager
    def local_copy(self, rec):
        """A file path holding the take, for tools that read files (the ASR worker)."""
        path = self.local_path(rec)
        if path is not None and path.exists():
            yield path
            return
        with tempfile.TemporaryDirectory(prefix="take-", dir=self.root) as tmp:
            copy = Path(tmp) / "take.wav"
            copy.write_bytes(self.read_remote(rec))
            yield copy

    def remove(self, c, rec):
        """Delete one take's audio. Call inside the transaction that deletes its row."""
        path = self.local_path(rec)
        if path is not None:
            path.unlink(missing_ok=True)

    def status(self, root):
        return {"backend": "local"}


class HubStore(LocalStore):
    """Spool locally, keep in private Hugging Face datasets."""

    remote = True

    def __init__(self, root, client, benchmark_repo, open_repo=None):
        super().__init__(root)
        if open_repo == benchmark_repo:
            raise ValueError("Benchmark and open donation repos must be different.")
        self.client, self.benchmark_repo, self.open_repo = client, benchmark_repo, open_repo

    def repos(self):
        return [r for r in (self.benchmark_repo, self.open_repo) if r]

    def repo_for(self, rec):
        if rec["prompt_set"] == ANCHOR:
            return self.benchmark_repo
        if not self.open_repo:
            raise ValueError("Set NEPTTS_HF_DONATIONS_OPEN_REPO before collecting open prompt sets.")
        return self.open_repo

    @staticmethod
    def remote_path(rec):
        return f"audio/{rec['donor_id']}/{rec['id']}.wav"

    def save(self, donor_id, rec_id, body):
        folder = self.root / "donation_spool" / donor_id
        folder.mkdir(exist_ok=True)
        (folder / f"{rec_id}.wav").write_bytes(body)
        return "spool"

    def check_capacity(self):
        super().check_capacity()
        spooled = sum(p.stat().st_size for p in (self.root / "donation_spool").rglob("*.wav"))
        if spooled > SPOOL_CAP:
            raise StorageUnavailable("Recording storage is temporarily unavailable.")

    def read_remote(self, rec):
        if rec["stored"] != "hub":
            raise FileNotFoundError(rec["id"])
        return self.client.read(self.repo_for(rec), self.remote_path(rec))

    def remove(self, c, rec):
        super().remove(c, rec)
        if rec["stored"] == "hub":
            c.execute("INSERT INTO donor_hub_deletions(repo,path,sha256,created) VALUES(?,?,?,?)",
                      (self.repo_for(rec), self.remote_path(rec), rec["sha256"], time.time()))

    def status(self, root):
        with db(root) as c:
            spool = c.execute(
                "SELECT COUNT(*) AS n, MIN(created) AS oldest FROM donor_recordings "
                "WHERE stored IN ('spool','local')").fetchone()
            pending = c.execute("SELECT COUNT(*) FROM donor_hub_deletions").fetchone()[0]
            repos = {r["repo"]: {"last_sync": r["last_sync"], "last_error": r["last_error"]}
                     for r in c.execute("SELECT * FROM donor_hub_sync")}
        return {"backend": "huggingface", "repos": {r: repos.get(r, {}) for r in self.repos()},
                "spooled": spool["n"],
                "oldest_spooled_seconds": round(time.time() - spool["oldest"]) if spool["oldest"] else None,
                "pending_deletions": pending}


class HubClient:
    """The few Hugging Face calls the store needs. Tests replace it with a fake."""

    def __init__(self, token=None):
        from huggingface_hub import HfApi, HfFileSystem

        self.api, self.fs = HfApi(token=token), HfFileSystem(token=token)

    def ensure_private(self, repo):
        if not self.api.dataset_info(repo).private:
            raise PermissionError(f"{repo} is not private. Refusing to upload donor audio.")

    def commit(self, repo, adds, deletes, message):
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete

        ops = [CommitOperationAdd(path_in_repo=p, path_or_fileobj=src) for p, src in adds]
        ops += [CommitOperationDelete(path_in_repo=p) for p in deletes]
        self.api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                               commit_message=message)

    def read(self, repo, path):
        return self.fs.read_bytes(f"datasets/{repo}/{path}")

    def purge(self, repo, sha256s):
        """Erase stored objects with these SHA-256s from storage and history. Returns the count."""
        doomed = [f for f in self.api.list_lfs_files(repo, repo_type="dataset")
                  if f.file_oid in sha256s]
        if doomed:
            self.api.permanently_delete_lfs_files(repo, doomed, rewrite_history=True,
                                                  repo_type="dataset")
        else:
            # Objects not tracked as LFS: drop old revisions instead.
            self.api.super_squash_history(repo, repo_type="dataset",
                                          commit_message="Remove deleted donor audio from history")
        return len(doomed)


def from_env(root, open_sets=False):
    """The store this deployment uses. Local unless the benchmark repo is configured."""
    bench = os.environ.get("NEPTTS_HF_DONATIONS_BENCHMARK_REPO", "").strip()
    if not bench:
        return LocalStore(root)
    open_repo = os.environ.get("NEPTTS_HF_DONATIONS_OPEN_REPO", "").strip() or None
    if open_sets and not open_repo:
        raise ValueError("Open prompt sets are loaded: set NEPTTS_HF_DONATIONS_OPEN_REPO too.")
    return HubStore(root, HubClient(), bench, open_repo)


def _free(root):
    import shutil

    return shutil.disk_usage(root).free


def metadata(c, repo, store, sets, extra_ids=()):
    """metadata.jsonl for one repo, in Hugging Face's audio-folder layout (`file_name`)."""
    extra = set(extra_ids)
    rows = [r for r in c.execute("SELECT * FROM donor_recordings ORDER BY id")
            if (r["stored"] == "hub" or r["id"] in extra) and store.repo_for(r) == repo]
    lines = []
    for r in rows:
        signal = json.loads(r["signal"])
        lines.append(json.dumps({
            "file_name": store.remote_path(r), "recording_id": r["id"], "speaker_id": r["donor_id"],
            "prompt_set": r["prompt_set"], "prompt_id": r["prompt_id"],
            "text": sets.get(r["prompt_set"], {}).get("prompts", {}).get(r["prompt_id"], ""),
            "seconds": r["seconds"], "sample_rate": signal.get("sample_rate"),
            "snr_db": signal.get("snr_db"), "rani_cer": r["cer"], "state": r["state"],
            "month": time.strftime("%Y-%m", time.gmtime(r["created"])),
        }, ensure_ascii=False, sort_keys=True))
    return ("\n".join(lines) + "\n" if lines else "").encode()


def sync(root, store, sets, now=time.time):
    """Push spooled takes and queued deletions to Hugging Face. Returns a per-repo report."""
    if not store.remote:
        return {}
    report = {}
    for repo in store.repos():
        try:
            report[repo] = _sync_repo(root, store, sets, repo, now)
        except Exception as error:  # an outage must not lose takes: they stay spooled
            log.exception("Donation sync to %s failed", repo)
            with db(root) as c:
                _record(c, repo, last_error=f"{type(error).__name__}: {str(error)[:300]}")
            report[repo] = {"error": type(error).__name__}
    return report


def _record(c, repo, **fields):
    c.execute("INSERT OR IGNORE INTO donor_hub_sync(repo) VALUES(?)", (repo,))
    for key, value in fields.items():
        c.execute(f"UPDATE donor_hub_sync SET {key}=? WHERE repo=?", (value, repo))


def _sync_repo(root, store, sets, repo, now):
    with db(root) as c:
        pending = [r for r in c.execute(
            "SELECT * FROM donor_recordings WHERE stored IN ('spool','local') ORDER BY created")
            if store.repo_for(r) == repo][:BATCH]
        deletes = c.execute(
            "SELECT * FROM donor_hub_deletions WHERE repo=? AND committed IS NULL", (repo,)).fetchall()
        state = c.execute("SELECT * FROM donor_hub_sync WHERE repo=?", (repo,)).fetchone()
        meta = metadata(c, repo, store, sets, [r["id"] for r in pending])
    meta_sha = hashlib.sha256(meta).hexdigest()
    last_sha = state["metadata_sha"] if state and state["metadata_sha"] else hashlib.sha256(b"").hexdigest()
    changed = pending or deletes or meta_sha != last_sha
    uploaded = 0
    if changed:
        store.client.ensure_private(repo)
        adds = [(store.remote_path(r), str(store.local_path(r))) for r in pending]
        adds.append(("metadata.jsonl", meta))
        store.client.commit(repo, adds, [d["path"] for d in deletes],
                            f"Sync {len(pending)} donated takes, {len(deletes)} deletions")
        uploaded = len(pending)
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            for r in pending:
                if c.execute("SELECT 1 FROM donor_recordings WHERE id=?", (r["id"],)).fetchone():
                    c.execute("UPDATE donor_recordings SET stored='hub' WHERE id=?", (r["id"],))
                else:  # deleted by the donor while this commit was running
                    c.execute(
                        "INSERT INTO donor_hub_deletions(repo,path,sha256,created) VALUES(?,?,?,?)",
                        (repo, store.remote_path(r), r["sha256"], now()))
            c.executemany("UPDATE donor_hub_deletions SET committed=? WHERE id=?",
                          [(now(), d["id"]) for d in deletes])
            _record(c, repo, metadata_sha=meta_sha, last_sync=now(), last_error=None)
        for r in pending:
            store.local_path(r).unlink(missing_ok=True)
    purged = _purge(root, store, repo)
    return {"uploaded": uploaded, "deleted": len(deletes), "purged": purged}


def _purge(root, store, repo):
    """Erase the audio of committed deletions from Hugging Face storage and history."""
    with db(root) as c:
        done = c.execute(
            "SELECT * FROM donor_hub_deletions WHERE repo=? AND committed IS NOT NULL", (repo,)
        ).fetchall()
        live = {r["sha256"] for r in c.execute("SELECT sha256 FROM donor_recordings WHERE stored='hub'")}
    if not done:
        return 0
    doomed = {d["sha256"] for d in done} - live  # identical audio still kept elsewhere stays
    if doomed:
        store.client.purge(repo, doomed)
    with db(root) as c:
        c.executemany("DELETE FROM donor_hub_deletions WHERE id=?", [(d["id"],) for d in done])
    return len(done)


def start_sync_thread(app, root):
    """Background sync inside the web server, which already holds the spool."""
    seconds = int(os.environ.get("NEPTTS_HF_SYNC_SECONDS", SYNC_SECONDS))
    if seconds <= 0:
        return None

    def loop():
        while True:
            time.sleep(seconds)
            try:
                sync(root, app.state.donation_store, app.state.prompt_sets)
            except Exception:
                log.exception("Donation sync loop error")

    thread = threading.Thread(target=loop, name="donation-sync", daemon=True)
    thread.start()
    return thread
