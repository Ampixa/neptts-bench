"""Donated audio on private Hugging Face datasets: spool, sync, playback, deletion, outages."""

import hashlib
import json
from pathlib import Path

import pytest

from submission_app import donation_store
from submission_app.donation_store import HubStore, LocalStore, from_env, sync
from submission_app.storage import db
from submission_app.tests.test_donations import browser, enroll, make_app, record, take
from submission_app.tests.test_flow import login
from submission_app.worker import process_donation

BENCH = "ampixa/neptts-donations-benchmark"


class FakeHub:
    """In-memory stand-in for the Hugging Face calls HubStore makes."""

    def __init__(self):
        self.files, self.objects, self.commits, self.purged = {}, {}, [], []
        self.private, self.down, self.during_commit = True, False, None

    def ensure_private(self, repo):
        if not self.private:
            raise PermissionError(f"{repo} is public")

    def commit(self, repo, adds, deletes, message):
        if self.down:
            raise ConnectionError("Hugging Face unreachable")
        uploads = [(path, src if isinstance(src, bytes) else Path(src).read_bytes())
                   for path, src in adds]  # file contents are sent before the commit lands
        if self.during_commit:
            self.during_commit()
        files = self.files.setdefault(repo, {})
        for path, data in uploads:
            files[path] = data
            self.objects.setdefault(repo, set()).add(hashlib.sha256(data).hexdigest())
        for path in deletes:
            del files[path]  # the real API also fails on a missing path
        self.commits.append(message)

    def read(self, repo, path):
        try:
            return self.files[repo][path]
        except KeyError:
            raise FileNotFoundError(path) from None

    def purge(self, repo, sha256s):
        self.purged.append(set(sha256s))
        self.objects[repo] -= set(sha256s)
        return len(sha256s)


@pytest.fixture
def app(tmp_path):
    return make_app(tmp_path)


@pytest.fixture
def hub(app):
    fake = FakeHub()
    app.state.donation_store = HubStore(app.state.root, fake, BENCH)
    return fake


def synced(app):
    return sync(app.state.root, app.state.donation_store, app.state.prompt_sets)


def rows(app):
    with db(app.state.root) as c:
        return {r["id"]: dict(r) for r in c.execute("SELECT * FROM donor_recordings")}


def audio_files(hub):
    return sorted(p for p in hub.files.get(BENCH, {}) if p.startswith("audio/"))


def spool(app):
    return sorted((app.state.root / "donation_spool").rglob("*.wav"))


def donor(app, gain=0.1):
    c = browser(app)
    assert enroll(c).status_code == 200
    _, r = record(c, take(gain=gain))
    assert r.status_code == 200, r.text
    return c, r.json()["id"]


def test_take_is_spooled_then_synced_to_the_private_repo(app, hub):
    c, rec = donor(app)
    assert rows(app)[rec]["stored"] == "spool" and len(spool(app)) == 1
    assert c.get(f"/api/donate/recordings/{rec}/audio").content == take()  # from the spool
    report = synced(app)
    assert report[BENCH]["uploaded"] == 1
    assert rows(app)[rec]["stored"] == "hub" and spool(app) == []
    assert audio_files(hub) == [f"audio/{rows(app)[rec]['donor_id']}/{rec}.wav"]
    assert c.get(f"/api/donate/recordings/{rec}/audio").content == take()  # now from Hugging Face
    meta = [json.loads(line) for line in hub.files[BENCH]["metadata.jsonl"].decode().splitlines()]
    assert meta[0]["recording_id"] == rec and meta[0]["text"] and meta[0]["state"] == "asr_pending"
    assert not {"payout", "device", "profile", "recovery_hash", "age_band"} & set(meta[0])
    commits = len(hub.commits)
    synced(app)
    assert len(hub.commits) == commits  # nothing changed, nothing committed


def test_worker_scores_a_take_that_lives_on_hugging_face(app, hub):
    _, rec = donor(app)
    synced(app)
    seen = {}

    def score(path, text):
        seen["path"], seen["data"] = Path(path), Path(path).read_bytes()
        return 0.05

    assert process_donation(app.state.root, app.state.prompt_sets, score, "rani-test", set(),
                            app.state.donation_store)
    assert seen["data"] == take() and not seen["path"].exists()  # temporary copy removed
    assert rows(app)[rec]["state"] == "accepted"
    synced(app)  # the state change reaches the metadata
    assert '"state": "accepted"' in hub.files[BENCH]["metadata.jsonl"].decode()


def test_deleting_a_synced_take_removes_it_and_erases_its_history(app, hub):
    c, rec = donor(app)
    synced(app)
    sha = rows(app)[rec]["sha256"]
    assert c.delete(f"/api/donate/recordings/{rec}").status_code == 200
    report = synced(app)
    assert report[BENCH] == {"uploaded": 0, "deleted": 1, "purged": 1}
    assert audio_files(hub) == [] and hub.purged == [{sha}] and sha not in hub.objects[BENCH]
    with db(app.state.root) as conn:
        assert conn.execute("SELECT COUNT(*) FROM donor_hub_deletions").fetchone()[0] == 0


def test_withdrawal_reaches_synced_and_spooled_takes(app, hub):
    c, _ = donor(app, gain=0.1)
    synced(app)
    _, r = record(c, take(gain=0.12))  # still in the spool
    assert r.status_code == 200
    assert c.post("/api/donate/withdraw").status_code == 200
    synced(app)
    assert audio_files(hub) == [] and spool(app) == [] and rows(app) == {}


def test_outage_keeps_takes_spooled_and_is_reported(app, hub):
    _, rec = donor(app)
    hub.down = True
    assert synced(app)[BENCH] == {"error": "ConnectionError"}
    assert rows(app)[rec]["stored"] == "spool" and len(spool(app)) == 1
    admin = browser(app)
    login(admin)
    storage = admin.get("/api/admin/donations").json()["storage"]
    assert storage["spooled"] == 1 and "ConnectionError" in storage["repos"][BENCH]["last_error"]
    hub.down = False
    synced(app)
    assert rows(app)[rec]["stored"] == "hub"
    assert admin.get("/api/admin/donations").json()["storage"]["repos"][BENCH]["last_error"] is None


def test_public_repo_is_never_written(app, hub):
    _, rec = donor(app)
    hub.private = False
    assert synced(app)[BENCH] == {"error": "PermissionError"}
    assert hub.files == {} and rows(app)[rec]["stored"] == "spool"


def test_take_deleted_while_its_upload_runs_is_cleaned_up(app, hub):
    c, rec = donor(app)
    hub.during_commit = lambda: c.delete(f"/api/donate/recordings/{rec}")
    synced(app)  # uploads it, then finds the row gone and queues a deletion
    hub.during_commit = None
    assert len(audio_files(hub)) == 1
    synced(app)
    assert audio_files(hub) == [] and rows(app) == {}


def test_full_spool_refuses_new_takes(app, hub, monkeypatch):
    donor(app)
    monkeypatch.setattr(donation_store, "SPOOL_CAP", 10)
    c = browser(app)
    assert enroll(c).status_code == 200
    _, r = record(c, take(gain=0.13))
    assert r.status_code == 503


def test_configuration_from_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("NEPTTS_HF_DONATIONS_BENCHMARK_REPO", raising=False)
    assert type(from_env(tmp_path)) is LocalStore
    monkeypatch.setenv("NEPTTS_HF_DONATIONS_BENCHMARK_REPO", BENCH)
    store = from_env(tmp_path)
    assert store.remote and store.benchmark_repo == BENCH and store.open_repo is None
    with pytest.raises(ValueError, match="OPEN_REPO"):
        from_env(tmp_path, open_sets=True)
    monkeypatch.setenv("NEPTTS_HF_DONATIONS_OPEN_REPO", BENCH)
    with pytest.raises(ValueError, match="different"):
        from_env(tmp_path)


def test_hub_client_calls_the_real_api_correctly():
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    from submission_app.donation_store import HubClient

    client = HubClient(token="hf_test")
    client.api = MagicMock()
    client.api.dataset_info.return_value = SimpleNamespace(private=False)
    with pytest.raises(PermissionError):
        client.ensure_private(BENCH)
    client.commit(BENCH, [("audio/d/r.wav", b"RIFF"), ("metadata.jsonl", b"{}\n")],
                  ["audio/d/old.wav"], "Sync")
    kwargs = client.api.create_commit.call_args.kwargs
    assert kwargs["repo_type"] == "dataset" and kwargs["repo_id"] == BENCH
    assert [type(op) for op in kwargs["operations"]] == [
        CommitOperationAdd, CommitOperationAdd, CommitOperationDelete]
    keep, gone = SimpleNamespace(file_oid="a" * 64), SimpleNamespace(file_oid="b" * 64)
    client.api.list_lfs_files.return_value = [keep, gone]
    assert client.purge(BENCH, {"b" * 64}) == 1
    args = client.api.permanently_delete_lfs_files.call_args
    assert args.args[1] == [gone] and args.kwargs["rewrite_history"] is True
    client.api.list_lfs_files.return_value = [keep]
    assert client.purge(BENCH, {"c" * 64}) == 0
    client.api.super_squash_history.assert_called_once()
