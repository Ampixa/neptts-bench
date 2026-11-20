# Agent guide: NepTTS-Bench / Ampixa Benchmarks Arena

This guide is for coding agents (Claude Code, Codex, …) working on this repo, especially an
**owner agent preparing or operating the production deployment** of https://ampixa.benchmarks.cloud.
Human-oriented detail lives in [deploy/arena/PRODUCTION_SETUP.md](deploy/arena/PRODUCTION_SETUP.md).

## Map

| Path | What it is |
|---|---|
| `submission_app/` | FastAPI web app (`server.py`), worker (`worker.py`), arena (`arena.py`, `bt.py`), donations, auto-review, readiness |
| `submission_app/static/` | Vanilla JS SPA. Theme = `style.css` → `dashboard.css` → `arena.css` → `donate.css` (later files override). Strict CSP: **no inline styles or scripts** (set styles via JS `el.style`, never `style="…"`) |
| `neptts-eval/` | Evaluator toolkit; `src/neptts_eval/rani_contract.json` pins the primary judge |
| `deploy/arena/` | systemd units, receiver, nginx, backup, `environment.example`, setup checklist |
| `seeds/` | Manifest examples for `python -m submission_app.seed` |
| `benchmark/sentences.json` | The 205-prompt contract. Changing it changes the benchmark ID; don't edit casually |

## The one command that tells you what to do

```bash
# On the VM (as the service user, with the production environment loaded):
sudo -u ampixa-arena bash -c 'set -a; . /etc/ampixa-arena/environment; set +a; \
  cd /opt/ampixa-arena/current && .venv/bin/python -m submission_app.doctor --data /var/lib/ampixa-arena --json'
# Locally against a scratch dir:
python -m submission_app.doctor --data .local/submissions --json
```

Exit code 0 = launch-ready. Each check has `id`, `status` (`ok` / `todo` / `warn` / `info`), `detail` and `fix`.
Work through every `todo`, re-run, repeat. Reviewers see the same list at `GET /api/admin/readiness`.

## Check id → where to fix it

| `id` | Fix | Needs a human? |
|---|---|---|
| `origin` | `NEPTTS_ORIGIN=https://ampixa.benchmarks.cloud` in `/etc/ampixa-arena/environment` | No |
| `admin_password` | `NEPTTS_ADMIN_PASSWORD` ≥ 24 chars (`python3 -c 'import secrets;print(secrets.token_urlsafe(32))'`) | Share it with reviewers |
| `github_oauth` | `NEPTTS_GITHUB_CLIENT_ID/SECRET` from an OAuth App owned by the Ampixa org; callback `<origin>/api/auth/github/callback` | **Yes**: org admin creates the app |
| `primary_judge` | `neptts-eval/src/neptts_eval/rani_contract.json` must have `checkpoint_sha256` + `packages` | **Yes**: ML lead approves the pin |
| `worker` | Install worker venv + `deploy/arena/ampixa-arena-worker.service` (PRODUCTION_SETUP §4). Same host as the DB | VM needs ≥ 8 GiB RAM |
| `auto_review` | `NEPTTS_AUTO_REVIEW` = `shadow` first, `on` after ≥ 10 agreeing shadow verdicts | **Yes**: policy decision |
| `arena_systems` | `python -m submission_app.seed --kind baseline` (or `--kind human_reference`) (≥ 4 systems incl. a human reference) | Generating vendor audio needs API keys |
| `donation_consent` | `submission_app/consent/voice_donation_v1.json` → `"status": "approved"` | **Yes**: legal + native-speaker review. **Never flip this yourself** |
| `hf_storage` / `hf_private` | `scripts/create_hf_donation_repos.py --org Ampixa`, then the three `NEPTTS_HF_*`/`HF_TOKEN` vars | Org admin token |
| `ffmpeg` | `apt install ffmpeg` | No |
| `disk` | Keep > 5 GiB free on the data volume | No |
| `backup` | Install `deploy/arena/backup.sh` + timer; needs `/etc/ampixa-arena/backup.env` | **Yes**: owns the age key and off-site bucket |
| `open_votes` | `NEPTTS_ARENA_PUBLIC_EXPORT=on` once the privacy notice covers it | **Yes**: policy decision |

After changing `/etc/ampixa-arena/environment`: `sudo systemctl restart ampixa-arena ampixa-arena-worker`.

## Configuration surface (all of it)

Every runtime setting is an environment variable; the full annotated list is
[deploy/arena/environment.example](deploy/arena/environment.example). Code constants worth knowing:

| Constant | File | Meaning |
|---|---|---|
| `DAILY_CAP`, `LIFETIME_CAP`, `IP_DAILY_CAP`, `VOTE_WEIGHT_CAP` | `submission_app/arena.py` | Anti-gaming vote limits |
| `RATINGS_REFRESH`, `MIN_CATEGORY_VOTES`, `PROVISIONAL_VOTES` | `submission_app/arena.py` | Board refresh and display thresholds |
| `MAX_SHARED_AUDIO`, `MAX_FAILED_SHARE`, `PLAUSIBLE_FLOOR` | `submission_app/autoreview.py` | What auto-review holds for a person |
| `WORKER_STALE`, `BACKUP_STALE` | `submission_app/readiness.py` | When doctor reports worker/backups as down |

## Hard rules

- **Secrets never go in the repo, CI logs or chat.** They live only in `/etc/ampixa-arena/environment` (root:ampixa-arena, 640) and GitHub environment secrets (`arena-production`).
- **Don't change the Rani pin or `sentences.json` to make a check pass.** Re-pinning creates a new judge profile; editing prompts creates a new benchmark. Both need owner sign-off.
- **Don't publish anything consent-gated.** Donation consent status and owners' listener-playback consent are human decisions.
- **SQLite is the source of truth** at `/var/lib/ampixa-arena/submissions.db`. Back it up before schema changes; migrations in `storage.py` must stay additive (rollback does not undo them).
- The deploy receiver restarts only `ampixa-arena.service`. After changes to `worker.py` or `neptts-eval/`, restart `ampixa-arena-worker` yourself.
- Pushes that touch `.github/workflows/` need a token with the `workflow` scope.

## Verify changes

```bash
pip install -r submission_app/requirements.txt pytest ruff
python -m pytest submission_app/tests neptts-eval/tests -q   # run together: submission_app puts neptts_eval on sys.path
ruff check submission_app
for f in submission_app/static/*.js; do node --check "$f"; done
node --test submission_app/tests/ranking.test.cjs
```

Local preview: `NEPTTS_ADMIN_PASSWORD=<12+ chars> NEPTTS_ORIGIN=http://localhost:5193 python -m submission_app.server`
(the origin must match the browser's address bar or POSTs are rejected as cross-origin).
