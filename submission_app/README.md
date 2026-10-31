# Ampixa Benchmarks Arena (local implementation)

Ampixa Benchmarks Arena hosts NepTTS-Bench as its first benchmark. It is a separate FastAPI application for audio submissions, server-owned evaluation, private tracking and admin-approved publication. It does not change or deploy the existing `rating-app` service. No submissions or models are sent to an external evaluation API.

## Run locally

Python 3.11+, Node only for optional JavaScript syntax checking. Install the web service:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r submission_app/requirements.txt
export NEPTTS_ADMIN_PASSWORD='choose-a-long-local-password'
python -m submission_app.server
```

Open http://127.0.0.1:5193. Admin access uses the password above. There is no default password. The process binds only to loopback. The local preview uses SQLite and stores audio under `.local/submissions`, excluded from Git. `NEPTTS_DATA`, `NEPTTS_PORT` and `NEPTTS_ORIGIN` configure the root, port and exact browser origin. If changing the port, update the origin too.

For evaluation, install `submission_app/requirements-worker.txt` and FFmpeg, then run one worker:

```sh
pip install -r submission_app/requirements-worker.txt
python -m submission_app.worker --data .local/submissions --model small
```

The worker serializes jobs, runs CPU inference with two Torch threads, and holds a process lock. Model weights are downloaded by the trusted Whisper package and checksum-verified by its loader. Set `NEPTTS_MODEL_CACHE` for the model cache. Use `--once` for one queued job. Interrupted jobs become failed on worker restart; retry requires an admin decision. Logs stay on the operator's machine; raw exceptions are not shown to submitters.

`--model tiny` is a development-only profile. It runs genuine ASR but **cannot be approved for the public leaderboard**. Automated tests inject a deterministic scorer to exercise publication without downloading models or claiming real quality results.

## Submission contract

The service selects the 205 `sent_NNN` synthesis records in the checked-in `benchmark/sentences.json`, excluding 160 natural-speech references. IDs are not consecutive. The complete sentence content is hashed into the benchmark ID; old or mismatched manifests fail validation. This is a proposed new submission split, not a claim to reproduce the historical study subset. Freeze/review this split before production launch.

Download `/api/benchmark/pack` for the exact prompts, manifest template and instructions. A submission ZIP contains only `submission.json` and `audio/<sentence-id>.wav` for every required prompt. Audio must be mono PCM16 WAV, 16–48 kHz, 0.2–30 seconds. No missing IDs, extra files, duplicate ZIP paths, links, encrypted entries or silence. Metadata declares model version, voice, generation settings, post-processing, selection/retries, training overlap and permission to evaluate/publish. Do not upload model weights, keys or personal reference recordings.

To validate/package locally without evaluation models:

```sh
python -m submission_app.pack --audio ./outputs --manifest submission.json --output submission.zip
```

The helper refuses to overwrite an existing archive. Uploads are capped at 150 MiB compressed, 300 MiB expanded and 8 MiB per audio file. Extraction uses server-generated paths rather than archive paths. Only two uploads are processed concurrently; there is a 2 GiB free-space floor, a 100-submission capacity cap and a per-client hourly limit. These are initial operational limits, not a full internet-scale abuse service.

## Review and publication

1. GitHub sign-in is required to upload. Submissions belong to the immutable GitHub user ID and appear under My submissions across browsers. Successful validation returns a random private receipt token. The browser stores it locally and offers a private tracking link; retain it for optional bearer-link access. Earlier anonymous receipts remain valid, but are not automatically claimed by an account.
2. The reviewer reads declarations, can listen to protected audio and queues evaluation.
3. The worker verifies stored audio hashes, evaluates the full set and requires finite metrics for every prompt. Any failure leaves the entry unpublished.
4. The reviewer supplies a note and approves or rejects. Only an approved result appears on the public API/leaderboard.

The initial automated track uses Whisper small Nepali ASR round-trip macro CER/WER with a fixed normalization implementation. Its profile records model weight SHA, evaluator/normalizer/scorer code identity, package versions and inference settings. Each profile is ranked separately. CER/WER are intelligibility proxies, not human quality scores. NepaliMOS/SCOREQ and human MOS remain absent rather than using incompatible checkpoints or invented ratings. Later metrics must have explicit versions, full coverage and a new comparison profile.

Historical human MOS in the UI is a labelled snapshot of the source README, linked to the cloned commit, and is shown separately. It is not silently compared against new automated submissions.

Only approved metadata, per-sentence numeric metrics and evaluator provenance are public. Audio requires the receipt or an authenticated reviewer. Admin sessions are HttpOnly/SameSite cookies, expire after eight hours and are invalidated by server restart. Mutations require a custom header and valid origin when supplied. There is no cross-origin API access.

## Checks

```sh
pip install pytest httpx ruff
python -m pytest submission_app/tests -q
ruff check submission_app
node --check submission_app/static/app.js
```

Tests cover the full upload → queue → evaluate → approve → public-report flow; unauthorized access; failed/incomplete/development results; archive paths, duplicate paths, silence/stereo/missing IDs; exact benchmark identity; audio integrity; rejection; and cleanup. Fixture audio is synthetic test data, not a model-quality claim.

## Before deployment / suggested PR scope

One PR for the contract, portal, worker and tests is the coherent first change. Review locally first; no existing production service needs replacing. Deploy behind HTTPS with the configured public origin, a request-body cap, edge throttling, operator-managed credentials, backed-up private storage and a separately supervised worker with time/memory/disk budgets. Decide the submission retention policy and legacy receipt recovery mechanism before accepting external submissions. Add named reviewer accounts if review responsibility is shared.

The benchmark split, ASR choice and ranking policy need research sign-off. In particular, the repository itself documents weaknesses in Whisper for Nepali; this implementation makes the metric explicit rather than calling it an overall quality ranking. Existing full NepaliMOS weights are not silently substituted with a public head-only model. Production evaluator choices can change without changing the upload contract, provided they receive distinct profile identities.

Ampixa wordmark: copied from the organization's `https://ampixa.com/assets/wordmark.svg` for its own benchmark portal. The UI does not embed the separately licensed Devanagari website wordmark.

## GitHub sign-in setup

Create a dedicated GitHub OAuth App with homepage set to the Arena origin and callback exactly `<NEPTTS_ORIGIN>/api/auth/github/callback`. For local testing, use `http://127.0.0.1:5193/api/auth/github/callback`; use a separate OAuth app for production HTTPS.

Set `NEPTTS_GITHUB_CLIENT_ID` and `NEPTTS_GITHUB_CLIENT_SECRET` in the server environment, along with `NEPTTS_ORIGIN`, and restart the web service. Never commit the secret. Until configured, upload is closed and the UI explains that sign-in is awaiting setup. No mock login or bypass is exposed.

The authorization-code flow uses single-use, ten-minute state bound to an HttpOnly browser cookie plus PKCE S256. No repository or email scopes are requested. Provider access tokens are used only to obtain the public GitHub ID/login and are not retained. Hashed Arena sessions persist in SQLite for seven days, with HttpOnly/SameSite cookies (Secure on HTTPS). Logout revokes the current session. Reviewer login remains separate: GitHub login does not grant admin rights. Account IDs are not included in public leaderboard reports.

OAuth callback and account isolation tests use a mocked GitHub HTTP provider, including cross-browser callback rejection, replay, expiry, persistence, logout and owner-only audio. A real GitHub OAuth consent round trip still requires operator-provided app credentials. See [GitHub's authorization flow](https://docs.github.com/en/apps/oauth-apps/building-oauth-apps/authorizing-oauth-apps).

## Production domain

Portal name: **Ampixa Benchmarks Arena**. Intended production origin: `https://voices.benchmarks.cloud`.

- `NEPTTS_ORIGIN=https://voices.benchmarks.cloud`
- GitHub OAuth App homepage: `https://voices.benchmarks.cloud`
- GitHub OAuth callback: `https://voices.benchmarks.cloud/api/auth/github/callback`

The local preview remains on port 5193. These settings document the deployment target; they do not configure DNS, TLS, GitHub OAuth credentials or deploy the service.
