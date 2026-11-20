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

For evaluation, set up the primary Rani worker (see [Rani judge](#rani-judge-primary-profile) for the one-time pinning):

```sh
pip install -r submission_app/requirements-worker-rani.txt
python -m submission_app.worker --data .local/submissions --checkpoint ~/models/rani-asr/rani/rani.nemo
```

The worker serializes jobs and holds a per-engine process lock. Use `--once` to stop when idle. Interrupted jobs are requeued on worker restart, once; after a second failure a reviewer must retry. Whisper remains available as a fallback or second opinion (`pip install -r submission_app/requirements-worker.txt`, then `--engine whisper --model small`); its weights are downloaded by the trusted Whisper package and checksum-verified by its loader, with `NEPTTS_MODEL_CACHE` for the cache. Logs stay on the operator's machine; raw exceptions are not shown to submitters.

`--model tiny` is a development-only profile. It runs genuine ASR but **cannot be approved for the public leaderboard**. Automated tests inject a deterministic scorer to exercise publication without downloading models or claiming real quality results.

## Submission contract

The service selects the 205 `sent_NNN` synthesis records in the checked-in `benchmark/sentences.json`, excluding 160 natural-speech references. IDs are not consecutive. The complete sentence content is hashed into the benchmark ID; old or mismatched manifests fail validation. This is a proposed new submission split, not a claim to reproduce the historical study subset. Freeze/review this split before production launch.

Download `/api/benchmark/pack` for the exact prompts, manifest template and instructions. A submission ZIP contains only `submission.json` and `audio/<sentence-id>.wav` for every required prompt. Audio must be mono PCM16 WAV, 16–48 kHz, 0.2–30 seconds. No missing IDs, extra files, duplicate ZIP paths, links, encrypted entries or silence. Metadata declares model version, voice, generation settings, post-processing, selection/retries, training overlap and permission to evaluate/publish. The manifest additionally requires:

- `generation_mode`: `single_take` or `best_of_n`.
- `candidate_count`: integer N, the maximum total generation attempts per prompt, including retries and discarded failures (1 for single-take; at least 2 for best-of-N). This is not merely the selection-pool size.
- `selector_model`: `none` for single-take; name/version of the selector, or `manual` for human selection.
- `judge_family_used`: explicit boolean disclosing judge-family models used for selection, filtering or reinforcement learning.
- `judge_family_details`: which model families and uses, or an explicit statement that none were used.

Single-take means exactly one generation attempt per prompt, including attempts that fail or are discarded. Regenerating a failed or unsatisfactory prompt makes the run ineligible for single-take, even if only one waveform per prompt is submitted. Best-of-N declares the maximum number of attempts per prompt as N, counting failed/discarded candidates and retries; `selection_policy` must explain selection, retries and any per-prompt variation in counts. Disclose the selection-pool size separately from total generation attempts: for example, eight candidates reaching a selector after twelve attempts means N=12, not N=8. The UI labels this as “Best-of-N · max 12 attempts/prompt.” Validation checks declaration structure and consistency; it cannot verify how many attempts actually ran. Re-downloading the same generated file without another inference attempt does not add a candidate. These declarations are submitter-reported; reviewer approval is not execution attestation. The leaderboard defaults to single-take declarations. Legacy submissions without the structured fields remain `unknown`, even if their old free text says “single take”; users can select them explicitly. Do not upload model weights, keys or personal reference recordings.

To validate/package locally without evaluation models:

```sh
python -m submission_app.pack --audio ./outputs --manifest submission.json --output submission.zip
```

The helper refuses to overwrite an existing archive. Uploads are capped at 150 MiB compressed, 300 MiB expanded and 8 MiB per audio file. Extraction uses server-generated paths rather than archive paths. Only two uploads are processed concurrently; there is a 2 GiB free-space floor, a 100-submission capacity cap and a per-client hourly limit. These are initial operational limits, not a full internet-scale abuse service.

## Review and publication

1. GitHub sign-in is required to upload. Submissions belong to the immutable GitHub user ID and appear under My submissions across browsers. Successful validation returns a random private receipt token. The browser stores it locally and offers a private tracking link; retain it for optional bearer-link access. Earlier anonymous receipts remain valid, but are not automatically claimed by an account.
2. A validated upload is queued for evaluation automatically.
3. The worker verifies stored audio hashes, evaluates the full set with the primary judge (Rani) and requires finite metrics for every prompt. A failed or interrupted run is retried once automatically; after that it waits for a reviewer. Any failure leaves the entry unpublished.
4. Automatic review (`autoreview.py`, policy `auto-review-v1`) publishes the result if every check passes. Otherwise it holds the submission in `awaiting_review` with its reasons, shown to the reviewer and the submitter. The policy never rejects.
5. A reviewer handles only the held submissions: approve or reject with a note. Reviewers can also **take down** anything already published (reject on a published entry). Only published results appear on the public API/leaderboard.

Automatic review holds a submission when:

| Reason | Check |
|---|---|
| `no_primary_judge` / `not_primary_judge` | Rani is not pinned, or the report came from another judge (for example an offline D32 import) |
| `development_judge`, `invalid_report` | Development profile, or the report fails the consistency checks used for manual approval |
| `selection_unknown` | Single-take / best-of-N declaration incomplete |
| `judge_family` | Declares a judge-family model (Rani) used for selection, filtering or RL |
| `placeholder` | Name, version, organization, voice, settings or overlap still contain the template text |
| `duplicate_name` | Another non-rejected submission has the same system name and version |
| `shared_audio` | More than 5% of WAVs are byte-identical to another submission's |
| `failed_prompts` | More than 20% of prompts score CER ≥ 0.9: wrong files, order or language |
| `implausibly_good` | CER below `PLAUSIBLE_FLOOR`, the real-speech floor. Off until Rani's real-speech calibration is done |

Set `NEPTTS_AUTO_REVIEW=off` on the web server **and** the worker to go back to fully manual review. Results from an
additional judge (for example Whisper as a second opinion) are published automatically once they pass the report checks.

The automated track originally used Whisper small Nepali ASR round-trip macro CER/WER with a fixed normalization implementation. Its profile records model weight SHA, evaluator/normalizer/scorer code identity, package versions and inference settings. Each profile is ranked separately. CER/WER are intelligibility proxies, not human quality scores. NepaliMOS/SCOREQ and human MOS remain absent rather than using incompatible checkpoints or invented ratings. Later metrics must have explicit versions, full coverage and a new comparison profile.

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

The cross-revision offline regression creates a synthetic submission with the actual
server at pre-change main commit `210dd973ca5ea80149fcccbe68c73ca8f203c624`, then exports,
evaluates (with a deterministic scorer) and imports it using the current code. Run the
tests from a Git checkout containing that commit; shallow clones need history fetched.
CI checks out full history for this test. No production data or model downloads are used.

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

Portal name: **Ampixa Benchmarks Arena**. Intended production origin: `https://ampixa.benchmarks.cloud`.

- `NEPTTS_ORIGIN=https://ampixa.benchmarks.cloud`
- GitHub OAuth App homepage: `https://ampixa.benchmarks.cloud`
- GitHub OAuth callback: `https://ampixa.benchmarks.cloud/api/auth/github/callback`

The local preview remains on port 5193. These settings document the deployment target; they do not configure DNS, TLS, GitHub OAuth credentials or deploy the service.


### Reviewer-local evaluation

In the reviewer screen, choose **Download evaluation package**. It includes the original
manifest, all WAVs, and an evaluation-job.json binding the submission ID, benchmark and
audio digest. It is available only to authenticated administrators.

On an evaluation machine, check out the same repository revision as the server, install
FFmpeg and `submission_app/requirements-worker.txt`, then run:

```sh
python -m submission_app.offline --package evaluation-SUBMISSION_ID.zip --output results.json
```

This runs the existing Whisper small CPU evaluator locally. No production worker or
GitHub Actions job is required. Upload results.json with **Upload evaluation results**.
The server requires all prompt scores, finite nonnegative CER/WER, consistent macro
averages, matching audio/submission/benchmark identity and a self-consistent publishable
profile. Import is admin-only and bounded to 2 MiB. Published, rejected, actively evaluated
and already-reviewed jobs cannot be overwritten. Approval remains a separate action.

These are consistency checks, not cryptographic proof of honest inference. An authenticated
reviewer is trusted to run the pinned evaluator and inspect provenance before approval.
Audio hashes bind results to a package but do not authenticate how scores were produced.
Different evaluator profiles remain separate on the leaderboard. No human MOS is inferred.

Reviewer packages exported from existing submissions may lack any or all of the five
structured generation fields. Offline validation accepts those historical declarations
without filling in defaults; incomplete declarations display as **Unknown / legacy**
and remain excluded from the default single-take leaderboard. New uploads and the local
submission packager still require all five fields. Audio, benchmark, consent and report
integrity checks also apply to legacy packages.


## Judge calibration and comparison

`GET /api/judge-profiles` returns the checked-in aggregate catalog in `benchmark/judge_calibration_v1/calibration_public.json`. The UI gives each exact profile its own tab and ranking, showing PV-read / Fresh real-speech CER, confidence intervals, loanword policy and qualification limits. Missing calibration remains explicitly pending; no number is inferred from a related recognizer. The historical Whisper-small profile is labelled **uncalibrated for spontaneous Nepali**, with measured calibration when available.

Calibration metadata is separate from the evaluator identity. The bytes of `evaluation.py`, whose hash participates in historical profile IDs, remain unchanged. The serial worker continues to use the pinned Whisper-small contract. LONG1C and MMS catalog entries describe calibration evidence. The private DISTILL-32M v2 profile can be run offline on an authorized reviewer machine; its call-trained weights remain private and reproducibility outside Ampixa depends on the owner's weight-release decision.

For that private profile, use `python -m submission_app.offline --profile d32v2-s60000-full-translit` with `--package`, `--output`, `--checkpoint`, `--bpe`, `--recipe`, `--icefall`, and `--normalizer-root`. Every source/model pin is checked before decoding. The evaluator uses CPU, one thread, full context and the frozen legacy score2 transliteration policy. Its output contains numeric scores and edit counts, with no audio, hypotheses, reference transcripts or private input paths. It must cover all benchmark prompts.

Additional D32 scores on an already published submission use a separate review path after deployment:

1. POST the bound result JSON to `/api/admin/submissions/{sid}/profiles/{profile_id}/results` with the reviewer session and `X-Neptts-Request: submission` header.
2. Inspect the pending report with GET `/api/admin/submissions/{sid}/profiles`.
3. POST `{"action":"approve","note":"<review evidence>"}` (or `reject`) to `/api/admin/submissions/{sid}/profiles/{profile_id}/decision`.

Import does not publish. Only an approved additional report appears at `/api/leaderboard?selection=all&profile={profile_id}` and `/api/leaderboard/{sid}/report?profile={profile_id}`. Existing `submissions.report`, `submissions.profile`, publication state, timestamps and the default leaderboard remain unchanged; additional reports have their own immutable identity and review state. Reimporting or deciding an already decided profile returns HTTP 409. Reviewer authentication, benchmark/audio binding, profile/source pins, full coverage, edit-count consistency and aggregate checks apply. These checks establish consistency, not proof of honest execution. See the job's `IMPORT.md` for deployment gates and the exact reviewed commands.

`GET /api/leaderboard` defaults to `selection=single_take`. Explicit alternatives are `best_of_n`, `unknown` and `all`; `profile=<exact-id>` restricts results to one evaluator identity. The API returns `total_approved` before filtering. Scores returned across profiles have no common rank; the UI sorts only within the selected exact ID. Human MOS remains a separate historical study.


## Historical study snapshot

`study_snapshot.json` is a compact source-derived aggregate artifact used by `/api/historical-study` and the UI. Its self-documenting provenance records creator, source hash/revision/date, purpose, owner, review/retention status, reproduction, known duplicate aggregate locations, HF exclusion and usage. It contains the 12 rows of the checked-in paper's human-MOS table (10 TTS systems and 2 human baselines), copied reported 95% rater-bootstrap intervals and counts from `neptts-eval/documentation_scopes_v1.json`; no private rating rows were accessed or re-estimated. The UI dates and links this historical table separately from current ASR submissions. Top-four intervals overlap, so the display does not establish a unique winner.

Snapshot bytes: 3759; SHA256: `e0028fdca4aa06e73bec5ee04fc1146223fd9b0f9c457b14ce59d734cd714423`. Retain as the current historical display input until superseded by a reviewed study/filter snapshot; last reviewed 2026-11-03, next review 2026-11-11. Source/Git history provides code and documentation revision provenance; this compact JSON is NOT an audio/model backup.

## Rani judge (primary profile)

[Rani](https://github.com/Ampixa/rani-asr) (Ampixa Labs) is a NeMo TDT/CTC Conformer for conversational Nepali. It is the
**primary judge**: it scores the submission queue, and its profile is the default leaderboard tab (`primary` in
`/api/judge-profiles`). Submissions published earlier under Whisper are backfilled into `profile_evaluations` by the same
worker. Whisper-small real-speech CER was 49% (PV) and 115% (Fresh) in phase-1b calibration, which is why it was
replaced. Licence: evaluating with Rani requires the credit
"Rani (Ampixa Labs)" with a link, which the profile and catalog entry carry.

Scoring matches Rani's official NepTel 0.2 scorer (`neptts_eval/rani_text.py`, verified identical on all 368 benchmark
texts): digits become spoken Nepali words, English in either script maps to one Devanagari form, CER excludes spaces.

One-time setup on the evaluation machine (separate environment from the Whisper worker):

```sh
pip install -r submission_app/requirements-worker-rani.txt
hf download ampixa/rani-asr --local-dir ~/models/rani-asr          # gated: request access first
python -m neptts_eval.rani pin --checkpoint ~/models/rani-asr/rani/rani.nemo
git add neptts-eval/src/neptts_eval/rani_contract.json              # review and commit the pins
```

Until the contract is pinned, the Rani profile, catalog entry and import all fail closed. Re-pinning (`--force`)
creates a new profile ID, so old and new Rani scores are never mixed.

Run the primary worker. Each loop it checks donated takes first, then the submission queue, then backfills published
submissions that lack a Rani score:

```sh
python -m submission_app.worker --data .local/submissions --checkpoint ~/models/rani-asr/rani/rani.nemo
```

Optional Whisper second opinion, in the Whisper environment, scoring every published submission:

```sh
python -m submission_app.worker --data .local/submissions --engine whisper --model small --role additional
```

`--once` stops when there is no work left. A failed backfill is logged, recorded as an event and skipped until the
worker restarts. Offline reviewer packages work too: `python -m submission_app.offline --profile rani --checkpoint ... --package ... --output ...`.

Validation (2026-11-15, pinned checkpoint `b015e46e…`, NeMo 3.0.0, torch 2.15.0+cpu, 2 threads on x86 under WSL): on
NepTel 0.2 the pinned judge scores **33.81** WER with Rani's official `score_v02.py`, against **33.51** for Rani's
published outputs. Transcripts are not byte-identical (226 word edits in 2,520, mostly word boundaries such as
"एम्पिक्सा" / "एम पिक्सा"), as expected from greedy decoding of 8 kHz phone audio on a different CPU and torch build;
re-runs on the same machine are identical. One 205-prompt submission takes about 3 minutes on CPU. Scores are only
comparable within one pinned profile, which records exactly this runtime.

Limitations: Rani is an intelligibility proxy with real-speech calibration pending. Because it is trained for messy
conversational speech, it may forgive mispronounced minimal pairs (149 of the 205 prompts). Validate it against the
human pairwise ranking before treating it as more than a content check.

## Blind listening arena (human preference)

`#vote` lets native listeners compare two anonymous systems on the same prompt. The question is "Which sounds more
natural, as though spoken by a real person?", answered A, B, both natural, or neither natural, with optional
per-side issue tags (including Nepali-specific ones: schwa deletion, numbers, English words). Names appear after the
vote. `#human` shows the ranking.

- **Vote caps.** A listener gets at most 50 comparisons per 24 hours and 200 in total; one network address at most
  300 per 24 hours, so fresh browser sessions from one machine do not reset the limit. In the ranking each listener
  counts for at most 50 votes: someone with more has every vote scaled down (`bt.cap_listener_weight`), so 500 votes
  move the ratings no more than 50 would. The leaderboard reports how many listeners were capped.
- **Who is compared:** approved submissions whose owners allow it, via `permission_to_play_audio_to_listeners` in
  the manifest or the toggle on their submission page (owner account or receipt holder only; reviewers cannot decide
  for them). Withdrawing stops new comparisons and deletes the playback cache; votes already cast are kept.
- **Blinding:** clips are re-encoded to mono 24 kHz PCM16 at -23 dBFS active-speech RMS (`arena_audio.py`) and served
  under one-time battle tokens without ETag/Last-Modified, so format, level, URL and cache headers reveal nothing.
- **Listener quality:** a listening check first (hear a clip, pick its sentence from four, including its minimal-pair
  partner; three correct, three attempts). Votes count only if both clips were fetched and the longer one had time to
  play. About one battle in eight after warm-up is a hidden check (identical clips, or clean against degraded);
  listeners failing most checks are excluded. A listener votes on any one system at most 60 times.
- **Ranking (`bt.py`):** Bradley-Terry fitted by the minorise-maximise method, ties as half wins, one virtual tie per
  system against a fixed anchor, so ratings stay finite. Elo scale centred on 1000. 95% intervals and rank ranges come
  from 200 bootstrap resamples of whole listeners. Systems under 30 counted votes are marked provisional.
- **Pair choice:** pairs with fewer counted votes are preferred (weight `1/(1+n)^2`); prompts not yet heard by the
  listener come first.
- **Privacy:** listeners are anonymous (random ID in an HttpOnly cookie). Stored: answers, tags, timings.

Reviewer statistics (listeners, screened, battle kinds, excluded votes by reason) are at `GET /api/admin/arena`.

Not yet: a real-speech anchor (no human recordings of the 205 prompts exist), an export of vote rows for research,
and a paid listener panel kept separate from volunteers.

## Voice donation and paid voice programme

`#donate` → `#record` → `#my-voice`; reviewers use `#donations-admin`.

- **Closed until consent is approved.** `submission_app/consent/voice_donation_v1.json` holds the bilingual consent
  text and per-purpose checkboxes. It ships as `"status": "draft"`. After legal and native-speaker review, set it to
  `"approved"`. Donors record the version and SHA-256 they agreed to, and any edit to the file forces them to re-read it.
  Changing approved wording means a new versioned file, not an in-place edit.
- **Anonymous donors.** No account. Enrollment collects age band, gender, province, first language and device (each
  with a prefer-not option), sets an HttpOnly session cookie and shows a one-time recovery code. Recovering on another
  device ends the old session.
- **Prompt sets.** `anchor-v1` is the 205 benchmark prompts. These recordings are evaluation-only (human reference
  speech and judge calibration) and must never be released for training. Open sets come from
  `benchmark/donation_prompts.json` (`{"sets":[{"id":"open-…","source":…,"license":…,"prompts":[{"id","text"}]}]}`), are
  validated at start-up and may not repeat benchmark prompts. None are shipped: add a licensed corpus. Training and
  licensing purposes apply to open sets only.
- **Recording.** An AudioWorklet captures raw PCM with echo cancellation, noise suppression and auto-gain off, and
  the page encodes 16-bit mono WAV, so there is no lossy codec and no FFmpeg on the server. A mic check comes first.
- **Instant signal checks** (`donation_audio.py`) reject before storing: wrong format, under 1 s or over 20 s,
  silent, quieter than -40 dBFS, clipped, or under 15 dB above the noise floor (the quietest 100 ms).
- **ASR check.** The Rani worker also processes donated takes, before submissions. Prompt CER at most 0.15 is
  accepted, at most 0.40 goes to a reviewer, anything higher is rejected. Reviewers accept or reject borderline takes with a note
  the donor can see.
- **Donor control.** Donors can play and delete any take, or withdraw. Withdrawing deletes all audio and the profile.
  The payout ledger stays as a financial record.
- **Paid programme.** Set `NEPTTS_DONOR_RATE_NPR_PER_MIN` to open it. Donors apply with an eSewa, Khalti or bank handle
  (visible to reviewers only). Reviewers approve at a per-minute rate. Earnings are accepted minutes times rate. Payouts are
  recorded manually with a transaction reference and cannot exceed the balance. No money moves automatically.

### Where donated audio is stored

Donated audio lives in **private Hugging Face datasets** owned by Ampixa (`donation_store.py`). The server keeps only
metadata in SQLite and a short-lived spool. Without configuration (local development, tests) takes stay under
`<NEPTTS_DATA>/donations/`.

| Variable | Meaning |
|---|---|
| `NEPTTS_HF_DONATIONS_BENCHMARK_REPO` | Private dataset for `anchor-v1` takes, e.g. `ampixa/neptts-donations-benchmark`. Setting it switches storage to Hugging Face |
| `NEPTTS_HF_DONATIONS_OPEN_REPO` | Private dataset for open prompt sets. Required once open sets exist; must differ from the benchmark repo |
| `HF_TOKEN` | Fine-grained token with write access to these two repos only. The Rani worker needs read access |
| `NEPTTS_HF_SYNC_SECONDS` | Sync interval inside the web server, default 300. `0` disables the in-server loop |

1. An upload passes the signal checks and is written to `<NEPTTS_DATA>/donation_spool/` (`stored='spool'`).
2. Every sync interval the server checks the repo is still **private** (it refuses to write otherwise) and commits up to
   200 spooled takes plus `metadata.jsonl` in one commit, then deletes the spooled files (`stored='hub'`). Existing
   local takes are uploaded the same way when Hugging Face is first configured.
3. Playback and the Rani worker read from the spool while a take is there, otherwise from Hugging Face (the worker
   uses a temporary copy that is deleted after scoring). Browsers never see the repo.
4. Deleting or withdrawing queues a delete commit. After it lands, the audio object is permanently erased from the
   repo's storage and history (`permanently_delete_lfs_files`, matched by the take's SHA-256), so it does not survive
   in old revisions. Objects not stored as LFS fall back to a history squash.
5. If Hugging Face is down, takes wait in the spool and the error shows on the donation review page. Uploads are
   refused with 503 once 2 GiB is waiting or the disk has less than 2 GiB free.

Repo layout: `audio/<donor id>/<recording id>.wav` plus `metadata.jsonl` in Hugging Face's audio-folder format
(`file_name`, recording and donor ids, prompt and text, duration, sample rate, SNR, Rani CER, review state, month).
Demographics, device, payout details and recovery hashes are never uploaded. Before going live, check against the
real Hub that permanent deletion covers these repos' storage backend, and add Hugging Face as storage provider to the
consent text.

Not yet: wiring accepted `anchor-v1` takes into the arena as a "Human reference" system (needs the
`listener_playback` purpose), encryption at rest for payout handles, and an export for consented open-set data.

## Operating in production

- **Readiness:** `python -m submission_app.doctor --data <dir>` (also `GET /api/admin/readiness`, shown on the reviewer page) lists every launch step still missing, with the fix. See [deploy/arena/PRODUCTION_SETUP.md](../deploy/arena/PRODUCTION_SETUP.md).
- **Shadow review:** `NEPTTS_AUTO_REVIEW=shadow` queues and scores automatically and records what `auto-review-v1` would decide, but a reviewer approves everything. The readiness report shows how often reviewers agreed.
- **First-party systems:** `python -m submission_app.seed --kind baseline|human_reference --wav-dir … --manifest …` adds Ampixa-generated baselines and human readings through the normal checks and scoring. The arena labels them; a human reference anchors the Elo scale.
- **Worker heartbeat:** each worker loop records itself in `worker_heartbeats`; a primary worker silent for 15 minutes shows as down.
- **Health:** `GET /api/health` returns `{ok, commit}` for uptime monitors.
- **Arena extras:** `GET /api/arena/samples/<id>` plays one fixed prompt that is never used in battles or screening (so sample bytes can't unblind a battle). `GET /api/arena/votes.csv` exports pseudonymised counted votes (reviewers only unless `NEPTTS_ARENA_PUBLIC_EXPORT=on`). The board recomputes at most every `RATINGS_REFRESH` seconds and adds point-estimate ratings per prompt category once it has `MIN_CATEGORY_VOTES` votes. Listeners may declare a region (optional).
- **Shareable paths:** `/arena`, `/leaderboard`, `/methodology`, `/submit`, `/donate` … serve the app and open that view.
