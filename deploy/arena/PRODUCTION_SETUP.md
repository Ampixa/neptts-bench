# Production setup: what Ampixa needs to do

Code is ready. These steps need **Ampixa's accounts, decisions or machines**, so engineering can't do them.
Each step names an owner, the time it takes, and how to check it worked.

**Check progress at any time:**

```bash
sudo -u ampixa-arena bash -c 'set -a; . /etc/ampixa-arena/environment; set +a; \
  cd /opt/ampixa-arena/current && .venv/bin/python -m submission_app.doctor --data /var/lib/ampixa-arena'
```

Reviewers see the same checklist at the top of **Reviewer access** (`/api/admin/readiness`). Every unfinished item shows the fix.

| # | Step | Owner | Time | Unlocks |
|---|------|-------|------|---------|
| 1 | GitHub environment and deploy key | GitHub org admin | 15 min | Auto-deploy on merge |
| 2 | GitHub OAuth App | GitHub org admin | 10 min | Public submissions |
| 3 | Approve the Rani contract | ML lead | 30 min | Primary judge |
| 4 | Upgrade the VM and run the Rani worker | Ops | 1 h | Scoring, auto-review |
| 5 | Seed baselines and a human reference | ML + ops | ½ day | Arena opens |
| 6 | Backups | Ops | 30 min | Safe to accept uploads |
| 7 | Policy decisions | Founders / legal | – | Shadow → on, open data |
| 8 | Voice donation: consent and storage | Legal + HF org admin | 1–2 weeks | Donations, paid programme |
| 9 | Benchmark v2 and private set | Linguists | 2–4 weeks | Credible, un-gameable ranking |
| 10 | Brand assets and fonts | Design | 1 day | Link previews, Devanagari typography |

---

## 1. GitHub environment and deploy key

Already described in [README.md](README.md). Summary:

1. Repo → Settings → Environments → **New environment** `arena-production`. Add required reviewers if wanted.
2. Run `ssh-keygen -t ed25519 -f arena-deploy -N ''` on a trusted machine.
3. Put `arena-deploy.pub` in `/var/lib/ampixa-arena/.ssh/authorized_keys` on the VM, prefixed with
   `restrict,command="/usr/local/bin/ampixa-arena-receive" `.
4. Add environment secrets: `ARENA_HOST`, `ARENA_SSH_KEY` (the private key), and `ARENA_KNOWN_HOSTS`. Copy the known-hosts value from `ssh-keyscan` run **on the VM itself**.
5. The pusher's token needs the **`workflow`** scope, because PR #8 changes `.github/workflows/arena.yml`.

✓ Check: merge to `main`, and the *deploy* job goes green. `curl https://ampixa.benchmarks.cloud/api/health` should then show the new commit.

## 2. GitHub OAuth App (for submitters)

Go to GitHub → **Ampixa org** → Settings → Developer settings → OAuth Apps → New:

- Homepage: `https://ampixa.benchmarks.cloud`
- Callback: `https://ampixa.benchmarks.cloud/api/auth/github/callback`

Put the client ID and a new client secret into `/etc/ampixa-arena/environment` (see [environment.example](environment.example)). Then run `sudo systemctl restart ampixa-arena`.

✓ Check: the doctor shows `GitHub sign-in for submitters ✓`. "Continue with GitHub" works on `/submit`.

## 3. Approve the Rani contract

`neptts-eval/src/neptts_eval/rani_contract.json` pins the checkpoint sha256 (`b015e46e…`), NeMo 3.0.0 and torch 2.15.0+cpu.
The ML lead confirms these are the intended public Rani weights, then approves PR #8.

Attribution "Rani (Ampixa Labs)" is required by Rani's licence, and the site already shows it.

## 4. Upgrade the VM and run the Rani worker

The queue lives in SQLite, so the worker must run **on the same host** as the web service. Never put the database on a network filesystem. NeMo needs about 3–4 GiB of RAM, but the current shared VM has 2 GiB. **Resize to ≥ 8 GiB** (4 vCPU is recommended), or move Arena to its own VM.

```bash
sudo apt install -y ffmpeg python3.12-venv
sudo install -d -o ampixa-arena -g ampixa-arena /opt/ampixa-arena/worker-venv /var/lib/ampixa-arena-models
sudo -u ampixa-arena python3.12 -m venv /opt/ampixa-arena/worker-venv
sudo -u ampixa-arena /opt/ampixa-arena/worker-venv/bin/pip install \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  -r /opt/ampixa-arena/current/submission_app/requirements-worker-rani.txt
# Gated weights: request access to ampixa/rani-asr for the service account first.
sudo -u ampixa-arena env HF_TOKEN=... /opt/ampixa-arena/worker-venv/bin/hf download ampixa/rani-asr \
  --local-dir /var/lib/ampixa-arena-models/rani-asr
sudo install -m 644 ampixa-arena-worker.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ampixa-arena-worker
```

Set `NEPTTS_AUTO_REVIEW=shadow` (see step 7). The worker refuses to start if the checkpoint hash or package versions differ from the contract.

✓ Check: the doctor shows `Rani evaluation worker running ✓ rani seen N s ago`. `journalctl -u ampixa-arena-worker` should show no errors.

Note: the deploy receiver restarts only the web service. After a deploy that changes `submission_app/worker.py` or `neptts-eval`, run `sudo systemctl restart ampixa-arena-worker`. Alternatively, add it to the receiver's sudoers line.

## 5. Seed baselines and a human reference

The arena only opens once two consented systems exist. Seed the systems from the 2026 study, which Ampixa generates itself. That way their claims aren't self-reported. Also add **one human reading of all 205 prompts**, so a rating means "this far from a real person".

```bash
# 1. Generate (on any machine with the API keys). Each script writes one WAV per prompt.
python scripts/generate_tts.py ...            # see each script's --help
# 2. Convert to what the contract accepts: mono PCM16 WAV, 16–48 kHz, 0.2–30 s.
for f in out/*.{mp3,wav}; do ffmpeg -loglevel error -y -i "$f" -ac 1 -c:a pcm_s16le "wav/$(basename "${f%.*}").wav"; done
# 3. On the VM, add each system (copy and edit seeds/example_baseline.json):
sudo -u ampixa-arena /opt/ampixa-arena/current/.venv/bin/python -m submission_app.seed \
  --data /var/lib/ampixa-arena --wav-dir /tmp/wav/tingting_asmita \
  --manifest seeds/tingting_asmita.json --kind baseline --note "generated 2026-11-20"
```

Suggested first set: TingTing (Asmita, Subina, Sambriddhi), Edge TTS (Sagar, Hemkala), Piper, ElevenLabs v3, Gemini, gTTS, and **Human reference** (`--kind human_reference`, `seeds/example_human_reference.json`).

The human reader must sign the voice-donation consent, or a separate studio release, that allows public playback.

Check the commercial APIs' terms before publishing their audio. Most allow publishing benchmark samples, but someone should confirm per vendor.

✓ Check: the doctor shows `Systems in the blind arena ✓ (≥4, with a human reference)`. `/arena` shows the listening check.

## 6. Backups

```bash
sudo apt install -y age rclone
age-keygen -o arena-backup.key   # on a trusted laptop; keep the .key in the password manager, NOT on the VM
sudo tee /etc/ampixa-arena/backup.env >/dev/null <<'EOF'
BACKUP_RECIPIENT=age1...   # public key printed by age-keygen
BACKUP_UPLOAD="rclone copy {} ampixa-offsite:ampixa-arena-backups/"
EOF
sudo chmod 600 /etc/ampixa-arena/backup.env
sudo rclone config   # create remote "ampixa-offsite" (B2 / S3 / Drive), as root
sudo install -m 755 backup.sh /usr/local/bin/ampixa-arena-backup
sudo install -m 644 ampixa-arena-backup.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ampixa-arena-backup.timer
sudo systemctl start ampixa-arena-backup   # first run now
```

✓ Check: the doctor shows `Recent off-host backup ✓`. **Do one restore test** into a scratch directory: run `age -d`, then `tar -x`, and open `submissions.db`.

## 7. Policy decisions (no engineering needed; set in the environment file)

| Decision | Setting | Recommendation |
|---|---|---|
| Automatic publishing | `NEPTTS_AUTO_REVIEW` = `shadow` → `on` | Start with `shadow`. Switch to `on` after ≥10 shadow verdicts, if the doctor's agreement figure is acceptable (it counts held-but-approved as disagreement) |
| Publish vote data | `NEPTTS_ARENA_PUBLIC_EXPORT=on` | Yes, once the privacy notice mentions it. The export is pseudonymised (per-file listener numbers, UTC day only) |
| Vote caps | `arena.py` constants (50/day, 200 total, 300/IP/day) | Keep for launch |
| Who may review | Share the reviewer password | Two named reviewers minimum |
| Public sample clips | On for every consented system | Confirm that the submitter consent wording ("play audio to listeners") covers the leaderboard's play buttons; if not, add a sentence to the template |

## 8. Voice donation: consent and storage

1. **Legal and native-speaker review** of `submission_app/consent/voice_donation_v1.json`. The review must:
   - name Hugging Face as the storage provider
   - state the retention period and the withdrawal process
   - state the payout terms (NPR per accepted minute, payment method and tax)
   - name who can access the data

   Then set `"status": "approved"`. Changing the text later needs a new version (`v2`); the app records which version each donor accepted.
2. **Hugging Face:** an org admin runs:
   ```bash
   HF_TOKEN=<admin token> python scripts/create_hf_donation_repos.py --org Ampixa
   ```
   This creates two **private** datasets, then prints the environment lines.
3. Create a **fine-grained** token with write access on those two repos and read on `ampixa/rani-asr`. Don't use the admin token here. Put it in the environment file as `HF_TOKEN`.
4. Set `NEPTTS_DONOR_RATE_NPR_PER_MIN` if the paid programme opens.
5. Do one real take end to end, **including delete**. Confirm that the file has gone from the repo's history on huggingface.co.

✓ Check: run `doctor --online`. It should show `Donation dataset is private ✓`.

## 9. Benchmark v2 and a private held-out set

- [`benchmark/proposals/prompts_v2_draft.json`](../../benchmark/proposals/prompts_v2_draft.json) has 28 draft prompts. It covers numbers, dates and currency, code-switching, named entities, conversation, IVR and long-form. Native speakers from several regions need to review and extend it (target is about 100).
- **Private set:** write another 100–200 prompts that are **never committed** to GitHub, and keep them in a private HF dataset. Rankings should use the private set and show public-set scores alongside, like VoiceArena. Systems can't tune to prompts they have never seen. Rotate part of the private set yearly.
- Engineering then adds it as a new benchmark version. Old scores stay attached to v1, and the two are never mixed.

## 10. Brand assets and fonts

- Add a 1200×630 PNG at `submission_app/static/og.png`. Then add `<meta property="og:image" content="https://ampixa.benchmarks.cloud/static/og.png">` to `index.html`. Link previews on Facebook, Viber and X need a raster image.
- Bundle **Noto Sans Devanagari** (OFL) in `submission_app/static/fonts/` and add an `@font-face`. The CSS already prefers it for every `lang="ne"` element, but without the file, phones fall back to their system font.
- Point the ampixa.com navigation at `https://ampixa.benchmarks.cloud/arena`. Paths such as `/arena`, `/leaderboard` and `/methodology` are now shareable links.
