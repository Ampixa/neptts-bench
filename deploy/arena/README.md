# Shared VM deployment (no Docker)

Arena runs independently of AFFiNE and mail: system user `ampixa-arena`, loopback port 5193, unit `ampixa-arena.service`, releases in `/opt/ampixa-arena/releases`, persistent SQLite/audio in `/var/lib/ampixa-arena`, root-only credentials in `/etc/ampixa-arena/environment`.

The web service is capped at 512 MiB and 50% of one CPU. **No Whisper worker is installed or started on the shared 2 GiB VM.** Validated submissions remain pending until an operator queues them and provisions an appropriately sized worker with access to the data. Do not claim automatic scoring is operational before that step. The small model may contend with AFFiNE and mail; choose a separate evaluation machine or upgrade capacity first.

## First installation

Prerequisites: Ubuntu 24.04, Python 3.12 with venv, Nginx, Certbot and SSH. Run `bootstrap.sh` as root from this directory. It creates only Arena resources and generates a random reviewer password; it does not restart other services. Install a dedicated deployment public key in the Arena user's `authorized_keys`, prefixed with:

```
restrict,command="/usr/local/bin/ampixa-arena-receive" ssh-ed25519 ...
```

The key can invoke only `deploy <40-character commit SHA>`. The receiver bounds and validates a source archive, builds a web-only venv as the unprivileged user, switches the release and restarts only Arena. A failed health check restores the prior application if one exists. Five recent releases are kept. Bootstrap/receiver/unit changes require an operator install; the ordinary application deployment cannot overwrite root-owned deployment controls.

Set up a GitHub environment named `arena-production`. Store `ARENA_HOST`, `ARENA_SSH_KEY` (dedicated private key), and `ARENA_KNOWN_HOSTS` (verified host key; never use runtime ssh-keyscan as trust) as environment secrets. No VM IP or credentials belong in the workflow.

PRs run the lightweight backend tests, lint and JS syntax checks. Merges to `main` affecting Arena run those checks and deploy only if they pass. Manual workflow dispatch deploys **main only**. Deployment is serialized and not cancelled mid-switch. No Docker, GPU model downloads or inference tests run in CI. GitHub Actions begins only after the workflow exists in the remote branch; push-triggered production deployment requires merge to main.

## DNS and HTTPS

Canonical domain: `ampixa.benchmarks.cloud`. Point its A record at the VM. Install `nginx.conf` as a separate site, validate with `nginx -t`, then reload Nginx (do not restart unrelated units). Before a certificate exists, serve only the ACME challenge location and return 503 elsewhere; do not expose logins over HTTP.

After DNS resolves, run `sudo bash enable-https.sh <VM-public-IPv4>`. It verifies DNS, obtains the certificate using the existing account, validates the dedicated TLS site, reloads Nginx and installs a renewal hook.

Use the existing Certbot account to issue a webroot certificate using `/var/www/ampixa-arena-acme`. Once issued, configure the HTTPS server with that certificate and the proxy settings from the template, and redirect HTTP to HTTPS. Install a successful-renewal hook that validates and reloads Nginx. Leave other hosts, ports and certificates unchanged.

## Authentication and launch

Set `NEPTTS_ORIGIN=https://ampixa.benchmarks.cloud`, `NEPTTS_GITHUB_CLIENT_ID`, and `NEPTTS_GITHUB_CLIENT_SECRET` in the protected environment file. OAuth callback is `https://ampixa.benchmarks.cloud/api/auth/github/callback`. Restart only Arena. Until OAuth is configured, the public leaderboard works but uploads are closed. GitHub sign-in does not confer reviewer rights. Retrieve the generated admin password directly on the VM, never in CI logs.

Before accepting public uploads, decide retention and configure regular off-host backups. Stop only Arena and its evaluator for a consistent backup of `/var/lib/ampixa-arena` plus the protected environment file. Keep backups encrypted/private. Application rollback does not roll back SQLite migrations; current ownership migration is additive. Future schema changes require a backup/compatibility plan.

The ampixa.com navigation entry should be an ordinary external link to `https://ampixa.benchmarks.cloud` labelled **Ampixa Benchmarks Arena**. No rewrite or redirect of the entire ampixa.com site is intended.
