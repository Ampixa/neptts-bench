#!/usr/bin/env bash
set -euo pipefail
# Run as root from this directory. Requires an already-installed Python venv module.
[[ $(id -u) == 0 ]]
python3 -c 'import ensurepip'
if ! id ampixa-arena >/dev/null 2>&1; then
  useradd --system --create-home --home-dir /var/lib/ampixa-arena --shell /bin/bash ampixa-arena
fi
install -d -o ampixa-arena -g ampixa-arena -m 700 /var/lib/ampixa-arena
install -d -o ampixa-arena -g ampixa-arena -m 755 /opt/ampixa-arena /opt/ampixa-arena/releases
install -d -m 750 -o root -g ampixa-arena /etc/ampixa-arena
if [[ ! -f /etc/ampixa-arena/environment ]]; then
  umask 077
  python3 - <<'PY'
import secrets
from pathlib import Path
Path('/etc/ampixa-arena/environment').write_text('NEPTTS_ORIGIN=https://ampixa.benchmarks.cloud\nNEPTTS_ADMIN_PASSWORD='+secrets.token_urlsafe(40)+'\n')
PY
fi
install -o root -g root -m 755 receive.py /usr/local/bin/ampixa-arena-receive
install -o root -g root -m 644 ampixa-arena.service /etc/systemd/system/ampixa-arena.service
printf 'ampixa-arena ALL=(root) NOPASSWD: /usr/bin/systemctl restart ampixa-arena.service\n' > /etc/sudoers.d/ampixa-arena
chmod 440 /etc/sudoers.d/ampixa-arena
visudo -cf /etc/sudoers.d/ampixa-arena
systemctl daemon-reload
# No service is started or enabled until the first release is installed.
