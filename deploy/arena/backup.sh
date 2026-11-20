#!/usr/bin/env bash
# Daily consistent backup of the arena: SQLite snapshot, submitted audio, protected env file.
# Installed as /usr/local/bin/ampixa-arena-backup and run by ampixa-arena-backup.timer (as root).
#
# Settings in /etc/ampixa-arena/backup.env:
#   BACKUP_DIR=/var/backups/ampixa-arena          local staging, kept 7 days
#   BACKUP_RECIPIENT=age1...                      age public key; private key kept OFF the VM
#   BACKUP_UPLOAD="rclone copy {} ampixa-b2:ampixa-arena-backups/"   {} = the encrypted file
#
# The readiness marker is touched only after the encrypted archive was uploaded off-host.
set -euo pipefail
source /etc/ampixa-arena/backup.env
DATA=/var/lib/ampixa-arena
: "${BACKUP_DIR:=/var/backups/ampixa-arena}" "${BACKUP_RECIPIENT:?set BACKUP_RECIPIENT}" "${BACKUP_UPLOAD:?set BACKUP_UPLOAD}"
install -d -m 700 "$BACKUP_DIR"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
work=$(mktemp -d "$BACKUP_DIR/.work-XXXX")
trap 'rm -rf "$work"' EXIT

# Online, consistent SQLite copy (no need to stop the service).
python3 - "$DATA/submissions.db" "$work/submissions.db" <<'PY'
import sqlite3, sys
src, dst = sqlite3.connect(sys.argv[1]), sqlite3.connect(sys.argv[2])
src.backup(dst)
assert dst.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
PY
cp /etc/ampixa-arena/environment "$work/environment"
archive="$BACKUP_DIR/ampixa-arena-$stamp.tar.gz.age"
tar -czf - -C "$work" submissions.db environment -C "$DATA" uploads \
  | age -r "$BACKUP_RECIPIENT" -o "$archive"
chmod 600 "$archive"

eval "${BACKUP_UPLOAD//\{\}/\"$archive\"}"
touch "$DATA/.last-backup"
chown ampixa-arena:ampixa-arena "$DATA/.last-backup"
find "$BACKUP_DIR" -name 'ampixa-arena-*.age' -mtime +7 -delete
echo "Backup $archive uploaded."
