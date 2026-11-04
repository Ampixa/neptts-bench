"""Post-merge reviewer import/approval on the arena VM; never print secrets.

This operator command is deliberately separate from inference and deployment.
Credentials are read with sudo into memory; only numeric public reports are sent.
"""
import argparse
import hashlib
import http.cookiejar
import json
from pathlib import Path
import shlex
import subprocess
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployed-commit', required=True)
    parser.add_argument('--mode', choices=['import', 'approve'], required=True)
    parser.add_argument('--note', default='Reviewed arena_d32_rescore_v1 calibration, provenance and complete205-prompt reports.')
    args = parser.parse_args()
    if len(args.deployed_commit) != 40 or any(c not in '0123456789abcdef' for c in args.deployed_commit):
        raise ValueError('Expected a complete deployed merge commit SHA')
    folder = Path(__file__).resolve().parent
    release = folder.parents[1]
    if (release / 'SOURCE_COMMIT').read_text().strip() != args.deployed_commit:
        raise RuntimeError('STOP: deployed commit differs from reviewed merged commit')
    inventory = json.loads((folder / 'approved_inventory.json').read_text())
    secret_file = subprocess.run(['sudo', 'cat', '/etc/ampixa-arena/environment'],
                                 check=True, capture_output=True, text=True).stdout
    env = {}
    for line in secret_file.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        values = shlex.split(value)
        env[key.strip()] = values[0] if len(values) == 1 else value
    origin = 'https://ampixa.benchmarks.cloud'
    jar = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    def request(path, body=None):
        data = json.dumps(body, allow_nan=False).encode() if body is not None else None
        with client.open(urllib.request.Request(origin + path, data=data,
                headers={'Content-Type': 'application/json', 'X-Neptts-Request': 'submission'}), timeout=120) as response:
            return json.load(response)
    request('/api/admin/login', {'password': env['NEPTTS_ADMIN_PASSWORD']})
    def check_legacy():
        current = {row['id']: row for row in request('/api/admin/submissions')}
        hashes = {}
        for old in inventory['approved']:
            sid = old['id']
            row = current[sid]
            if row['state'] != 'published' or row['audio_sha256'] != old['audio_sha256']:
                raise RuntimeError('STOP: legacy state/audio changed for ID ' + sid)
            if digest(row['report']) != old['report_sha256']:
                raise RuntimeError('STOP: legacy report changed for ID ' + sid)
            hashes[sid] = digest(row['report'])
        return hashes
    before = check_legacy()
    events = []
    for old in inventory['approved']:
        sid = old['id']
        payload = json.loads((folder / ('result-' + sid + '.json')).read_text())
        profile = payload['report']['profile']['id']
        base = '/api/admin/submissions/' + sid + '/profiles'
        existing = {row['profile']: row for row in request(base)}
        item = existing.get(profile)
        if item is not None and digest(item['report']) != digest(payload['report']):
            raise RuntimeError('STOP: existing D32 report differs for ID ' + sid)
        if args.mode == 'import':
            if item is None:
                result = request(base + '/' + profile + '/results', payload)
            elif item['state'] in ['awaiting_review', 'published']:
                result = dict(id=sid, profile=profile, state=item['state'], unchanged=True)
            else:
                raise RuntimeError('STOP: existing D32 report is rejected for ID ' + sid)
        else:
            if item is None:
                raise RuntimeError('Import is required before approval for ID ' + sid)
            if item['state'] == 'published':
                result = dict(id=sid, profile=profile, state='published', unchanged=True)
            else:
                result = request(base + '/' + profile + '/decision', {'action': 'approve', 'note': args.note})
        events.append(result)
        check_legacy()
    after = check_legacy()
    if before != after:
        raise RuntimeError('Legacy report hashes changed')
    print(json.dumps(dict(mode=args.mode, deployed_commit=args.deployed_commit,
                         legacy_report_hashes_before=before, legacy_report_hashes_after=after,
                         legacy_reports_unchanged=True, results=events), indent=2))


if __name__ == '__main__':
    main()
