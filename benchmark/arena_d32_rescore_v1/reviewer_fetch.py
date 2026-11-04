"""Read-only reviewer acquisition on the VM; secrets stay in process memory.

Run over SSH stdin. Package bytes go to stdout and must be piped directly to k2.
Only IDs and public score metadata are emitted by inventory mode.
"""
import argparse
import hashlib
import http.cookiejar
import json
import shlex
import subprocess
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id')
    args = parser.parse_args()
    raw = subprocess.run(['sudo', 'cat', '/etc/ampixa-arena/environment'],
                         check=True, capture_output=True, text=True).stdout
    env = {}
    for line in raw.splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        key, value = line.split('=', 1)
        parts = shlex.split(value)
        env[key.strip()] = parts[0] if len(parts) == 1 else value
    origin = 'https://ampixa.benchmarks.cloud'
    jar = http.cookiejar.CookieJar()
    client = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
    def request(path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        return client.open(urllib.request.Request(origin + path, data=body,
                           headers={'Content-Type': 'application/json', 'X-Neptts-Request': 'submission'}), timeout=120)
    with request('/api/admin/login', {'password': env['NEPTTS_ADMIN_PASSWORD']}) as response:
        if response.status != 200:
            raise RuntimeError('Reviewer login failed')
    with request('/api/admin/submissions') as response:
        rows = json.load(response)
    approved = {row['id']: row for row in rows if row['state'] == 'published'}
    if args.id:
        if args.id not in approved:
            raise RuntimeError('Requested ID is not approved')
        with request('/api/admin/submissions/' + args.id + '/package') as response:
            while chunk := response.read(1024 * 1024):
                sys.stdout.buffer.write(chunk)
        return
    output = []
    for sid, row in sorted(approved.items()):
        report = row['report']
        output.append({'id': sid, 'state': row['state'],
                       'system_name': row['manifest']['system_name'],
                       'model_version': row['manifest']['model_version'],
                       'audio_sha256': row['audio_sha256'],
                       'report_sha256': hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest(),
                       'report': report})
    print(json.dumps({'submission_keys': sorted(rows[0]) if rows else [],
                      'approved_count': len(output), 'approved': output}, allow_nan=False))


if __name__ == '__main__':
    main()
