"""Pipe approved reviewer packages VM -> k2 without a local audio copy."""
import datetime
import json
from pathlib import Path
import subprocess


def main():
    here = Path(__file__).parent
    inventory = json.loads((here / 'approved_inventory.json').read_text())
    if inventory['approved_count'] != 5:
        raise RuntimeError('Approved snapshot count differs from job request')
    source = (here / 'reviewer_fetch.py').read_bytes()
    records = []
    for row in inventory['approved']:
        sid = row['id']
        if len(sid) != 32 or any(c not in '0123456789abcdef' for c in sid):
            raise RuntimeError('Invalid submission ID')
        sender = subprocess.Popen(['ssh', '-i', str(Path.home() / '.ssh/affine-test-ap-south-1'),
            'ubuntu@3.108.203.159', 'python3 - --id ' + sid],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        sender.stdin.write(source)
        sender.stdin.close()
        # noclobber refuses accidental replacement. No credentials traverse this pipe.
        receiver = subprocess.run(['ssh', 'k2', 'umask 077; set -C; cat > '
            '~/bench2/arena_d32_rescore_v1/package-' + sid + '.zip'], stdin=sender.stdout)
        sender.stdout.close()
        stderr = sender.stderr.read()
        code = sender.wait()
        if code or receiver.returncode:
            raise RuntimeError('Package transfer failed for ID ' + sid + '; sender=' +
                               str(code) + ', receiver=' + str(receiver.returncode) +
                               ', stderr bytes=' + str(len(stderr)))
        records.append({'submission_id': sid, 'sender_returncode': code,
                        'receiver_returncode': receiver.returncode})
        print('Transferred package ID', sid, flush=True)
    (here / 'download_receipt.json').write_text(json.dumps({
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'route': 'reviewer API -> VM stdout -> SSH pipe -> k2 job directory; no Mac audio file',
        'records': records}, indent=2) + '\n')


if __name__ == '__main__':
    main()
