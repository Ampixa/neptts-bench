"""Delete only this job's consumed private audio copies, keeping a hash ledger."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path.home() / 'bench2/arena_d32_rescore_v1'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    inventory = json.loads((ROOT / 'approved_inventory.json').read_text())
    manifest_path = ROOT / 'private_audio_manifest.json'
    manifest = json.loads(manifest_path.read_text())
    done = json.loads((ROOT / 'package_scoring_completion.json').read_text())
    retest = json.loads((ROOT / 'submission_retest.json').read_text())
    if done['n_submissions'] != 5 or done['n_prompts_each'] != 205 or not retest['passed']:
        raise RuntimeError('Consumers are incomplete')
    if {r['submission_id'] for r in manifest} != {r['id'] for r in inventory['approved']}:
        raise RuntimeError('Private copy manifest mismatch')
    temp = ROOT / 'private_tmp'
    if any(temp.iterdir()):
        raise RuntimeError('Unexpected retained extracted input')
    targets = [ROOT / ('package-' + r['submission_id'] + '.zip') for r in manifest]
    if any(p.is_symlink() or p.parent != ROOT for p in targets):
        raise RuntimeError('Unexpected cleanup target')
    probe = subprocess.run(['lsof', '-nP', '--', *map(str, targets)], capture_output=True, text=True)
    (ROOT / 'cleanup_open_files.log').write_text(json.dumps({'returncode': probe.returncode,
            'stdout': probe.stdout, 'stderr': probe.stderr}, indent=2) + '\n')
    if probe.returncode not in (0, 1) or probe.stdout.strip():
        raise RuntimeError('Open files or inconclusive lsof; refuse cleanup')
    records = []
    for row, path in zip(manifest, targets):
        if sha(path) != row['package_sha256'] or path.stat().st_size != row['package_bytes']:
            raise RuntimeError('Private package drift before deletion')
        records.append(dict(submission_id=row['submission_id'], bytes=path.stat().st_size,
            allocated_bytes=path.stat().st_blocks * 512, sha256=row['package_sha256'],
            reason='Complete205-prompt result +21-prompt cold numeric retest; original arena upload retained'))
    free_before = shutil.disk_usage(ROOT).free
    # Retain the detailed ID/hash/size manifest on k2; never audio or transcripts.
    for path in targets:
        path.unlink()
    if any(p.exists() for p in targets) or list(temp.iterdir()):
        raise RuntimeError('Private copy cleanup verification failed')
    remaining = list(ROOT.rglob('*.wav')) + list(ROOT.rglob('*.zip'))
    if remaining:
        raise RuntimeError('Unexpected remaining audio/archive in job directory')
    free_after = shutil.disk_usage(ROOT).free
    output = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Only five new reviewer-package ZIP copies and transient extracted WAVs in this job',
        originals='Arena VM originals unchanged; initial reviewer export verified every original WAV SHA; same legacy/audio hashes rechecked after scoring',
        source_data_unique=False, no_active_consumer=True, lsof_returncode=probe.returncode,
        detailed_audio_manifest_sha256=sha(manifest_path), n_packages_deleted=len(records),
        n_audio_members=1025, n_retest_temporary_copies=105,
        extraction_cleanup='TemporaryDirectory removed each offline extraction and retest copy; parent verified empty after every consumer',
        package_bytes_deleted=sum(r['bytes'] for r in records),
        allocated_package_bytes_deleted=sum(r['allocated_bytes'] for r in records),
        free_bytes_before=free_before, free_bytes_after=free_after, measured_free_delta_bytes=free_after-free_before,
        measured_delta_limit='Host-concurrent activity may affect free-space delta',
        verified_zero_job_wav_zip=True, records=records,
        restoration='Original arena uploads remain reviewer-only; authorized re-download required. Exact ZIP container serialization may differ; audio hashes bind exact WAV bytes.',
        hf='NOT on HF', backups_created=False)
    (ROOT / 'audio_cleanup.json').write_text(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
