"""Remove consumed job-local staging dependencies, calibration panel and duplicates."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path.home() / 'bench2/arena_d32_rescore_v1'


def main():
    if not json.loads((ROOT / 'audio_cleanup.json').read_text())['verified_zero_job_wav_zip']:
        raise RuntimeError('Audio cleanup must finish first')
    targets = [ROOT / n for n in ['validation_deps', 'normalization_snapshot', 'src',
                                  'panel.json', 'calibration_tasks.json', 'calibration_numeric.json']]
    ledger = []
    for target in targets:
        if target.is_symlink():
            raise RuntimeError('Unexpected symbolic cleanup root')
        command = ['lsof', '-nP', '+D', str(target)] if target.is_dir() else ['lsof', '-nP', '--', str(target)]
        check = subprocess.run(command, capture_output=True, text=True)
        if check.returncode not in [0, 1] or check.stdout.strip():
            raise RuntimeError('Open staging artifact or inconclusive lsof')
        files = [p for p in target.rglob('*') if p.is_file()] if target.is_dir() else [target]
        for p in files:
            ledger.append(dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size,
                allocated_bytes=p.stat().st_blocks*512, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    (ROOT / 'scratch_cleanup_manifest.json').write_text(json.dumps(ledger, indent=2) + '\n')
    before = shutil.disk_usage(ROOT).free
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    after = shutil.disk_usage(ROOT).free
    if any(p.exists() for p in targets):
        raise RuntimeError('Staging deletion incomplete')
    record = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        reason='Scoring/retest consumers complete; original normalizer snapshot15hashes reverified on Ubuntu; source panel pin verified at calibration; validator deps reinstallable from pinned pydantic2.12.5; obsolete preflight source superseded by final Git source',
        paths=[p.name for p in targets], n_files=len(ledger), apparent_bytes=sum(r['bytes'] for r in ledger),
        allocated_bytes=sum(r['allocated_bytes'] for r in ledger), free_before=before, free_after=after,
        measured_free_delta_bytes=after-before,
        manifest_sha256=hashlib.sha256((ROOT / 'scratch_cleanup_manifest.json').read_bytes()).hexdigest(),
        no_open_files=True, originals_deleted=False, hf='NOT on HF',
        restoration='Recreate job-local normalizer from verified Ubuntu phase1b snapshot; install pydantic2.12.5 to job target; run prepare_k2.py; private panel requires authorization. Preflight staging layout is disposable; final executable retained in Git/repo copy.')
    (ROOT / 'scratch_cleanup.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
