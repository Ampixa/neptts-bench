"""Prepare job-local rerun tasks after verifying the original instrument pins."""
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path.home() / 'bench2/arena_d32_rescore_v1'
    old = root.parent / 'arena_judges_v1'
    meta = json.loads((old / 'calibration/d32_full.meta.json').read_text())
    pins = json.loads((root / 'instrument_pins.json').read_text())
    print('SCHEMA metadata', sorted(meta), 'pins', sorted(pins))
    for key, value in pins.items():
        if meta.get(key) != value:
            raise RuntimeError('Calibration metadata drift: ' + key)
    paths = {'checkpoint_sha256': Path(meta['checkpoint']), 'bpe_sha256': Path(meta['tokenizer'])}
    verified = {}
    for key, path in paths.items():
        if sha(path) != pins[key]:
            raise RuntimeError('Instrument input hash mismatch: ' + key)
        verified[key] = sha(path)
    for name, digest in pins['recipe_hashes'].items():
        if sha(Path(meta['recipe']) / name) != digest:
            raise RuntimeError('Recipe drift: ' + name)
    worker = old / 'calibration/run_zipformer.py'
    if sha(worker) != pins['worker_sha256']:
        raise RuntimeError('Frozen calibration worker drift')
    tasks = json.loads((old / 'calibration/zip_tasks.json').read_text())
    print('SCHEMA task', sorted(tasks[0]))
    selected = [r for r in tasks if r['role'] == 'real' and not r['repeat']]
    if len(selected) != 196 or len({r['id'] for r in selected}) != 196:
        raise RuntimeError('Calibration coverage drift')
    for row in selected:
        path = old / 'calibration/input' / row['audio'].lstrip('/')
        if {'eval_private', 'khalti', 'calls', 'orbit'} & set(path.parts):
            raise RuntimeError('Forbidden input')
        if sha(path) != row['sha256']:
            raise RuntimeError('Calibration audio hash mismatch')
        row['audio'] = str(path)
    (root / 'calibration_tasks.json').write_text(json.dumps(selected))
    (root / 'instrument_verification.json').write_text(json.dumps({
        'verified_checkpoint_tokenizer': verified, 'verified_recipe': pins['recipe_hashes'],
        'worker_sha256': sha(worker), 'n_calibration_targets': len(selected),
        'source_metadata_sha256': sha(old / 'calibration/d32_full.meta.json'),
        'source_task_sha256': sha(old / 'calibration/zip_tasks.json'),
        'normalizer_manifest_sha256': sha(root / 'normalization_manifest.json')}, indent=2) + '\n')
    print('Verified instrument and', len(selected), 'calibration waveforms; private paths suppressed')


if __name__ == '__main__':
    main()
