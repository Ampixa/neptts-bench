"""Recreate only phase1b controls, verifying all audio against its frozen hashes."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, resample_poly, sosfilt

ROOT = Path('/mnt/wd_data/arena_judges_v1/calibration')
OLD = Path('/mnt/wd_data/nepclone_bench_v1/phase1b')


def read(path):
    value = json.loads(path.read_text())
    print('SCHEMA', path.name, sorted(value[0] if isinstance(value, list) else value), flush=True)
    return value


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT/'controls').mkdir(exist_ok=True)
    panel = read(OLD/'panel.json')
    frozen = {r['audio']: r for r in read(OLD/'audio_tasks.json')}
    tasks = []
    for row in panel:
        real = Path(row['real_audio'])
        assert sha(real) == frozen[str(real)]['sha256'], 'Source audio drift'
        x, sr = sf.read(real, dtype='float32', always_2d=True)
        g = math.gcd(sr, 16000)
        x = resample_poly(x.mean(axis=1), 16000//g, sr//g).astype(np.float32)
        low = sosfilt(butter(5, [300, 3400], btype='bandpass', fs=16000, output='sos'), x)
        band = resample_poly(resample_poly(low, 1, 2), 2, 1)[:len(x)]
        rms = max(float(np.sqrt(np.mean(band**2))), 1e-5)
        seed = int(hashlib.sha256(row['id'].encode()).hexdigest()[:8], 16)
        noise = np.random.default_rng(seed).standard_normal(len(band))
        noise *= rms/np.sqrt(np.mean(noise**2))
        degraded = np.clip(band+noise, -rms, rms).astype(np.float32)
        target = ROOT/'controls'/(row['id']+'.wav')
        if target.exists():
            raise FileExistsError(target)
        sf.write(target, degraded, 16000, subtype='PCM_16')
        assert sha(target) == frozen[row['degraded_audio']]['sha256'], 'Control drift'
        for role, audio in [('real', real), ('degraded', target)]:
            tasks.append(dict(id=row['id'], voice=row['voice'], cohort=row['cohort'],
                              role=role, audio=str(audio), sha256=sha(audio)))
    retest = []
    for cohort in ['pv', 'fresh']:
        for role in ['real', 'degraded']:
            byvoice = {t['voice']: t for t in tasks if t['cohort'] == cohort and t['role'] == role}
            retest.extend(byvoice[v] for v in sorted(byvoice))
    assert len(tasks) == 392 and len(retest) == 32
    for name, rows in [('tasks.json', tasks), ('retest.json', retest)]:
        (ROOT/name).write_text(json.dumps(rows, indent=2))
    print('Verified', len(tasks), 'inputs; cold retests', len(retest), flush=True)


if __name__ == '__main__':
    main()
