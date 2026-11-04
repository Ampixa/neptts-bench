"""Score all five approved packages and verified reference outputs on k2 CPU."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from neptts_eval import d32
from submission_app.contract import BENCHMARK, SENTENCES
from submission_app.offline import validate_import
from calibrate_k2 import config

ROOT = Path.home() / 'bench2/arena_d32_rescore_v1'
OLD = ROOT.parent / 'arena_judges_v1'


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    gate = json.loads((ROOT / 'calibration_gate.json').read_text())
    if not gate['passed'] or gate['profile'] != d32.profile(BENCHMARK):
        raise RuntimeError('STOP: final executable profile lacks exact calibration gate')
    inventory = json.loads((ROOT / 'approved_inventory.json').read_text())
    manifest = []
    env = dict(os.environ, TMPDIR=str(ROOT / 'private_tmp'), PYTHONDONTWRITEBYTECODE='1')
    Path(env['TMPDIR']).mkdir(exist_ok=True, mode=0o700)
    for row in inventory['approved']:
        sid = row['id']
        package = ROOT / ('package-' + sid + '.zip')
        with zipfile.ZipFile(package) as archive:
            print('SCHEMA reviewer package job', sorted(json.loads(archive.read('evaluation-job.json'))), flush=True)
            entries = []
            for item in archive.infolist():
                if item.filename.startswith('audio/'):
                    prompt = Path(item.filename).stem
                    if prompt not in SENTENCES:
                        raise RuntimeError('Unexpected private audio ID')
                    entries.append(dict(id=prompt, bytes=item.file_size,
                                        sha256=hashlib.sha256(archive.read(item)).hexdigest()))
            if len(entries) != len(SENTENCES):
                raise RuntimeError('Incomplete package inventory')
        manifest.append(dict(submission_id=sid, package_bytes=package.stat().st_size,
                             package_sha256=d32.sha256(package), audio=entries))
        cmd = [sys.executable, '-B', '-m', 'submission_app.offline', '--package', str(package),
               '--output', str(ROOT / ('result-' + sid + '.json')),
               '--profile', d32.PROFILE_NAME]
        for key in ['checkpoint', 'bpe', 'recipe', 'icefall', 'normalizer_root']:
            cmd.extend(['--' + key.replace('_', '-'), str(config()[key])])
        with (ROOT / ('score-' + sid + '.log')).open('x') as log:
            subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT, env=env, cwd=ROOT / 'repo')
        payload = json.loads((ROOT / ('result-' + sid + '.json')).read_text())
        validate_import(payload, sid, row['audio_sha256'])
        if any(Path(env['TMPDIR']).iterdir()):
            raise RuntimeError('Offline extraction left temporary audio behind')
        print('Completed approved ID', sid, len(payload['report']['per_sentence']), flush=True)
        write('private_audio_manifest.json', manifest)
    write('package_scoring_completion.json', {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'n_submissions': len(manifest), 'n_prompts_each': len(SENTENCES),
          'temporary_extraction_verified_empty_after_each': True})

    # The frozen calibration job already decoded these exact615 public-reference
    # outputs using the identical instrument; rehash audio and raw hypotheses.
    source = OLD / 'ranking/d32_full_fleurs.jsonl'
    if d32.sha256(source) != '54970ffe2eb44c14ba183e2026736d71865a67a049a9a382ca96598a2faf01b9':
        raise RuntimeError('Frozen reference hypotheses changed')
    raw = [json.loads(line) for line in source.read_text().splitlines()]
    print('SCHEMA reference hypothesis', sorted(raw[0]), flush=True)
    score = d32.load_normalizer(ROOT)
    chosen = ['swar', 'vox2_clone', 'cosy3_elx_neutral']
    refs = {s: {} for s in chosen}
    for row in raw:
        if row['system'] not in refs:
            continue
        if row['status'] != 'ok' or d32.sha256(row['audio']) != row['sha256']:
            raise RuntimeError('Reference waveform/status drift')
        sid = row['id']
        if sid not in SENTENCES or row['text'] != SENTENCES[sid]['text_devanagari']:
            raise RuntimeError('Reference prompt drift')
        if sid in refs[row['system']]:
            raise RuntimeError('Duplicate reference row')
        refs[row['system']][sid] = score(row['text'], row['hyp'])
    if any(set(rows) != set(SENTENCES) for rows in refs.values()):
        raise RuntimeError('Incomplete reference coverage')
    write('reference_numeric.json', {'source_sha256': d32.sha256(source), 'systems': refs})
    print('Verified and rescored references', len(refs), 'x', len(SENTENCES), flush=True)


if __name__ == '__main__':
    main()
