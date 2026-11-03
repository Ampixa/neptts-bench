"""Freeze local system audio and verify arena prompt identity without copying WAVs."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path('/mnt/wd_data/arena_judges_v1/ranking')
SOURCE = Path('/mnt/wd_data/swar_headtohead_v1')
SYSTEMS = ['swar', 'vox2_beat1', 'vox2_beat2', 'cosy3_elx_neutral', 'vox2_clone']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    arena = json.loads((ROOT / 'arena_sentences.json').read_text())
    prompts = {r['sent_id']: r for r in arena if r['sent_id'].startswith('sent_')}
    assert len(prompts) == 205
    source = json.loads((SOURCE / 'manifest.json').read_text())
    print('SCHEMA source', sorted(source), 'job', sorted(source['jobs'][0]), flush=True)
    tasks, receipts, reused = [], [], []
    for system in SYSTEMS:
        directory = SOURCE / 'runs/primary' / system
        items = [json.loads(s) for s in (directory / 'items.jsonl').read_text().splitlines()]
        print('SCHEMA items', system, sorted(items[0]), flush=True)
        indexed = {r['key']: r for r in items}
        scores_path = SOURCE / 'scores' / system / 'mms_items.jsonl'
        scores = [json.loads(s) for s in scores_path.read_text().splitlines()]
        print('SCHEMA MMS', system, sorted(scores[0]), flush=True)
        old = {r['key']: r for r in scores}
        for condition in ['fleurs', 'rasa']:
            jobs = [r for r in source['jobs'] if r['condition'] == condition]
            assert len(jobs) == 205 and {r['id'] for r in jobs} == set(prompts)
            for job in jobs:
                assert job['text'] == prompts[job['id']]['text_devanagari']
                path = directory / condition / (job['id'] + '.wav')
                digest = sha(path)
                item = indexed[job['key']]
                assert item['status'] == 'ok' and item['sha256'] == digest
                row = dict(key=f'{system}/{job["key"]}', system=system, condition=condition,
                           id=job['id'], text=job['text'], category=job['category'],
                           audio=str(path), sha256=digest, bytes=path.stat().st_size)
                tasks.append(row)
                mms = old[job['key']]
                assert mms['status'] == 'ok' and mms['sha256'] == digest
                reused.append(dict(key=row['key'], sha256=digest, hyp=mms['hyp'], status='ok',
                                   source=str(scores_path), source_sha256=sha(scores_path)))
        model_path = directory / 'model.json'
        receipts.append(dict(system=system, model=json.loads(model_path.read_text()),
                             model_receipt_sha256=sha(model_path), items_sha256=sha(directory / 'items.jsonl'),
                             mms_source_sha256=sha(scores_path)))
    assert len(tasks) == 2050
    (ROOT / 'tasks.json').write_text(json.dumps(tasks, ensure_ascii=False, indent=2))
    (ROOT / 'mms_reused.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in reused))
    (ROOT / 'source_receipts.json').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        source_manifest_sha256=sha(SOURCE/'manifest.json'), arena_sha256=sha(ROOT/'arena_sentences.json'),
        systems=receipts, n_prompts=205, n_audio=len(tasks)), ensure_ascii=False, indent=2))
    print('VERIFIED', len(tasks), 'system/condition/prompt WAV hashes; no audio copied', flush=True)

if __name__ == '__main__':
    main()
