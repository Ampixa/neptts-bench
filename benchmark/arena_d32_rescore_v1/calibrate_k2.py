"""Rerun the actual new profile on all196 calibrated real recordings, CPU only."""
import datetime
import json
from pathlib import Path
import subprocess

from neptts_eval import d32

ROOT = Path.home() / 'bench2/arena_d32_rescore_v1'
OLD = ROOT.parent / 'arena_judges_v1'


def config():
    meta = json.loads((OLD / 'calibration/d32_full.meta.json').read_text())
    return dict(checkpoint=meta['checkpoint'], bpe=meta['tokenizer'], recipe=meta['recipe'],
                icefall=OLD / 'icefall', normalizer_root=ROOT, threads=1)


def main():
    subprocess.run(['memory_pressure'], check=True)
    tasks = json.loads((ROOT / 'calibration_tasks.json').read_text())
    panel = json.loads((ROOT / 'panel.json').read_text())
    print('SCHEMA panel', sorted(panel[0]), flush=True)
    if d32.sha256(ROOT / 'panel.json') != 'cde2d472e3f026ebcc8ac85a64152ceb5cef72a14724fc591d533b21e8034d30':
        raise RuntimeError('Reference panel drift')
    refs = {r['id']: r for r in panel}
    baseline = {r['id']: r for r in (json.loads(s) for s in (OLD / 'calibration/d32_full.jsonl').read_text().splitlines())
                if r['role'] == 'real' and not r['repeat']}
    counts = d32.load_normalizer(ROOT)
    decode = d32.build_decoder(config())
    result, drift = [], 0
    for i, task in enumerate(tasks, 1):
        if d32.sha256(task['audio']) != task['sha256']:
            raise RuntimeError('Audio hash drift')
        hyp = decode(d32.read_audio(task['audio']))
        drift += hyp != baseline[task['id']]['hyp']
        row = dict(id=task['id'], cohort=task['cohort'], **counts(refs[task['id']]['text'], hyp))
        result.append(row)
        if i % 20 == 0:
            print('Calibration', i, '/', len(tasks), flush=True)
    expected = {'pv': 7.875536480686695, 'fresh': 13.68944386634293}
    output = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  profile=d32.profile('neptts-synthesis-v1-e85f91670812'),
                  target_count=len(result), changed_hypotheses_vs_calibration=drift, cohorts={})
    for cohort, n in [('pv', 100), ('fresh', 96)]:
        rows = [r for r in result if r['cohort'] == cohort]
        if len(rows) != n:
            raise RuntimeError('Incomplete calibration cohort')
        cer = 100 * sum(r['char_errors'] for r in rows) / sum(r['chars'] for r in rows)
        wer = 100 * sum(r['word_errors'] for r in rows) / sum(r['words'] for r in rows)
        output['cohorts'][cohort] = dict(n=n, corpus_cer_percent=cer, corpus_wer_percent=wer,
            expected_cer_percent=expected[cohort], delta_pp=cer-expected[cohort])
        if abs(cer - expected[cohort]) > .05:
            raise RuntimeError('STOP: calibration threshold failed: ' + cohort)
    if drift:
        raise RuntimeError('STOP: new decoder differs from original full-context hypotheses')
    output['passed'] = True
    (ROOT / 'calibration_gate.json').write_text(json.dumps(output, indent=2) + '\n')
    (ROOT / 'calibration_numeric.json').write_text(json.dumps(result) + '\n')
    print('Calibration gate passed:', json.dumps(output['cohorts']), flush=True)


if __name__ == '__main__':
    main()
