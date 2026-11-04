"""Reproducible paired numerical analysis of public reports; no audio or ASR APIs."""
import argparse
import datetime
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'neptts-eval/src'))
from neptts_eval import d32

B = 10000
SEED = 20261104
FIELDS = ['cer', 'wer', 'char_errors', 'chars', 'word_errors', 'words']


def read(path):
    value = json.loads(path.read_text())
    print('SCHEMA', path.name, sorted(value) if isinstance(value, dict) else sorted(value[0]))
    return value


def ci(a):
    return np.quantile(a, [.025, .975]).tolist()


def kendall(a, b):
    pairs = list(itertools.combinations(range(a.shape[1]), 2))
    x = np.stack([np.sign(a[:, i] - a[:, j]) for i, j in pairs], axis=1)
    y = np.stack([np.sign(b[:, i] - b[:, j]) for i, j in pairs], axis=1)
    denom = np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))
    if np.any(denom == 0):
        raise ValueError('Kendall tau undefined for tied complete ordering')
    return (x * y).sum(axis=1) / denom


def draws(matrix, weights):
    out = {}
    for name, numerator, denominator in [('cer', 'cer', None), ('wer', 'wer', None),
            ('corpus_cer', 'char_errors', 'chars'), ('corpus_wer', 'word_errors', 'words')]:
        a = matrix[:, FIELDS.index(numerator)]
        if denominator is None:
            out[name] = 100 * (weights @ a) / weights.sum(axis=1)
        else:
            out[name] = 100 * (weights @ a) / (weights @ matrix[:, FIELDS.index(denominator)])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=Path, default=Path(__file__).parent)
    args = parser.parse_args()
    folder = args.input
    inventory = read(folder / 'approved_inventory.json')
    gate = read(folder / 'calibration_gate.json')
    if not gate['passed'] or gate['changed_hypotheses_vs_calibration']:
        raise ValueError('Calibration reproduction failed')
    sentences = read(ROOT / 'benchmark/sentences.json')
    prompt = {r['sent_id']: r for r in sentences if r['sent_id'].startswith('sent_')}
    ids = sorted(prompt)
    if len(ids) != 205 or inventory['approved_count'] != 5:
        raise ValueError('Panel coverage mismatch')
    systems, old = {}, {}
    for item in inventory['approved']:
        sid = item['id']
        payload = read(folder / ('result-' + sid + '.json'))
        if payload['submission_id'] != sid or payload['audio_sha256'] != item['audio_sha256']:
            raise ValueError('Submission binding mismatch')
        d32.validate_profile(payload['report']['profile'], payload['benchmark'])
        if payload['report']['profile'] != gate['profile'] or payload['benchmark'] != gate['profile']['benchmark']:
            raise ValueError('Report instrument/benchmark differs from calibration gate')
        for report in [payload['report'], item['report']]:
            if set(report['per_sentence']) != set(ids):
                raise ValueError('Incomplete system')
        systems[sid] = {'name': item['system_name'], 'kind': 'approved_submission',
                        'rows': payload['report']['per_sentence']}
        old[sid] = item['report']['per_sentence']
    refs = read(folder / 'reference_numeric.json')
    for name, rows in refs['systems'].items():
        if set(rows) != set(ids):
            raise ValueError('Incomplete reference system')
        systems[name] = dict(name=name, kind='reference_only', rows=rows)
    output = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        profile=gate['profile'], n_prompts=205, bootstrap_draws=B, seed=SEED,
        calibration=gate['cohorts'], analyses={})
    for grouped in [False, True]:
        groups = sorted({prompt[sid].get('pair_id') or sid for sid in ids}) if grouped else ids
        index = {name: i for i, name in enumerate(groups)}
        assignment = [index[prompt[sid].get('pair_id') or sid] if grouped else index[sid] for sid in ids]
        sampled = np.random.default_rng(SEED).integers(0, len(groups), (B, len(groups)))
        counts = np.zeros((B, len(groups)), dtype=np.int16)
        np.add.at(counts, (np.arange(B)[:, None], sampled), 1)
        weights = counts[:, assignment].astype(float)
        summary, distributions, points = {}, {}, {}
        for key, system in systems.items():
            values = np.array([[system['rows'][sid][field] for field in FIELDS] for sid in ids])
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError('Invalid numeric observations')
            distributions[key] = draws(values, weights)
            points[key] = {k: float(v[0]) for k, v in draws(values, np.ones((1, len(ids)))).items()}
            summary[key] = dict(name=system['name'], kind=system['kind'],
                **{k: dict(percent=points[key][k], ci95=ci(v)) for k, v in distributions[key].items()})
        contrasts = []
        for a, b in itertools.combinations(systems, 2):
            contrasts.append(dict(a=a, b=b, **{metric: dict(delta_pp=points[a][metric] - points[b][metric],
                ci95=ci(distributions[a][metric] - distributions[b][metric]))
                for metric in ['cer', 'wer', 'corpus_cer', 'corpus_wer']}))
        keys = list(old)
        legacy_values = np.array([[old[key][sid]['cer'] for key in keys] for sid in ids])
        legacy_point = 100 * legacy_values.mean(axis=0)
        legacy_draws = 100 * (weights @ legacy_values) / weights.sum(axis=1)[:, None]
        new_point = np.array([points[key]['cer'] for key in keys])
        new_draws = np.stack([distributions[key]['cer'] for key in keys], axis=1)
        tau_point = float(kendall(legacy_point[None, :], new_point[None, :])[0])
        tau_draws = kendall(legacy_draws, new_draws)
        output['analyses']['pair_groups' if grouped else 'prompts'] = dict(
            bootstrap_groups=len(groups), systems=summary, paired_contrasts=contrasts,
            approved_d32_rank=sorted(keys, key=lambda k: points[k]['cer']),
            approved_whisper_rank=sorted(keys, key=lambda k: float(np.mean([old[k][sid]['cer'] for sid in ids]))),
            kendall_tau_b=dict(point=tau_point, ci95=ci(tau_draws)),
            whisper_scores={key: {'cer': dict(percent=float(legacy_point[j]), ci95=ci(legacy_draws[:, j])),
                'wer': dict(percent=float(100*np.mean([old[key][sid]['wer'] for sid in ids])),
                            ci95=ci(100*(weights @ np.array([old[key][sid]['wer'] for sid in ids]))/weights.sum(axis=1)))}
                for j, key in enumerate(keys)})
    (folder / 'numbers.json').write_text(json.dumps(output, allow_nan=False, indent=2) + '\n')
    print('Wrote complete8-system205-prompt paired analysis')


if __name__ == '__main__':
    main()
