"""Paired prompt-bootstrap system contrasts and judge rank sensitivity."""
from functools import lru_cache
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import sys
import time

import numpy as np

ROOT=Path('/mnt/wd_data/arena_judges_v1/ranking')
sys.path[:0]=['/mnt/wd_data/tts_eval_suite_v1/code','/mnt/wd_data/swar_headtohead_v1/repos/swar',
             '/mnt/wd_data/swar_headtohead_v1/repos/neptts-bench/neptts-eval/src']
NORMALIZATION_MANIFEST=Path('/mnt/wd_data/nepclone_bench_v1/phase1b/normalization_manifest.json')
NORMALIZATION_PINS=json.loads(NORMALIZATION_MANIFEST.read_text())
for pin in NORMALIZATION_PINS['files']:
    if hashlib.sha256(Path(pin['original_path']).read_bytes()).hexdigest()!=pin['sha256']:
        raise RuntimeError('Legacy normalizer source changed: '+pin['original_path'])
from suite_common import cer_wer
from frontend import normalize
from neptts_eval.normalize import normalize_nepali_text

SEED=20261104
B=10000
LAT=re.compile('[A-Za-z]')
SYSTEMS=['swar','vox2_beat1','vox2_beat2','cosy3_elx_neutral','vox2_clone']


def read_rows(path):
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    if rows:
        print('SCHEMA',path.name,sorted(rows[0]),flush=True)
    return rows


@lru_cache(maxsize=None)
def scores(ref,hyp,policy):
    if policy=='legacy_transliterated':
        hyp=normalize(hyp,english=True,nepal=False) if LAT.search(hyp) and not LAT.search(ref) else hyp
    if policy=='arena_raw':
        ref,hyp=normalize_nepali_text(ref).replace(" ",""),normalize_nepali_text(hyp).replace(" ","")
        # Reuse the phase1b exact edit-distance implementation on normalized strings.
        from suite_common import edit_distance
        errors=edit_distance(ref,hyp)
        return dict(char_errors=errors,chars=len(ref),cer=errors/max(1,len(ref)))
    return cer_wer(ref,hyp)


def ci(samples):
    return np.quantile(samples,[.025,.975]).tolist()


def tau(a,b):
    """Vectorized Kendall tau-b over five systems, one row per paired bootstrap."""
    pairs=list(itertools.combinations(range(len(SYSTEMS)),2))
    x=np.stack([np.sign(a[:,i]-a[:,j]) for i,j in pairs],axis=1)
    y=np.stack([np.sign(b[:,i]-b[:,j]) for i,j in pairs],axis=1)
    denominator=np.sqrt(np.sum(x*x,axis=1)*np.sum(y*y,axis=1))
    result=np.full(len(a),np.nan)
    np.divide(np.sum(x*y,axis=1),denominator,out=result,where=denominator!=0)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pair-clusters',action='store_true')
    args=parser.parse_args()
    prompts={r['sent_id']:r for r in json.loads((ROOT/'arena_sentences.json').read_text()) if r['sent_id'].startswith('sent_')}
    tasks=json.loads((ROOT/'tasks.json').read_text())
    print('SCHEMA tasks',sorted(tasks[0]),flush=True)
    output=dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),seed=SEED,bootstrap_draws=B,
                normalization_manifest_sha256=hashlib.sha256(NORMALIZATION_MANIFEST.read_bytes()).hexdigest(),
                normalization_source_pins=[dict(path=p['original_path'],sha256=p['sha256']) for p in NORMALIZATION_PINS['files']],
                bootstrap_unit=('paired minimal-pair group or singleton prompt' if args.pair_clusters else 'paired prompt')+'; conditional on fixed system, reference voice and saved synthesis take',
                analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                normalizations=['suite_raw','legacy_transliterated','arena_raw'],conditions={},sources={})
    for condition in ['fleurs','rasa']:
        chosen=[r for r in tasks if r['condition']==condition]
        indexed={r['key']:r for r in chosen}
        ids=sorted({r['id'] for r in chosen})
        assert len(ids)==205 and len(chosen)==1025
        inputs={}
        for judge,pattern in [('whisper_small',f'whisper_{condition}_shard*.jsonl'),('mms',f'mms_{condition}.jsonl'),
                              ('long1c',f'long1c_{condition}.jsonl'),('d32_streaming',f'd32_streaming_{condition}.jsonl'),
                              ('d32_full',f'd32_full_{condition}.jsonl')]:
            paths=sorted(ROOT.glob(pattern))
            if not paths:
                continue
            rows=[]
            for path in paths:
                rows.extend(read_rows(path))
                output['sources'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            rows=[r for r in rows if not r.get('repeat',False)]
            bykey={r['key']:r for r in rows}
            if len(rows)!=1025 or set(bykey)!=set(indexed):
                raise RuntimeError(f'{judge}/{condition}: incomplete or duplicated observations {len(rows)}')
            for key,row in bykey.items():
                if row['status']!='ok' or row['sha256']!=indexed[key]['sha256']:
                    raise RuntimeError(f'Invalid {judge} observation {key}')
            inputs[judge]=bykey
        if not inputs:
            continue
        index=np.random.default_rng(SEED).integers(0,len(ids),(B,len(ids)))
        weights=None
        groups=sorted({prompts[sid].get('pair_id') or sid for sid in ids})
        if args.pair_clusters:
            draw=np.random.default_rng(SEED).integers(0,len(groups),(B,len(groups)))
            counts=np.zeros((B,len(groups)),dtype=np.int16)
            np.add.at(counts,(np.arange(B)[:,None],draw),1)
            group_index={group:i for i,group in enumerate(groups)}
            weights=counts[:,[group_index[prompts[sid].get('pair_id') or sid] for sid in ids]].astype(float)
        condition_output=dict(n_prompts=205,n_systems=5,bootstrap_groups=len(groups) if args.pair_clusters else 205,policies={})
        for policy in ['suite_raw','legacy_transliterated','arena_raw']:
            ranks={};draws={};point={};corpus_draws={};corpus_point={};aggregate={};contrasts=[]
            for judge,rows in inputs.items():
                e=np.zeros((205,5));d=np.zeros((205,5));x=np.zeros((205,5))
                for i,sid in enumerate(ids):
                    for j,system in enumerate(SYSTEMS):
                        key=f'{system}/{condition}/{sid}'
                        task=indexed[key];row=rows[key]
                        result=scores(task['text'],row['hyp'],policy)
                        e[i,j],d[i,j],x[i,j]=result['char_errors'],result['chars'],result['cer']
                point[judge]=100*x.mean(axis=0)
                draws[judge]=100*x[index,:].mean(axis=1) if weights is None else 100*(weights@x)/weights.sum(axis=1)[:,None]
                corpus_point[judge]=100*e.sum(axis=0)/d.sum(axis=0)
                corpus_draws[judge]=100*e[index,:].sum(axis=1)/d[index,:].sum(axis=1) if weights is None else 100*(weights@e)/(weights@d)
                aggregate[judge]={system:dict(mean_sentence_cer_percent=float(point[judge][j]),
                    ci95_mean_sentence=ci(draws[judge][:,j]),corpus_cer_percent=float(corpus_point[judge][j]),
                    ci95_corpus=ci(corpus_draws[judge][:,j])) for j,system in enumerate(SYSTEMS)}
                ranks[judge]=sorted(SYSTEMS,key=lambda s:point[judge][SYSTEMS.index(s)])
                for a,b in itertools.combinations(range(5),2):
                    contrasts.append(dict(judge=judge,a=SYSTEMS[a],b=SYSTEMS[b],
                        sentence_delta_pp=float(point[judge][a]-point[judge][b]),
                        ci95_sentence_delta=ci(draws[judge][:,a]-draws[judge][:,b]),
                        corpus_delta_pp=float(corpus_point[judge][a]-corpus_point[judge][b]),
                        ci95_corpus_delta=ci(corpus_draws[judge][:,a]-corpus_draws[judge][:,b])))
            comparisons=[]
            for a,b in itertools.combinations(inputs,2):
                boot=tau(draws[a],draws[b]);valid=boot[np.isfinite(boot)]
                comparisons.append(dict(judge_a=a,judge_b=b,
                    kendall_tau_b=float(tau(point[a][None,:],point[b][None,:])[0]),
                    ci95_tau=ci(valid),undefined_bootstraps=int(B-len(valid)),
                    same_order_bootstrap_fraction=float(np.mean(valid==1)),
                    system_deltas={system:dict(sentence_delta_pp=float(point[a][j]-point[b][j]),
                        ci95_sentence_delta=ci(draws[a][:,j]-draws[b][:,j])) for j,system in enumerate(SYSTEMS)}))
            condition_output['policies'][policy]=dict(aggregates=aggregate,ranks=ranks,
                 paired_system_contrasts=contrasts,judge_comparisons=comparisons)
        output['conditions'][condition]=condition_output
    filename='numbers_pair_clusters.json' if args.pair_clusters else 'numbers.json'
    (ROOT/filename).write_text(json.dumps(output,indent=2))
    print('WROTE',filename,flush=True)


if __name__=='__main__':
    main()
