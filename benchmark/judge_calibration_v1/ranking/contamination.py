"""Reuse the existing public-corpus audit and add a finite Common Voice text check."""
import collections
import csv
import hashlib
import json
from pathlib import Path
import unicodedata

ROOT=Path('/mnt/wd_data/arena_judges_v1/ranking')
SOURCE=Path('/mnt/wd_data/swar_contamination_audit_v1/results/text_matches.json')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def norm(text):
    text=unicodedata.normalize('NFC',text).lower()
    return ''.join(c for c in text if unicodedata.category(c)[0] in ['L','M','N'])


def main():
    prompts={r['sent_id']:r['text_devanagari'] for r in json.loads((ROOT/'arena_sentences.json').read_text())
             if r['sent_id'].startswith('sent_')}
    data=json.loads(SOURCE.read_text())
    print('SCHEMA contamination',sorted(data),'row',sorted(data['results'][0]),flush=True)
    chosen=[r for r in data['results'] if r['set']=='neptts_scored']
    assert len(chosen)==205
    assert {r['q'].removeprefix('neptts:'):r['text'] for r in chosen}==prompts
    bycorpus={}
    for corpus,rows in data['corpus_rows'].items():
        counts=collections.Counter();hits=[]
        for row in chosen:
            matches=row['hits'].get(corpus,{})
            kinds=[k for k in matches if not k.startswith('count_')]
            if kinds:
                hits.append(dict(id=row['q'].removeprefix('neptts:'),kinds=kinds))
                counts.update(kinds)
        bycorpus[corpus]=dict(source_rows=rows,matching_queries_by_kind=dict(counts),hits=hits)
    inputs={}
    cvroot=Path('/mnt/wd_data/tts_eval_suite_v1/sources/cv17_ne_np/files/transcript/ne-NP')
    cv={};n=0
    for filename in ['train.tsv','dev.tsv','test.tsv','validated.tsv']:
        path=cvroot/filename
        with path.open() as handle:
            reader=csv.DictReader(handle,delimiter='\t')
            print('SCHEMA CV',filename,reader.fieldnames,flush=True)
            for row in reader:
                if row.get('sentence'):
                    cv.setdefault(norm(row['sentence']),set()).add(filename);n+=1
        inputs[filename]=dict(path=str(path),sha256=digest(path))
    hits=[]
    for sid,text in prompts.items():
        query=norm(text)
        if query in cv:
            hits.append(dict(id=sid,kind='exact',splits=sorted(cv[query])))
        elif len(query)>=15 and any(query in text or (len(text)>=15 and text in query) for text in cv):
            hits.append(dict(id=sid,kind='containment_15char'))
    output=dict(public_audit_source=dict(path=str(SOURCE),sha256=digest(SOURCE)),n_queries=205,
                existing_audit=bycorpus,common_voice17=dict(rows=n,unique_normalized=len(cv),hits=hits,inputs=inputs),
                methods=dict(existing_audit='Existing audit n1/n2 exact, substring and 5gram matches; original code/source hashes in source README',
                             common_voice='NFC lowercase letters/marks/numbers only; exact and >=15-character containment'),
                limitations='Finite pinned public text snapshots only; matches of short common phrases do not establish memorization; absence does not establish training or speaker exclusion. FLEURS/Rasa reference source exposure and public benchmark tuning remain confounds.')
    (ROOT/'contamination.json').write_text(json.dumps(output,indent=2))
    print('COMPLETE CV',len(hits),'matching query IDs',flush=True)


if __name__=='__main__':
    main()
