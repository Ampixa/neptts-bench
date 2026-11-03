"""Validate Zipformer parts, then run remaining CPU Whisper shards with exit receipts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path.home()/'bench2/arena_judges_v1/ranking'
ZIP_PIDS=[82127,82128,82129]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def alive(pid):
    try:
        os.kill(pid,0)
        return True
    except ProcessLookupError:
        return False


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def validate(rows,tasks):
    indexed={r['key']:r for r in rows}
    assert len(rows)==len(tasks) and len(indexed)==len(tasks)
    assert set(indexed)=={r['key'] for r in tasks}
    for task in tasks:
        row=indexed[task['key']]
        assert row['status']=='ok' and row['sha256']==task['sha256']


def main():
    tasks=json.loads((ROOT/'tasks.json').read_text())
    while any(alive(pid) for pid in ZIP_PIDS):
        time.sleep(15)
    zip_receipts={}
    for judge in ['long1c','d32_streaming','d32_full']:
        paths=[ROOT/f'{judge}_fleurs_part{i}.jsonl' for i in range(2)]
        rows=sum([read(path) for path in paths],[])
        validate(rows,tasks)
        out=ROOT/f'{judge}_fleurs.jsonl'
        if out.exists():
            raise FileExistsError(out)
        out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        zip_receipts[judge]=dict(n=len(rows),parts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths],
                                 output_sha256=hashlib.sha256(out.read_bytes()).hexdigest())
    print('ALL ZIPFORMERS COMPLETE',len(tasks),flush=True)
    env=dict(os.environ,PATH='/opt/homebrew/bin:/usr/local/bin:'+os.environ.get('PATH',''),PYTHONDONTWRITEBYTECODE='1')
    python=ROOT.parent/'whisper-env/bin/python'
    launched=[]
    handles=[]
    for shard in range(3):
        output=ROOT/f'whisper_fleurs_shard{shard}_part1.jsonl'
        if output.exists():
            raise FileExistsError(output)
        log=ROOT/f'whisper_fleurs_shard{shard}_part1.log'
        handle=log.open('x');handles.append(handle)
        command=[str(python),'-B',str(ROOT/'whisper_worker.py'),'--root',str(ROOT),
                 '--tasks',str(ROOT/'tasks_part1.json'),'--tag','_part1','--shard',str(shard),'--shards','3',
                 '--model-cache',str(ROOT.parent/'calibration/whisper_cache')]
        process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        launched.append((process,dict(pid=process.pid,command=command,started_utc=utc(),log=str(log))))
    (ROOT/'remaining_whisper_launches.json').write_text(json.dumps([r for p,r in launched],indent=2))
    codes=[]
    for process,receipt in launched:
        code=process.wait();codes.append(code);receipt['returncode']=code;receipt['finished_utc']=utc()
        (ROOT/'remaining_whisper_launches.json').write_text(json.dumps([r for p,r in launched],indent=2))
    for handle in handles:
        handle.close()
    if any(codes):
        raise RuntimeError('Whisper subprocess failed: '+str(codes))
    paths=sorted(ROOT.glob('whisper_fleurs_shard*_part*.jsonl'))
    while any(alive(pid) for pid in [81746,81747]):
        time.sleep(15)
    rows=sum([read(path) for path in paths],[])
    validate(rows,tasks)
    for shard in range(2):
        log=ROOT/f'whisper_fleurs_shard{shard}_part0.log'
        assert log.read_text().rstrip().endswith('COMPLETED 205')
    summary=dict(completed_utc=utc(),n=1025,zipformers=zip_receipts,whisper=dict(n=len(rows),
         returncodes_part1=codes,part0_completion='both logs end COMPLETED 205; all205 rows/status/hash per part independently validated',
         parts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths]),
         controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/'completion.json').write_text(json.dumps(summary,indent=2))
    print('ALL FIVE JUDGES COMPLETE',len(tasks),flush=True)


if __name__=='__main__':
    main()
