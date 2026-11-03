"""Launch and verify five arena-setting CPU Whisper shards; retain exit receipts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path.home()/'bench2/arena_judges_v1/ranking'


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    env=dict(os.environ,PATH='/opt/homebrew/bin:/usr/local/bin:'+os.environ.get('PATH',''),PYTHONDONTWRITEBYTECODE='1')
    python=ROOT.parent/'whisper-env/bin/python'
    source=ROOT/'whisper_worker.py'
    text=source.read_text()
    assert 'set_num_interop_threads' not in text and 'manual_seed' not in text
    tasks=json.loads((ROOT/'tasks.json').read_text())
    assert len(tasks)==1025
    handles=[];launched=[]
    for shard in range(5):
        output=ROOT/f'whisper_fleurs_shard{shard}_exact.jsonl'
        if output.exists():
            raise FileExistsError(output)
        log=ROOT/f'whisper_fleurs_shard{shard}_exact.log'
        handle=log.open('x');handles.append(handle)
        command=[str(python),'-B',str(source),'--root',str(ROOT),'--tag','_exact',
                 '--shard',str(shard),'--shards','5','--model-cache',str(ROOT.parent/'calibration/whisper_cache')]
        process=subprocess.Popen(command,stdout=handle,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
        launched.append((process,dict(pid=process.pid,command=command,started_utc=utc(),log=str(log))))
    receipt=ROOT/'exact_whisper_launches.json'
    receipt.write_text(json.dumps([r for p,r in launched],indent=2))
    codes=[]
    for process,row in launched:
        code=process.wait();codes.append(code);row.update(returncode=code,finished_utc=utc())
        receipt.write_text(json.dumps([r for p,r in launched],indent=2))
    for handle in handles:
        handle.close()
    if any(codes):
        raise RuntimeError('Whisper subprocess failure: '+str(codes))
    paths=sorted(ROOT.glob('whisper_fleurs_shard*_exact.jsonl'))
    rows=sum([[json.loads(s) for s in path.read_text().splitlines()] for path in paths],[])
    indexed={r['key']:r for r in rows}
    assert len(rows)==len(tasks) and len(indexed)==len(tasks)
    assert set(indexed)=={r['key'] for r in tasks}
    for task in tasks:
        row=indexed[task['key']]
        assert row['status']=='ok' and row['sha256']==task['sha256']
    out=dict(completed_utc=utc(),n=1025,returncodes=codes,
             worker_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
             controller_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             parts=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths])
    (ROOT/'whisper_completion.json').write_text(json.dumps(out,indent=2))
    print('EXACT WHISPER COMPLETE',len(rows),flush=True)


if __name__=='__main__':
    main()
