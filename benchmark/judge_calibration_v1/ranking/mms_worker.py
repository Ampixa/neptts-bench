"""Recompute raw MMS hypotheses with the pinned phase1b instrument on CPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path('/mnt/wd_data/arena_judges_v1/ranking')
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
    NUMBA_CACHE_DIR=str(ROOT/'cache/numba'),MPLCONFIGDIR=str(ROOT/'cache/matplotlib'),
    HF_HUB_OFFLINE='1',SUITE_SCRATCH=str(ROOT/'cache/scratch'),XDG_CACHE_HOME=str(ROOT/'cache'),
    TORCH_HOME=str(ROOT/'cache/torch'),TMPDIR=str(ROOT/'tmp'))
for member in ['cache/numba','cache/matplotlib','cache/scratch','tmp']:
    (ROOT/member).mkdir(parents=True,exist_ok=True)
sys.dont_write_bytecode=True
sys.path.insert(0,'/mnt/wd_data/tts_eval_suite_v1/code')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--condition',choices=['fleurs','rasa'],default='fleurs')
    args=parser.parse_args()
    import stages
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261104)
    tasks=json.loads((ROOT/'tasks.json').read_text())
    print('SCHEMA tasks',sorted(tasks[0]),flush=True)
    tasks=sorted([r for r in tasks if r['condition']==args.condition],key=lambda r:(r['id'],r['system']))
    path=ROOT/f'mms_{args.condition}.jsonl'
    existing=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    if any(r['status']!='ok' for r in existing):
        raise RuntimeError('Prior failures need explicit inspection')
    done={r['key']:r for r in existing}
    run,extra=stages.build_mms('cpu')
    meta=dict(torch=torch.__version__,extra=extra,pins=stages.PINS['mms'],threads=4,device='cpu',
              stages_sha256=hashlib.sha256(Path(stages.__file__).read_bytes()).hexdigest(),
              worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (ROOT/f'mms_{args.condition}.meta.json').write_text(json.dumps(meta,indent=2))
    for i,task in enumerate(tasks):
        if task['key'] in done:
            assert done[task['key']]['sha256']==task['sha256']
            continue
        start=time.perf_counter()
        result=dict(key=task['key'],sha256=task['sha256'],utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        try:
            data=Path(task['audio']).read_bytes()
            assert hashlib.sha256(data).hexdigest()==task['sha256']
            result.update(run(data))
        except Exception:
            result.update(status='failed',error=traceback.format_exc())
        result['elapsed_s']=time.perf_counter()-start
        with path.open('a') as handle:
            handle.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(i+1,len(tasks),task['key'],result['status'],round(result['elapsed_s'],2),flush=True)
        if result['status']!='ok':
            raise RuntimeError(result.get('error',result['status']))
    print('COMPLETED',len(tasks),flush=True)

if __name__=='__main__':
    main()
