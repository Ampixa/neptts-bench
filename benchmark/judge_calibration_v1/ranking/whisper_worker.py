"""Run exact arena Whisper-small decoding on verified local synthesis outputs."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import time
import traceback

ROOT = Path('/mnt/wd_data/arena_judges_v1/ranking')


def main():
    global ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--condition', choices=['fleurs', 'rasa'], default='fleurs')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--tasks', type=Path)
    parser.add_argument('--tag', default='')
    parser.add_argument('--model-cache', type=Path, default=Path('/mnt/wd_data/swar_headtohead_v1/cache/whisper'))
    args = parser.parse_args()
    ROOT=args.root
    for member in ['tmp','cache/numba']:
        (ROOT/member).mkdir(parents=True,exist_ok=True)
    os.environ['TMPDIR']=str(ROOT/'tmp')
    os.environ['NUMBA_CACHE_DIR']=str(ROOT/'cache/numba')
    ffmpeg=shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg must be on PATH before inference')
    import torch
    import whisper
    torch.set_num_threads(2)
    task_path=args.tasks or ROOT/'tasks.json'
    tasks = json.loads(task_path.read_text())
    print('SCHEMA tasks', sorted(tasks[0]), flush=True)
    # Interleave systems so partial progress does not privilege the earlier system.
    tasks = sorted([r for r in tasks if r['condition']==args.condition], key=lambda r:(r['id'],r['system']))
    selected = [r for i,r in enumerate(tasks) if i % args.shards == args.shard]
    output = ROOT/f'whisper_{args.condition}_shard{args.shard}{args.tag}.jsonl'
    old = [json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
    if any(r['status']!='ok' for r in old):
        raise RuntimeError('Previous inference failure requires explicit inspection')
    done = {r['key']:r for r in old}
    packages = {n:importlib.metadata.version(n) for n in ['openai-whisper','torch','numpy','tiktoken','numba']}
    spec = dict(reference_profile_id='whisper-small-ne-048909d9dafd',
                reference_code_sha256='9dea0cf2d556685cb4ca93e6daac794edb3fd34d875c190f863c981f46ef7bfb',
                ffmpeg_path=ffmpeg, ffmpeg_sha256=hashlib.sha256(Path(ffmpeg).read_bytes()).hexdigest(),
                weights_sha256=whisper._MODELS['small'].split('/')[-2],packages=packages,
                language='ne',task='transcribe',fp16=False,temperature=0,beam_size=5,
                condition_on_previous_text=False,threads=2,device='cpu',
                interop_threads_runtime=torch.get_num_interop_threads(),
                task_manifest_sha256=hashlib.sha256(task_path.read_bytes()).hexdigest(),
                worker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    meta = ROOT/f'whisper_{args.condition}_shard{args.shard}{args.tag}.meta.json'
    meta.write_text(json.dumps(spec,indent=2))
    model = whisper.load_model('small', device='cpu', download_root=str(args.model_cache))
    for i,row in enumerate(selected):
        if row['key'] in done:
            assert done[row['key']]['sha256']==row['sha256']
            continue
        path=Path(row['audio'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        result=dict(key=row['key'],sha256=row['sha256'],utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        start=time.perf_counter()
        try:
            decoded=model.transcribe(str(path),language='ne',task='transcribe',fp16=False,
                                     temperature=0,beam_size=5,condition_on_previous_text=False)
            result.update(status='ok',hyp=decoded.get('text','').strip())
        except Exception:
            result.update(status='failed',error=traceback.format_exc())
        result['elapsed_s']=time.perf_counter()-start
        with output.open('a') as handle:
            handle.write(json.dumps(result,ensure_ascii=False)+'\n')
        print(i+1,len(selected),row['key'],result['status'],round(result['elapsed_s'],2),flush=True)
        if result['status']!='ok':
            raise RuntimeError(result['error'])
    print('COMPLETED',len(selected),flush=True)


if __name__=='__main__':
    main()
