"""Exact arena decoding settings; restricted input/output stays on the CPU host."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
import whisper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--input-root', type=Path)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    args = parser.parse_args()
    tasks = json.loads(args.tasks.read_text())
    print('SCHEMA', sorted(tasks[0]), flush=True)
    torch.set_num_threads(2)
    options = dict(language='ne', task='transcribe', fp16=False, temperature=0,
                   beam_size=5, condition_on_previous_text=False)
    model = whisper.load_model('small', device='cpu', download_root=str(args.cache))
    weights = args.cache/'small.pt'
    meta = dict(packages={n: importlib.metadata.version(n) for n in ['openai-whisper', 'torch', 'numpy']},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest(),
                device='cpu', threads=2, options=options, pid=os.getpid())
    expected = {'openai-whisper': '20250625', 'torch': '2.14.1', 'numpy': '2.4.6'}
    if meta['packages'] != expected:
        raise RuntimeError('Arena package identity mismatch: '+str(meta['packages']))
    if meta['weights_sha256'] != '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794':
        raise RuntimeError('Arena small weight identity mismatch')
    meta.update(arena_profile_id='whisper-small-ne-048909d9dafd',
                arena_evaluator_code_sha256='9dea0cf2d556685cb4ca93e6daac794edb3fd34d875c190f863c981f46ef7bfb',
                reproducibility='Exact declared source/settings/packages/weights; cross-host bitwise equality is not assumed.')
    args.out.with_suffix('.meta.json').write_text(json.dumps(meta, indent=2))
    with args.out.open('x') as output:
        for i, task in enumerate(tasks):
            if i % args.shards != args.shard:
                continue
            start = time.perf_counter()
            path = Path(task['audio'])
            assert not {'eval_private', 'khalti', 'calls', 'orbit'} & set(path.parts)
            if args.input_root:
                path = args.input_root/str(path).lstrip('/')
            if hashlib.sha256(path.read_bytes()).hexdigest() != task['sha256']:
                raise RuntimeError('Input hash mismatch')
            result = model.transcribe(str(path), **options)
            output.write(json.dumps(dict(task, hyp=result['text'].strip(), status='ok',
                                         seconds=time.perf_counter()-start), ensure_ascii=False)+'\n')
            output.flush()
            print(i+1, len(tasks), flush=True)


if __name__ == '__main__':
    main()
