"""Bounded, job-local CPU controller; no listeners or shared service mutations."""
import concurrent.futures
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path.home()/'bench2/arena_judges_v1'
CAL = ROOT/'calibration'


def run(command, log):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               PATH='/opt/homebrew/bin:/usr/local/bin:'+os.environ.get('PATH', ''),
               NUMBA_CACHE_DIR=str(CAL/'cache/numba'), XDG_CACHE_HOME=str(CAL/'cache'))
    with log.open('x') as output:
        subprocess.run([str(x) for x in command], stdout=output, stderr=subprocess.STDOUT,
                       env=env, check=True, cwd=CAL)


def zipformer(name, mode, export=None):
    base = [ROOT/'venv/bin/python', '-B', CAL/'run_zipformer.py', '--mode', mode,
            '--threads', '1', '--input-root', CAL/'input']
    if export:
        base += ['--export', export]
    else:
        base += ['--checkpoint', '/Users/k2/cs_work/asr_a2_run/d32v2/d32v2_step60000-export.pt',
                 '--bpe', '/Users/k2/cs_work/tokenizer_v4_4k/bpe.model',
                 '--recipe', '/Users/k2/cs_work/ship_onnx/code/track8-cs-asr/train/zipformer_cs',
                 '--icefall', ROOT/'icefall']
    for suffix, tasks in [('', 'zip_tasks.json'), ('_retest2', 'zip_retest2.json')]:
        run(base+['--tasks', CAL/tasks, '--out', CAL/(name+suffix+'.jsonl')], CAL/(name+suffix+'.log'))


def small(shard):
    base = [ROOT/'whisper-env/bin/python', '-B', CAL/'run_whisper.py',
            '--cache', CAL/'whisper_cache', '--input-root', CAL/'input']
    run(base+['--tasks', CAL/'whisper_tasks.json', '--out', CAL/f'asr_small_shard{shard}.jsonl',
              '--shard', str(shard), '--shards', '3'], CAL/f'asr_small_shard{shard}.log')
    if shard == 0:
        run(base+['--tasks', CAL/'whisper_retest2.json', '--out', CAL/'asr_small_retest2.jsonl'],
            CAL/'asr_small_retest2.log')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--small-only', action='store_true')
    args = parser.parse_args()
    status_path = CAL/('small_run_status.json' if args.small_only else 'run_status.json')
    tasks = json.loads((CAL/'zip_tasks.json').read_text())
    deadline = time.monotonic()+3600
    # Sender writes this marker only after checking both transfer exit codes.
    while not (CAL/'packet_complete.json').exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Audio packet did not complete within one hour')
        time.sleep(10)
    for task in tasks:
        path = CAL/'input'/task['audio'].lstrip('/')
        if hashlib.sha256(path.read_bytes()).hexdigest() != task['sha256']:
            raise RuntimeError('Packet hash mismatch')
    start = datetime.datetime.now(datetime.timezone.utc).isoformat()
    state = dict(started_utc=start, packet_verified=True, status='running', completed=[])
    status_path.write_text(json.dumps(state, indent=2))
    jobs = [('d32_streaming', lambda: zipformer('d32_streaming', 'online',
             '/Volumes/T9/asr_a2/exports/d32v2_step60000_c16')),
            ('d32_full', lambda: zipformer('d32_full', 'offline'))]
    jobs += [(f'small_shard{i}', lambda i=i: small(i)) for i in range(3)]
    if args.small_only:
        jobs = jobs[2:]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(job): name for name, job in jobs}
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as error:
                state.update(status='failed', failed_job=futures[future], error=repr(error))
                status_path.write_text(json.dumps(state, indent=2))
                raise
            state['completed'].append(futures[future])
            status_path.write_text(json.dumps(state, indent=2))
    state.update(status='complete', completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    status_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__':
    main()
