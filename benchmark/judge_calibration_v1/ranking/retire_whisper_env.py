"""Retire this job's unused failed Ubuntu dependency environment after open-file checks."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path('/mnt/wd_data/arena_judges_v1')
TARGET=ROOT/'whisper-env'


def main():
    assert str(TARGET)=='/mnt/wd_data/arena_judges_v1/whisper-env'
    assert TARGET.is_dir() and not TARGET.is_symlink() and TARGET.resolve().parent==ROOT.resolve()
    excluded_mounts=['/run/user/120/doc',
        '/var/lib/docker/overlay2/7717f9d5ca5cf505aac65b870784b4c1e69247bc3bae658d91c6922703b43107/merged',
        '/var/lib/docker/overlay2/a5de7a68e32658f1ada4415f0699222411c5893217b49d544058b286905d607b/merged',
        '/var/lib/docker/overlay2/dd8bb7c48cf38596829a891408fcf423fa932adb49399884de4600d9af65defa/merged',
        '/run/docker/netns/87c2189a42ba','/run/docker/netns/fca2020a7645','/run/docker/netns/8102863bfb38']
    command=['lsof','-Fn']+sum([['-e',p] for p in excluded_mounts],[])+['+D',str(TARGET)]
    probe=subprocess.run(command,text=True,capture_output=True,check=False)
    if probe.returncode not in [0,1] or probe.stdout.strip() or probe.stderr.strip():
        raise RuntimeError('Open-file check refused deletion: '+str(probe.returncode)+' '+probe.stdout+' '+probe.stderr)
    files=[p for p in TARGET.rglob('*') if p.is_file()]
    metadata=[p for p in files if p.name=='METADATA' and p.parent.name.endswith('.dist-info')]
    receipt=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target=str(TARGET),
        reason='New job-created standalone env is unused and unusable after partial Linux CUDA dependency installation; all Whisper inference runs on the completed k2 environment.',
        provenance='Created by Codex /root/rank_assets 2026-11-04; no pre-existing environment, model weights, audio or dataset inputs included.',
        open_file_probe=dict(command=command,returncode=probe.returncode,stdout=probe.stdout,stderr=probe.stderr,
            excluded_mounts=excluded_mounts,limitation='Unrelated inaccessible FUSE/Docker namespaces excluded explicitly; target is the accessible bulk job directory and has no container consumer.'),
        files=len(files),apparent_bytes=sum(p.stat().st_size for p in files),allocated_bytes=sum(p.stat().st_blocks*512 for p in files),
        package_metadata=[dict(path=str(p.relative_to(TARGET)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in metadata],
        free_bytes_before=shutil.disk_usage(ROOT).free,
        restore='Recreate Python3.12 --copies venv and precreate lib64 directory; pip --no-cache-dir torch2.14.1 numpy2.4.6 openai-whisper20250625. Linux default Torch requires large CUDA dependencies even when inference selects CPU; prefer completed k2 environment. This failed state is not preserved/restorable; no model/data loss.',
        not_on_hf=True)
    path=ROOT/'ranking/whisper_env_retirement.json'
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(dict(receipt,status='verified_unused_before_deletion'),indent=2))
    shutil.rmtree(TARGET)
    assert not TARGET.exists()
    receipt.update(status='deleted',completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   free_bytes_after=shutil.disk_usage(ROOT).free)
    receipt['free_bytes_delta']=receipt['free_bytes_after']-receipt['free_bytes_before']
    path.write_text(json.dumps(receipt,indent=2))
    print('RETIRED',receipt['files'],receipt['apparent_bytes'],receipt['allocated_bytes'],receipt['free_bytes_delta'],flush=True)


if __name__=='__main__':
    main()
