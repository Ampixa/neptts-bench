"""Summarize recorded selection/retry evidence without copying hypotheses or audio."""
import collections
import datetime
import hashlib
import json
from pathlib import Path

ROOT=Path('/mnt/wd_data/arena_judges_v1/ranking')
SOURCE=Path('/mnt/wd_data/swar_headtohead_v1/runs/primary')


def main():
    output=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),systems={})
    for system in ['swar','vox2_beat1','vox2_beat2','cosy3_elx_neutral','vox2_clone']:
        directory=SOURCE/system
        model_path=directory/'model.json';items_path=directory/'items.jsonl'
        model=json.loads(model_path.read_text())
        items=[json.loads(s) for s in items_path.read_text().splitlines()]
        print('SCHEMA',system,'model',sorted(model),'item',sorted(items[0]),flush=True)
        primary=[r for r in items if r['condition']=='fleurs']
        assert len(primary)==205 and all(r['status']=='ok' for r in primary)
        result=dict(n_prompts=205,model_receipt_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    items_receipt_sha256=hashlib.sha256(items_path.read_bytes()).hexdigest(),
                    model=model.get('model',model.get('base_model')),model_revision=model.get('model_revision',model.get('base_revision')),
                    settings=model['settings'],n_selection_candidates_per_prompt=model.get('n'),
                    selector=model.get('frozen',{}).get('selection','not recorded as a selector'),
                    lora_weights_sha256=(model.get('lora') or {}).get('weights_sha256'),
                    retry_counts=dict(collections.Counter(str(r.get('badcase_retries','unrecorded')) for r in primary)))
        if system.startswith('vox2_beat'):
            candidates=[c for row in primary for c in row['candidates']]
            print('SCHEMA candidate',sorted(candidates[0]),flush=True)
            result.update(recorded_selection_candidates=len(candidates),
                candidates_with_changed_seed_used=sum(c['seed_used']!=c['seed'] for c in candidates),
                seed_used_delta_counts=dict(collections.Counter(str(c['seed_used']-c['seed']) for c in candidates)),
                total_generation_attempts='not fully reconstructed; eight recorded selection candidates do not include all internal retries',
                checkpoint_frozen_metadata=model.get('frozen'))
        output['systems'][system]=result
    output['limitations']='These are different saved model/checkpoint/generation conditions, not a causal selection ablation. Recorded retry/seed fields are reported as observed; missing fields are not treated as proof of no retries or no judge-family training.'
    (ROOT/'generation_contract_audit.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
