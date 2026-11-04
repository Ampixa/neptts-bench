"""Inventory retained public artifacts with sizes, hashes, provenance and review use."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    files = []
    for path in sorted(ROOT.iterdir()):
        if not path.is_file() or path.name in ['README.md', 'artifact_inventory.json']:
            continue
        role = ('Version-controlled execution/analysis/documentation source' if path.suffix in ['.py', '.md'] else
                'Sanitized numeric result, binding, aggregate or verification/deletion receipt')
        origin = 'mbp:/Users/cdjk/github/neptts-bench/benchmark/arena_d32_rescore_v1'
        if path.name.startswith('result-') or path.name in [
            'calibration_gate.json', 'instrument_verification.json', 'reference_numeric.json',
            'submission_retest.json', 'package_scoring_completion.json', 'audio_cleanup.json', 'scratch_cleanup.json']:
            origin = 'k2:~/bench2/arena_d32_rescore_v1/' + path.name
        if path.name == 'approved_inventory.json':
            origin = 'Arena VM authenticated reviewer inventory; only public reports/IDs copied to mbp'
        if path.name == 'instrument_pins.json':
            origin = 'Committed benchmark/judge_calibration_v1/calibration/numbers.json instruments.d32_full'
        creator = 'Codex /root/reference_context' if path.name == 'REFERENCE_AUDIT.md' else 'Codex /root'
        files.append(dict(path=path.name, role=role, bytes=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), creator=creator,
            model='GPT-6 family, exact serving ID unknown', job='arena_d32_rescore_v1',
            creation_utc='2026-11-04 UTC; precise execution timestamps in result/gate/deletion receipts where present',
            observed_mtime_utc=datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat(),
            original_host_path=origin, why='D32 profile calibration, results, import and privacy evidence for Arbiter review',
            continuing_value='Named current consumer: feature-branch PR review and post-merge import; numeric source of reproducible report',
            status='reference', owner='Ampixa / Arbiter nenglish-73', last_review_utc=now,
            next_review='2026-11-11 or PR closure',
            deletion_condition='Keep compact published source/metrics in Git; remove obsolete local execution copies after review with original Git/hash verification',
            known_duplicates='mbp Git checkout; GitHub feature branch and PR after push; k2 has source/numeric execution subset documented separately',
            hf='NOT on HF', licence_usage='MIT source; numeric evidence only. No model/audio licence granted. Private model and PV/Fresh/submitter inputs remain restricted.',
            reproduction='EXECUTION.md; analyze.py then report.py regenerate aggregate/report from numeric source; inference requires authorized private prerequisites'))
    (ROOT / 'artifact_inventory.json').write_text(json.dumps(dict(created_utc=now, files=files), indent=2) + '\n')
    print('Inventoried', len(files), 'public artifacts')


if __name__ == '__main__':
    main()
