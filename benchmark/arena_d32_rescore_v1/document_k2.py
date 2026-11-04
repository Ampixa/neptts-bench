"""Document the retained compact k2 audit evidence after private-copy cleanup."""
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path.home() / 'bench2/arena_d32_rescore_v1'


def main():
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    records = []
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file() or path.name in ['README.md', 'artifact_inventory.json']:
            continue
        rel = str(path.relative_to(ROOT))
        if path.suffix in ['.wav', '.zip', '.pt', '.onnx', '.mp3', '.flac']:
            raise RuntimeError('Unexpected retained audio/model')
        if rel.startswith('repo/') or path.suffix == '.py':
            role = 'Exact job source copy; final decoder source/contract pins identify scored instrument'
            origin = 'mbp:/Users/cdjk/github/neptts-bench; job source copied for execution'
        elif 'manifest' in path.name or 'cleanup' in path.name:
            role = 'ID/size/SHA inventory or guarded cleanup evidence; no audio content'
            origin = 'Generated locally on k2 in this job from consumed copies/source verification'
        elif path.suffix == '.log':
            role = 'Bounded execution/preflight/runtime/open-file audit log; final accepted runs named in EXECUTION.md'
            origin = 'k2 job CPU subprocess stdout/stderr; original author Codex /root'
        else:
            role = 'Numeric result, public snapshot or instrument binding for Arbiter review'
            origin = 'k2 job generation or same-job mbp copy; source inputs recorded in public EXECUTION.md'
        records.append(dict(path=rel, role=role, original_host='k2', original_path=str(path),
            source=origin, bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            observed_mtime_utc=datetime.datetime.fromtimestamp(path.stat().st_mtime, datetime.timezone.utc).isoformat(),
            creator='Codex /root, GPT-6 family exact serving model unknown', session='arena_d32_rescore_v1',
            owner='Ampixa / Arbiter nenglish-73', status='reference', last_review_utc=now,
            next_review='2026-11-11 or PR closure', deletion_condition='Delete transient logs/source copies within24h after Arbiter review completes; retain compact numeric/hash evidence in Git',
            continuing_value='Named current consumer: D32 profile PR provenance, calibration, results and privacy review',
            hf='NOT on HF', known_copies='Public numeric/source subset in mbp Git branch and GitHub PR after push; detailed copy/deletion manifests only k2',
            usage='Internal audit; no model/audio redistribution; original model/PV/Fresh/submitter limits remain applicable'))
    (ROOT / 'artifact_inventory.json').write_text(json.dumps(dict(created_utc=now, files=records), indent=2) + '\n')
    total = sum(r['bytes'] for r in records)
    (ROOT / 'README.md').write_text(f'''# arena_d32_rescore_v1 retained execution evidence

Created by Codex `/root` (GPT-6 family; exact serving model unknown), job `arena_d32_rescore_v1`, 2026-11-04 UTC on k2 at `{ROOT}`. Owner Ampixa / Arbiter nenglish-73. Status reference, current consumer D32 profile PR calibration/provenance/results review. Last review {now}; next review2026-11-11 or PR closure. Retain compact results and verification evidence; delete temporary logs/source copies within24h of completed review. No backups. NOT on HF.

`artifact_inventory.json` explicitly lists every retained file's role, size, SHA256, observed source time, author/session, source/original path, consumer, retention and usage. This README and that manifest document themselves (no recursive sidecars). Retained file apparent bytes excluding these two documents: {total:,}. Host free bytes: {shutil.disk_usage(ROOT).free:,}.

`result-ID.json`, `reference_numeric.json`, `calibration_gate.json`, `instrument_pins.json`, `instrument_verification.json`, `submission_retest.json`, `package_scoring_completion.json`, `approved_inventory.json` and `legacy_preservation.json` are compact numerical/binding evidence. Copies exist in the mbp repository and GitHub feature branch/PR after push. `repo/` holds actual executed source; ordinary authorship is in the final repository, inference source/contract hashed by profile ID. `calibration_final.log` is the accepted final-source196-recording rerun; earlier `calibration.log` was superseded by final source pins. `scoring_preflight_missing_pydantic.log` failed before scoring; corrected runs have complete successful205-ID outputs. Other score/retest logs and memory-pressure logs prove bounded CPU execution; no listener/service was started.

`private_audio_manifest.json` retains only member IDs, sizes and SHA256 for1025 consumed WAV members and their five ZIP containers; it contains no audio/transcripts/personal declarations. `audio_cleanup.json` and `cleanup_open_files.log` prove all five package copies were deleted after completed scoring/retests with no open inputs, every temporary extraction was empty, and zero job WAV/ZIP files remained. Original arena uploads were verified by the reviewer exporter before download; all five original report/audio hashes were rechecked unchanged after scoring. Original uploads remain on the VM and were never modified. Authorized reviewer re-download restores WAV bytes; exact deleted ZIP container serialization is not promised.

`scratch_cleanup_manifest.json` and `scratch_cleanup.json` record removal of redundant normalizer/dependency/preflight-source staging and restricted PV/Fresh panel/tasks/numeric copies. Normalizer originals were reverified against all15pins on Ubuntu before deletion. Original calibration inputs/runtime and private weights remain read-only in their original locations. No model weights were ever copied here. No submitted audio remains here or on the Mac. Dependencies can be reinstalled to this job, never to the shared runtime; source/panel prerequisites and exact commands are described by public `benchmark/arena_d32_rescore_v1/EXECUTION.md` and `neptts-eval/README.md`.

Source inputs: original calibration k2 `~/bench2/arena_judges_v1/`; original normalizer/panel Ubuntu `/mnt/wd_data/nepclone_bench_v1/phase1b`; reference output originals Ubuntu `/mnt/wd_data/swar_headtohead_v1`, verified retained hypotheses in original k2 calibration job; approved submitter source the authenticated arena reviewer package API. New source originated on mbp under `/Users/cdjk/github/neptts-bench`. This is a staged execution/audit directory, not an additional backup.

Software MIT; private call-trained D32 weights unreleased/owner-controlled, PV internal consented, Fresh MOU restricted, submitter audio private, original reference model/audio terms retained upstream. Public numerical reports do not grant weight/audio release. Never access call-data roots for this job or send submitter audio to any API. No human-rater study occurred; no rater/personal data is retained. No live arena import occurred: merge/deployment and separate import/approval remain the Arbiter's task per IMPORT.md.
''')
    print('Retained', len(records), 'documented compact artifacts;', total, 'apparent bytes; no WAV/ZIP/weights')


if __name__ == '__main__':
    main()
