"""Aggregate restricted calibration observations without exporting transcripts.

Run on ubuntu with the existing phase1b environment. All detailed scoring stays
under the remote job root; only numbers.json and calibration_public.json may be
copied into the public branch after review. Missing observations are fatal.
"""

import argparse
from collections import Counter
import datetime
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

from arena_metrics import _edit_distance
from arena_normalize import normalize_nepali_text

OLD_DEFAULT = Path('/mnt/wd_data/nepclone_bench_v1/phase1b')
ROOT_DEFAULT = Path('/mnt/wd_data/arena_judges_v1/calibration')
PINS = {
    'analyze.py': 'bf3822c5d5cd5fefcdbecdfd97dd956531d9a11358df743f45577f4a34aee05f',
    'panel.json': 'cde2d472e3f026ebcc8ac85a64152ceb5cef72a14724fc591d533b21e8034d30',
    'audio_tasks.json': 'cd34b4a1e74049f6ea6fdf1a72fd0bc834620d8ae5317a8e76055ef667741616',
    'scored.jsonl': '7b2aeda6f08b06ab14b24bc27626a98734e61a968cc331855bc62fb4a7a6c8b9',
    'numbers.json': '7edaf82328596bfeb6c9699d66118c1c89d217c3e429996323a1c708ca661181',
    'shuffle.json': '9904d06fb3a33b389ce2331133b92e71f89a08cf9412ddebc4f72a42213b49c3',
    'normalization_manifest.json': '67d30baf75129d68eac255a904202081b4cdea5778206171ad9cd1a7ecac57f5',
}
OLD_STAGES = ['asr_long1c', 'asr_mms', 'asr_whisper']
NEW_FILES = {
    'asr_small': [f'asr_small_shard{i}.jsonl' for i in range(3)],
    'd32_streaming': ['d32_streaming.jsonl'],
    'd32_full': ['d32_full.jsonl'],
}
RETEST_FILES = {
    'asr_small': 'asr_small_retest2.jsonl',
    'd32_streaming': 'd32_streaming_retest2.jsonl',
    'd32_full': 'd32_full_retest2.jsonl',
}
CONDITIONS = ['real', 'degraded', 'shuffled_text']
POLICIES = ['raw', 'translit', 'arena_raw']
PROFILES = {
    'asr_small': ('whisper-small-ne-048909d9dafd', 'Whisper small · arena legacy', 'arena_raw'),
    'asr_long1c': ('long1c-phase1b-legacy-v1', 'LONG1C · phase1b legacy', 'translit'),
    'asr_mms': ('mms-1b-all-npi-phase1b-legacy-v1', 'MMS-1b-all npi · phase1b legacy', 'translit'),
    'asr_whisper': ('whisper-v3-turbo-phase1b-legacy-v1', 'Whisper v3 turbo · phase1b legacy', 'translit'),
    'd32_streaming': ('d32-v2-step60000-streaming320-legacy-v1', 'D32 v2 step 60000 · streaming 320 ms', 'translit'),
    'd32_full': ('d32-v2-step60000-fullcontext-legacy-v1', 'D32 v2 step 60000 · full context', 'translit'),
}
ARENA_PINS = {
    'arena_normalize.py': 'b7198286f201131922cc5bf1183d1701f6bb85574fc1c55ae3f44a9f6b752630',
    'arena_metrics.py': 'e117eb03ba6814709d3fea2189064e285bd1378ea55a9b0c03495ba8214c4e0f',
}
D32_TOKEN_SHA = '424bddc0b38d961991b7717b9c5424918cd2d4c58f2810db182675b1a49d2105'
D32_STREAMING_PINS = {
    'encoder.int8.onnx': '3a8585ac92b29f77664221fcfed5e8be8aff60566f7eca09ee5940a6cf9b957f',
    'decoder.int8.onnx': 'f46ed86e4268d2fd9198e59d6b6570cd9ffe045d3ab7288e2b535977e36b90f7',
    'joiner.int8.onnx': '691b1201f85fec06a5ca4c6ec8a280d705f61470d60e406cbbc9a02b48e826e4',
}
D32_STREAMING_SETTINGS = {
    'chunk_size': 16,
    'left_context_frames': 128,
    'n_params': 33447414,
    'checkpoint_sha256': '97392051e0fd5aa764b0098c7a39ebe2223e521c76701bfffdf5b99f1f39b213',
    'bpe_sha256': 'efef2f02304881a9a5aa269aec829b9ee1ba1ada0ff48f1f2d3feae7fb51394f',
}
SMALL_PROFILE = {
    'benchmark': 'neptts-synthesis-v1-e85f91670812', 'model': 'small',
    'weights_sha256': '9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794',
    'code_sha256': '9dea0cf2d556685cb4ca93e6daac794edb3fd34d875c190f863c981f46ef7bfb',
    'packages': {'openai-whisper': '20250625', 'torch': '2.14.1', 'numpy': '2.4.6'},
    'language': 'ne', 'device': 'cpu', 'temperature': 0, 'beam_size': 5,
    'aggregation': 'macro-average sentence CER/WER', 'development_only': False,
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        value = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        value = json.loads(path.read_text())
    keys = sorted(value[0] if isinstance(value, list) and value else value)
    print('SCHEMA', path.name, keys, flush=True)
    return value


def load_frozen(old):
    for name, digest in PINS.items():
        require(sha(old/name) == digest, 'Frozen phase1b hash mismatch: '+name)
    manifest = read(old/'normalization_manifest.json')
    for item in manifest['files']:
        require(sha(item['original_path']) == item['sha256'],
                'Normalizer source drift: '+item['original_path'])
        require(sha(old/item['copy_path']) == item['sha256'],
                'Normalizer snapshot drift: '+item['copy_path'])
    spec = importlib.util.spec_from_file_location('frozen_phase1b_analysis', old/'analyze.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    # The frozen module's imports intentionally resolve the pinned source inputs.
    require(frozen.SEED == 20261103 and frozen.B == 10000, 'Bootstrap method drift')
    return frozen, manifest


def arena_counts(ref, hyp):
    ref, hyp = normalize_nepali_text(ref), normalize_nepali_text(hyp)
    rc, hc = list(ref.replace(' ', '')), list(hyp.replace(' ', ''))
    rw, hw = ref.split(), hyp.split()
    require(bool(rc) and bool(rw), 'Arena-normalized reference is empty')
    ce, we = _edit_distance(rc, hc), _edit_distance(rw, hw)
    return dict(char_errors=ce, chars=len(rc), word_errors=we, words=len(rw),
                cer=ce/len(rc), wer=we/len(rw))


def numeric_row(row):
    # Detailed IDs/voice clusters stay on the restricted remote host, never catalog.
    keys = ['id', 'voice', 'cohort', 'condition', 'stage', 'policy',
            'char_errors', 'chars', 'word_errors', 'words', 'cer', 'wer']
    return {key: row[key] for key in keys}


def check_counts(row):
    for key in ['char_errors', 'chars', 'word_errors', 'words', 'cer', 'wer']:
        require(isinstance(row[key], (int, float)) and math.isfinite(row[key]) and row[key] >= 0,
                'Nonfinite/negative scoring value')
    require(row['chars'] > 0 and row['words'] > 0, 'Empty score denominator')
    require(math.isclose(row['cer'], row['char_errors']/row['chars'], abs_tol=1e-12),
            'Character score ratio mismatch')
    require(math.isclose(row['wer'], row['word_errors']/row['words'], abs_tol=1e-12),
            'Word score ratio mismatch')


def compare_numeric(got, expected, label):
    if isinstance(expected, dict):
        for key, value in expected.items():
            if key in got:
                compare_numeric(got[key], value, label+'/'+key)
    elif isinstance(expected, list):
        require(len(got) == len(expected), 'Reused shape mismatch: '+label)
        for i, value in enumerate(expected):
            compare_numeric(got[i], value, label+'/'+str(i))
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        require(math.isclose(got, expected, rel_tol=1e-11, abs_tol=1e-9),
                'Reused aggregate mismatch: '+label)


def load_new(root, stage, expected, retest_expected):
    paths = [root/name for name in NEW_FILES[stage]]
    rows = sum((read(path) for path in paths), [])
    second_path = root/RETEST_FILES[stage]
    second = read(second_path)
    primary, repeats = {}, []
    for row in rows:
        require(row['status'] == 'ok', 'Failed inference observation: '+stage)
        key = row['id'], row['role']
        require(key in expected, 'Unexpected primary/repeat identity: '+stage)
        task = expected[key]
        for field in ['voice', 'cohort', 'sha256']:
            require(row[field] == task[field], 'Observation input metadata mismatch: '+stage)
        require(isinstance(row['hyp'], str), 'Hypothesis type mismatch: '+stage)
        if row['repeat']:
            repeats.append(row)
        else:
            require(key not in primary, 'Duplicate primary observation: '+stage)
            primary[key] = row
    require(set(primary) == set(expected) and len(primary) == 392,
            'Incomplete primary panel: '+stage)
    wanted = set(retest_expected)
    for pass_id, repeated in [(1, repeats), (2, second)]:
        require(len(repeated) == 32, 'Incomplete retest: '+stage)
        got = [(r['id'], r['role']) for r in repeated]
        require(set(got) == wanted and len(set(got)) == len(got),
                'Retest coverage mismatch: '+stage)
        for row in repeated:
            key = row['id'], row['role']
            require(row['status'] == 'ok' and row['repeat'], 'Invalid repeat: '+stage)
            require(row.get('pass_id', 1) == pass_id, 'Retest pass mismatch: '+stage)
            require(row['sha256'] == primary[key]['sha256'], 'Retest input hash mismatch: '+stage)
            require(row['voice'] == primary[key]['voice'] and row['cohort'] == primary[key]['cohort'],
                    'Retest identity mismatch: '+stage)
    first_repeat = {(r['id'], r['role']): r for r in repeats}
    info = dict(passes=[dict(pass_id=i, n=32,
                            changed_hypotheses=sum(r['hyp'] != primary[(r['id'], r['role'])]['hyp'] for r in rr))
                       for i, rr in [(1, repeats), (2, second)]],
                cold_to_cold_changed=sum(r['hyp'] != first_repeat[(r['id'], r['role'])]['hyp'] for r in second),
                primary_observations=len(primary), observations=len(rows)+len(second),
                unique_retest_inputs=32, unique_primary_inputs=392,
                retest_fraction=32/392, seed_spread='No TTS generation/seed experiment')
    return primary, info, paths+[second_path]


def instrument_metadata(stage, observation_paths):
    """Expose instrument hashes/settings while excluding all model/data paths."""
    allowed = (
        ['packages', 'source_sha256', 'weights_sha256', 'device', 'threads', 'options']
        if stage == 'asr_small' else
        ['mode', 'threads', 'provider', 'worker_sha256', 'sherpa_onnx', 'hashes', 'padding_samples']
        if stage == 'd32_streaming' else
        ['mode', 'threads', 'provider', 'worker_sha256', 'checkpoint_sha256', 'bpe_sha256',
         'chunk_size', 'left_context_frames', 'parameters', 'features', 'recipe_hashes', 'torch', 'lhotse']
    )
    paths = [path.with_suffix('.meta.json') for path in observation_paths]
    summaries = []
    for path in paths:
        value = read(path)
        require(all(key in value for key in allowed), 'Incomplete instrument metadata: '+path.name)
        summary = {key: value[key] for key in allowed}
        if stage == 'd32_streaming':
            require(isinstance(value.get('export_metadata'), dict),
                    'Missing D32 streaming export metadata: '+path.name)
            export = value['export_metadata']
            # Retain only verified settings/pins, never the checkpoint/tokenizer paths.
            settings = {key: export.get(key) for key in D32_STREAMING_SETTINGS}
            require(settings == D32_STREAMING_SETTINGS,
                    'D32 streaming chunk/context/checkpoint/tokenizer/parameter mismatch')
            require(isinstance(export.get('files'), dict), 'Missing D32 streaming export file hashes')
            for name, digest in D32_STREAMING_PINS.items():
                require(value['hashes'].get(name) == digest,
                        'D32 streaming actual model hash mismatch: '+name)
                require(export['files'].get(name, {}).get('sha256') == digest,
                        'D32 streaming declared export model hash mismatch: '+name)
            require(value['hashes'].get('tokens.txt') == D32_TOKEN_SHA,
                    'D32 streaming tokens.txt identity mismatch')
            require(value['mode'] == 'online' and value['provider'] == 'cpu',
                    'D32 streaming mode/provider mismatch')
            summary['export_settings'] = settings
        summaries.append(summary)
    require(all(value == summaries[0] for value in summaries),
            'Instrument drift between primary/cold-retest processes: '+stage)
    value = summaries[0]
    if stage == 'asr_small':
        require(value['weights_sha256'] == SMALL_PROFILE['weights_sha256'], 'Whisper-small weights drift')
        require(value['packages'] == SMALL_PROFILE['packages'],
                'Runtime does not reproduce whisper-small-ne-048909d9dafd; use a separate profile')
        require(value['device'] == 'cpu' and value['threads'] == 2, 'Whisper execution settings drift')
        require(value['options'] == dict(language='ne', task='transcribe', fp16=False,
                                        temperature=0, beam_size=5, condition_on_previous_text=False),
                'Whisper decoding options drift')
        digest = hashlib.sha256(json.dumps(SMALL_PROFILE, sort_keys=True).encode()).hexdigest()[:12]
        require('whisper-small-ne-'+digest == PROFILES['asr_small'][0], 'Historical profile identity mismatch')
        value['historical_profile'] = SMALL_PROFILE
    elif stage == 'd32_full':
        require(value['checkpoint_sha256'] == '97392051e0fd5aa764b0098c7a39ebe2223e521c76701bfffdf5b99f1f39b213',
                'D32 full-context checkpoint mismatch')
        require(value['bpe_sha256'] == 'efef2f02304881a9a5aa269aec829b9ee1ba1ada0ff48f1f2d3feae7fb51394f',
                'D32 full-context tokenizer mismatch')
        require(value['chunk_size'] == value['left_context_frames'] == -1,
                'D32 full-context setting mismatch')
    return value, paths


def verify_reuse_only(old, frozen):
    """Verify completed historical evidence while new inference is still running."""
    panel = read(old/'panel.json')
    rows = read(old/'scored.jsonl')
    numbers = read(old/'numbers.json')
    by_id = {p['id']: p for p in panel}
    shuffled = {r['id']: by_id[r['reference_id']] for r in read(old/'shuffle.json')}
    checked = 0
    for row in rows:
        if row['stage'] not in OLD_STAGES:
            continue
        check_counts(row)
        ref = (shuffled[row['id']]['text'] if row['condition'] == 'shuffled_text'
               else by_id[row['id']]['text'])
        hyp = (frozen.normalized(ref, row['hyp_original']) if row['policy'] == 'translit'
               else row['hyp_original'])
        score = frozen.cer_wer(ref, hyp)
        compare_numeric(score, {key: row[key] for key in
                                ['char_errors', 'chars', 'word_errors', 'words', 'cer', 'wer']},
                        'Reused numerical score preflight')
        checked += 1
    require(checked == len(OLD_STAGES)*196*3*2, 'Reused preflight coverage mismatch')
    for policy in ['raw', 'translit']:
        for stage in OLD_STAGES:
            for cohort in ['pv', 'fresh', 'all']:
                rr = [r for r in rows if r['policy'] == policy and r['stage'] == stage
                      and (cohort == 'all' or r['cohort'] == cohort)]
                conditions = {c: [r for r in rr if r['condition'] == c] for c in CONDITIONS}
                out = {c: frozen.aggregate(conditions[c]) for c in CONDITIONS}
                for condition in CONDITIONS[1:]:
                    out[condition]['paired'] = frozen.paired(conditions['real'], conditions[condition])
                compare_numeric(out, numbers['asr'][policy][stage][cohort], 'Reused aggregation preflight')
    print('PREFLIGHT_COMPLETE', checked,
          'recomputed numerical rows; all historical aggregates/paired intervals match; no output written',
          flush=True)


def make_catalog(numbers):
    profiles = []
    for stage, (profile_id, label, policy) in PROFILES.items():
        d32 = stage.startswith('d32_')
        calibration = {cohort: numbers['asr'][policy][stage][cohort]['real']
                       for cohort in ['pv', 'fresh']}
        profiles.append(dict(
            id=profile_id, name=label, stage=stage, primary_policy=policy,
            calibration_policy=('Historical arena normalization: NFC, punctuation removal, digits to ASCII, '
                                'lowercase, CER excludes spaces; no loanword transliteration'
                                if policy == 'arena_raw' else
                                'Frozen phase1b legacy score2.py conditional Latin-hypothesis transliteration; '
                                'suite normalization and CER including spaces'),
            loanword_policy=('Preserve script; lowercase Latin. Latin tokens are not transliterated.'
                             if policy == 'arena_raw' else
                             'Transliterate hypothesis Latin tokens only when the reference contains no ASCII Latin, '
                             'using frozen legacy score2.py/Swar frontend; otherwise preserve hypothesis.'),
            status=('uncalibrated for spontaneous Nepali' if stage == 'asr_small' else
                    'local calibration only; public profile depends on owner decision' if d32 else
                    'diagnostic calibration; independent qualification incomplete'),
            public_enabled=stage == 'asr_small', provisional=stage != 'asr_small',
            availability=('Public judge depends on the owner decision about the call-trained 32M model.'
                          if d32 else 'Current arena worker/import profile.' if stage == 'asr_small' else
                          'Catalog calibration only; executable public worker/import support pending.'),
            calibration=calibration,
            controls={cohort: {condition: numbers['asr'][policy][stage][cohort][condition]
                              for condition in ['degraded', 'shuffled_text']}
                      for cohort in ['pv', 'fresh']},
            retest=numbers['retest'][stage], evidence='calibration/numbers.json',
            limitations=['Fixed source sessions/texts; four PV speakers and twelve Fresh voices.',
                         'Fresh silver transcripts and historical ASR/speaker admission create selection bias.',
                         'Finite text contamination screen does not prove unseen training/speakers.',
                         'ASR CER is an intelligibility proxy; no new human MOS/transcription study.',
                         '32/392 cold retest inputs fall below prospective 10% sampling gate.']
        ))
    return dict(schema_version=1, job='arena_judges_v1', measured_utc=numbers['analysis_utc'],
                calibration_sets={'pv': {'n': 100, 'speakers': 4}, 'fresh': {'n': 96, 'speakers': 12}},
                bootstrap={'draws': numbers['bootstraps'], 'seed': numbers['seed'],
                           'unit': 'speaker within cohort; fixed text/session panel'},
                profiles=profiles)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT_DEFAULT)
    parser.add_argument('--phase1b', type=Path, default=OLD_DEFAULT)
    parser.add_argument('--preflight', action='store_true',
                        help='Recompute historical raw/translit metrics and CIs; write no result files.')
    args = parser.parse_args()
    root, old = args.root, args.phase1b
    require(root.resolve() != old.resolve(), 'Refuse to write into historical phase1b')
    frozen, normalization_manifest = load_frozen(old)
    for name, digest in ARENA_PINS.items():
        require(sha(Path(__file__).with_name(name)) == digest, 'Arena scoring source mismatch: '+name)
    if args.preflight:
        verify_reuse_only(old, frozen)
        return
    for name in ['numbers.json', 'calibration_public.json', 'scored_internal.jsonl']:
        require(not (root/name).exists(), 'Refuse to overwrite completed analysis: '+name)
    panel = read(old/'panel.json')
    audio = read(old/'audio_tasks.json')
    old_numbers = read(old/'numbers.json')
    old_scored = read(old/'scored.jsonl')
    shuffle_rows = read(old/'shuffle.json')
    tasks = read(root/'tasks.json')
    retest = read(root/'retest.json')
    require(len(panel) == 196 and Counter(p['cohort'] for p in panel) == {'pv': 100, 'fresh': 96},
            'Historical panel count mismatch')
    require(len({p['id'] for p in panel}) == 196, 'Duplicate panel identity')
    hashes = {a['audio']: a['sha256'] for a in audio}
    by_id = {p['id']: p for p in panel}
    shuffled = {r['id']: by_id[r['reference_id']] for r in shuffle_rows}
    require(set(shuffled) == set(by_id), 'Incomplete frozen text derangement')
    expected = {(r['id'], r['role']): r for r in tasks}
    require(len(tasks) == len(expected) == 392, 'Task panel cardinality mismatch')
    for p in panel:
        for role in ['real', 'degraded']:
            row = expected[(p['id'], role)]
            require(row['sha256'] == hashes[p[role+'_audio']], 'New task does not match historical input')
            require(row['voice'] == p['voice'] and row['cohort'] == p['cohort'], 'Task identity metadata mismatch')
            path = Path(row['audio'])
            require(not {'eval_private', 'khalti', 'calls', 'orbit'} & set(path.parts), 'Forbidden input path')
            require(sha(path) == row['sha256'], 'Current audio bytes drift from frozen panel')
    retest_expected = {(r['id'], r['role']): r for r in retest}
    require(len(retest) == len(retest_expected) == 32, 'Retest manifest cardinality mismatch')
    require(Counter((r['voice'], r['role']) for r in retest).most_common(1)[0][1] == 1,
            'Retest must cover one clean and one degraded input per voice')
    new, retests, source_paths, instruments = {}, {}, [], {}
    for stage in NEW_FILES:
        new[stage], retests[stage], paths = load_new(root, stage, expected, retest_expected)
        source_paths.extend(paths)
        instruments[stage], metadata_paths = instrument_metadata(stage, paths)
        source_paths.extend(metadata_paths)
    reused = {(r['stage'], r['id'], r['condition'], r['policy']): r
              for r in old_scored if r['stage'] in OLD_STAGES}
    require(len(reused) == len(OLD_STAGES)*196*3*2, 'Reused scored panel cardinality mismatch')
    scored = []
    stages = OLD_STAGES+list(NEW_FILES)
    for stage in stages:
        if stage in OLD_STAGES:
            t = old_numbers['retest'][stage]
            retests[stage] = dict(passes=[{k: v for k, v in p.items() if k in ['pass_id', 'n', 'changed_hypotheses']}
                                         for p in t['passes']],
                                 cold_to_cold_changed=t['cold_to_cold_changed'],
                                 primary_observations=392, observations=t['observations'],
                                 reused_primary_observations=t['reused'],
                                 new_primary_observations=t['new'],
                                 unique_retest_inputs=32, unique_primary_inputs=392,
                                 retest_fraction=32/392, reused_phase1b=True,
                                 seed_spread='No TTS generation/seed experiment')
        for p in panel:
            for condition in CONDITIONS:
                role = 'degraded' if condition == 'degraded' else 'real'
                ref = shuffled[p['id']]['text'] if condition == 'shuffled_text' else p['text']
                if stage in OLD_STAGES:
                    old_row = reused[(stage, p['id'], condition, 'raw')]
                    require(old_row['source_sha256'] == hashes[p[role+'_audio']], 'Reused audio identity mismatch')
                    hyp = old_row['hyp_original']
                else:
                    hyp = new[stage][(p['id'], role)]['hyp']
                for policy in POLICIES:
                    if policy == 'arena_raw':
                        values = arena_counts(ref, hyp)
                    else:
                        scored_hyp = frozen.normalized(ref, hyp) if policy == 'translit' else hyp
                        values = frozen.cer_wer(ref, scored_hyp)
                    row = dict(id=p['id'], voice=p['voice'], cohort=p['cohort'],
                               condition=condition, stage=stage, policy=policy, **values)
                    check_counts(row)
                    if stage in OLD_STAGES and policy != 'arena_raw':
                        previous = reused[(stage, p['id'], condition, policy)]
                        compare_numeric(row, {k: previous[k] for k in ['char_errors', 'chars', 'word_errors', 'words', 'cer', 'wer']},
                                        'Recomputed reused score')
                    scored.append(numeric_row(row))
    numbers = dict(schema_version=1, job='arena_judges_v1', targets=196,
                   analysis_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   creator='Codex /root/docs_reconcile; GPT-6 family, exact serving model unknown',
                   owner='Ampixa / Arbiter nenglish-73', status='reference', next_review='2026-11-11',
                   hf='NOT on HF; no audio, transcripts or weights in public aggregates',
                   seed=frozen.SEED, bootstraps=frozen.B, numpy=frozen.np.__version__,
                   asr={}, retest=retests, instruments=instruments,
                   policy_definitions={
                       'raw': 'Frozen phase1b suite normalization and CER including spaces; preserve Latin script.',
                       'translit': 'Exact frozen score2.py conditional transliteration then phase1b suite scoring.',
                       'arena_raw': 'Historical arena NFC/punctuation/digit/lowercase normalization; character CER excludes spaces; no transliteration.'},
                   normalization_effect={}, reuse_verification=dict(
                       scored_source_sha256=PINS['scored.jsonl'], numbers_source_sha256=PINS['numbers.json'],
                       every_reused_metric_recomputed=True, aggregates_match_phase1b=True),
                   limitations=['Speaker-cluster intervals condition on fixed text/session panel.',
                                'Four PV speakers; twelve Fresh voices; no population qualification.',
                                'Fresh silver transcription/admission bias and training exposure remain unresolved.',
                                'No human listening/transcription study or TTS seed spread.',
                                'Public 32M judge availability depends on owner decision; no weights released.'])
    for policy in POLICIES:
        numbers['asr'][policy] = {}
        for stage in stages:
            numbers['asr'][policy][stage] = {}
            for cohort in ['pv', 'fresh', 'all']:
                rr = [r for r in scored if r['policy'] == policy and r['stage'] == stage
                      and (cohort == 'all' or r['cohort'] == cohort)]
                conditions = {c: [r for r in rr if r['condition'] == c] for c in CONDITIONS}
                out = {c: frozen.aggregate(conditions[c]) for c in CONDITIONS}
                for c in CONDITIONS[1:]:
                    out[c]['paired'] = frozen.paired(conditions['real'], conditions[c])
                if stage in OLD_STAGES and policy != 'arena_raw':
                    compare_numeric(out, old_numbers['asr'][policy][stage][cohort],
                                    'Recomputed reused aggregate/'+policy+'/'+stage+'/'+cohort)
                numbers['asr'][policy][stage][cohort] = out
    for stage in stages:
        numbers['normalization_effect'][stage] = {}
        for cohort in ['pv', 'fresh', 'all']:
            rr = [r for r in scored if r['stage'] == stage and r['condition'] == 'real'
                  and (cohort == 'all' or r['cohort'] == cohort)]
            numbers['normalization_effect'][stage][cohort] = frozen.paired(
                [r for r in rr if r['policy'] == 'raw'], [r for r in rr if r['policy'] == 'translit'])
    # Publish only hashes/aggregate schemas, never detailed paths, text or identities.
    source_paths.extend([root/'tasks.json', root/'retest.json', Path(__file__),
                         Path(__file__).with_name('arena_normalize.py'), Path(__file__).with_name('arena_metrics.py')])
    numbers['input_receipts'] = {'phase1b/'+name: {'sha256': digest, 'bytes': (old/name).stat().st_size}
                                for name, digest in PINS.items()}
    for path in source_paths:
        numbers['input_receipts']['job/'+path.name] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    numbers['normalization_verified_files'] = len(normalization_manifest['files'])
    catalog = make_catalog(numbers)
    for filename, payload in [('numbers.json', numbers), ('calibration_public.json', catalog)]:
        (root/filename).write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n')
    with (root/'scored_internal.jsonl').open('x') as f:
        for row in scored:
            f.write(json.dumps(row)+'\n')
    print('COMPLETE', len(scored), 'numeric internal rows;', len(catalog['profiles']),
          'aggregate profiles; no transcript exported', flush=True)


if __name__ == '__main__':
    main()
