"""Render the review report only from complete, sanitized aggregate evidence.

Run from the repository after copying the completed calibration/ranking numbers.
This script reads no hypotheses, transcripts, speaker identities, audio or weights.
"""

import argparse
import datetime
import hashlib
import itertools
import json
import math
from pathlib import Path

STAGES = ['asr_long1c', 'd32_streaming', 'd32_full', 'asr_mms', 'asr_small', 'asr_whisper']
NAMES = {
    'asr_long1c': 'LONG1C', 'd32_streaming': 'D32 320 ms', 'd32_full': 'D32 full',
    'asr_mms': 'MMS npi', 'asr_small': 'Whisper small', 'asr_whisper': 'Whisper v3 turbo',
}
JUDGES = ['whisper_small', 'mms', 'long1c', 'd32_streaming', 'd32_full']
SHORT_JUDGES = {'whisper_small': 'Small', 'mms': 'MMS', 'long1c': 'LONG1C',
                'd32_streaming': 'D32s', 'd32_full': 'D32f'}
SYSTEMS = ['swar', 'vox2_beat1', 'vox2_beat2', 'cosy3_elx_neutral', 'vox2_clone']
SYSTEM_NAMES = {'swar': 'Swar', 'vox2_beat1': 'Vox1', 'vox2_beat2': 'Vox2',
                'cosy3_elx_neutral': 'Cosy', 'vox2_clone': 'Stock'}
POLICIES = ['suite_raw', 'legacy_transliterated', 'arena_raw']
CALIBRATION_POLICIES = ['raw', 'translit', 'arena_raw']
COHORTS = {'pv': (100, 4), 'fresh': (96, 12), 'all': (196, 16)}
CONDITIONS = ['real', 'degraded', 'shuffled_text']


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    value = json.loads(path.read_text())
    print('SCHEMA', path.name, sorted(value), flush=True)
    return value


def finite(value, label):
    require(type(value) in (int, float) and math.isfinite(value), 'Invalid numeric value: '+label)


def interval(value, label, minimum=None, maximum=None):
    require(isinstance(value, list) and len(value) == 2, 'Missing CI: '+label)
    for x in value:
        finite(x, label)
    require(value[0] <= value[1], 'Reversed CI: '+label)
    if minimum is not None:
        require(value[0] >= minimum-1e-12, 'CI below range: '+label)
    if maximum is not None:
        require(value[1] <= maximum+1e-12, 'CI above range: '+label)


def validate(calibration, ranking):
    require(calibration['targets'] == 196 and calibration['bootstraps'] == 10000,
            'Incomplete calibration panel/bootstrap')
    require(calibration['reuse_verification']['every_reused_metric_recomputed'] is True
            and calibration['reuse_verification']['aggregates_match_phase1b'] is True,
            'Historical scoring reproduction did not pass')
    for policy in CALIBRATION_POLICIES:
        require(set(calibration['asr'][policy]) == set(STAGES), 'Incomplete candidate panel: '+policy)
        for stage in STAGES:
            for cohort, (n, speakers) in COHORTS.items():
                for condition in CONDITIONS:
                    row = calibration['asr'][policy][stage][cohort][condition]
                    require(row['n'] == n and row['speakers'] == speakers,
                            'Incomplete calibration cohort: '+stage+'/'+cohort)
                    for key in ['corpus_cer_percent', 'mean_cer_percent', 'corpus_wer_percent']:
                        finite(row[key], stage+'/'+cohort+'/'+key)
                        require(row[key] >= 0, 'Negative error rate')
                    for key in ['ci95_corpus_speaker_bootstrap', 'ci95_mean_speaker_bootstrap',
                                'ci95_wer_speaker_bootstrap']:
                        interval(row[key], stage+'/'+cohort+'/'+key, minimum=0)
                    if condition != 'real':
                        p = row['paired']
                        require(p['n'] == n and 0 <= p['worse_count'] <= n,
                                'Incomplete paired controls')
                        interval(p['ci95_corpus_delta_speaker'], 'Control corpus difference')
                        interval(p['ci95_mean_delta_speaker'], 'Control sentence difference')
    for stage in STAGES:
        retest = calibration['retest'][stage]
        require([p['pass_id'] for p in retest['passes']] == [1, 2], 'Missing cold retest: '+stage)
        require(all(p['n'] == 32 for p in retest['passes']), 'Incomplete cold retest: '+stage)
        require(all(0 <= p['changed_hypotheses'] <= 32 for p in retest['passes']), 'Invalid retest drift')
        require(0 <= retest['cold_to_cold_changed'] <= 32, 'Invalid cold-to-cold drift')
    require(ranking['bootstrap_draws'] == 10000 and set(ranking['normalizations']) == set(POLICIES),
            'Ranking bootstrap/policy mismatch')
    require('fleurs' in ranking['conditions'], 'Primary FLEURS reference condition missing')
    primary = ranking['conditions']['fleurs']
    require(primary['n_prompts'] == 205 and primary['n_systems'] == 5, 'Incomplete primary track')
    expected_judge_pairs = {frozenset(pair) for pair in itertools.combinations(JUDGES, 2)}
    expected_system_pairs = {frozenset(pair) for pair in itertools.combinations(SYSTEMS, 2)}
    for policy in POLICIES:
        row = primary['policies'][policy]
        require(set(row['aggregates']) == set(JUDGES) and set(row['ranks']) == set(JUDGES),
                'Require all five primary judges: '+policy)
        for judge in JUDGES:
            require(set(row['aggregates'][judge]) == set(SYSTEMS), 'Incomplete ranked systems')
            require(len(row['ranks'][judge]) == 5 and set(row['ranks'][judge]) == set(SYSTEMS),
                    'Invalid rank permutation')
            means = []
            for system in row['ranks'][judge]:
                score = row['aggregates'][judge][system]
                finite(score['mean_sentence_cer_percent'], 'Ranked sentence CER')
                finite(score['corpus_cer_percent'], 'Ranked corpus CER')
                require(score['mean_sentence_cer_percent'] >= 0 and score['corpus_cer_percent'] >= 0,
                        'Negative ranking error rate')
                interval(score['ci95_mean_sentence'], 'Ranking sentence CI', minimum=0)
                interval(score['ci95_corpus'], 'Ranking corpus CI', minimum=0)
                means.append(score['mean_sentence_cer_percent'])
            require(means == sorted(means), 'Ranks disagree with sentence-CER aggregates')
        comparisons = row['judge_comparisons']
        require(len(comparisons) == 10 and
                {frozenset((c['judge_a'], c['judge_b'])) for c in comparisons} == expected_judge_pairs,
                'Missing/duplicate judge-pair tau results')
        for comparison in comparisons:
            finite(comparison['kendall_tau_b'], 'Kendall tau-b')
            require(-1 <= comparison['kendall_tau_b'] <= 1, 'Tau outside range')
            interval(comparison['ci95_tau'], 'Tau CI', -1, 1)
            require(0 <= comparison['undefined_bootstraps'] < ranking['bootstrap_draws'],
                    'No valid tau bootstrap draws')
            require(set(comparison['system_deltas']) == set(SYSTEMS), 'Incomplete judge-change contrasts')
            for delta in comparison['system_deltas'].values():
                finite(delta['sentence_delta_pp'], 'Judge-change paired difference')
                interval(delta['ci95_sentence_delta'], 'Judge-change paired CI')
        contrasts = row['paired_system_contrasts']
        require(len(contrasts) == 50, 'Expected five judges × ten system contrasts')
        for judge in JUDGES:
            subset = [c for c in contrasts if c['judge'] == judge]
            require(len(subset) == 10 and
                    {frozenset((c['a'], c['b'])) for c in subset} == expected_system_pairs,
                    'Incomplete system contrasts for '+judge)
        for contrast in contrasts:
            finite(contrast['sentence_delta_pp'], 'Paired sentence difference')
            finite(contrast['corpus_delta_pp'], 'Paired corpus difference')
            interval(contrast['ci95_sentence_delta'], 'Paired sentence CI')
            interval(contrast['ci95_corpus_delta'], 'Paired corpus CI')
    require(bool(calibration['input_receipts']) and bool(ranking['sources']), 'Missing evidence hashes')


def estimate(value, bounds, signed=False):
    fmt = '+.2f' if signed else '.2f'
    return f'{value:{fmt}} [{bounds[0]:{fmt}}, {bounds[1]:{fmt}}]'


def table(headers, rows):
    lines = ['| '+' | '.join(headers)+' |', '| '+' | '.join('---' for _ in headers)+' |']
    return '\n'.join(lines+['| '+' | '.join(str(cell) for cell in row)+' |' for row in rows])


def baseline_delta(row, judge, system):
    candidates = [c for c in row['paired_system_contrasts']
                  if c['judge'] == judge and {c['a'], c['b']} == {'swar', system}]
    require(len(candidates) == 1, 'Missing comparison with Swar')
    value = candidates[0]
    if value['a'] == system:
        delta, bounds = value['sentence_delta_pp'], value['ci95_sentence_delta']
    else:
        delta = -value['sentence_delta_pp']
        bounds = [-value['ci95_sentence_delta'][1], -value['ci95_sentence_delta'][0]]
    return estimate(delta, bounds, signed=True)


def validate_group_sensitivity(calibration, ranking, grouped):
    """Require complete grouped evidence with the same fixed-data estimands."""
    validate(calibration, grouped)
    require(grouped['conditions']['fleurs']['bootstrap_groups'] == 133,
            'Expected 133 minimal-pair/singleton groups')
    require(grouped['seed'] == ranking['seed'] and grouped['sources'] == ranking['sources'],
            'Grouped sensitivity must use identical hypotheses and bootstrap seed')
    require(grouped['normalization_manifest_sha256'] == ranking['normalization_manifest_sha256'],
            'Grouped sensitivity normalization drift')
    for policy in POLICIES:
        original = ranking['conditions']['fleurs']['policies'][policy]
        alternative = grouped['conditions']['fleurs']['policies'][policy]
        require(original['ranks'] == alternative['ranks'], 'Grouped bootstrap changed fixed-data ranks')
        for judge in JUDGES:
            for system in SYSTEMS:
                for key in ['mean_sentence_cer_percent', 'corpus_cer_percent']:
                    require(math.isclose(original['aggregates'][judge][system][key],
                                         alternative['aggregates'][judge][system][key],
                                         rel_tol=1e-12, abs_tol=1e-12),
                            'Grouped bootstrap changed a fixed-data score')
        for before in original['paired_system_contrasts']:
            after = next(c for c in alternative['paired_system_contrasts']
                         if (c['judge'], c['a'], c['b']) == (before['judge'], before['a'], before['b']))
            for key in ['sentence_delta_pp', 'corpus_delta_pp']:
                require(math.isclose(before[key], after[key], rel_tol=1e-12, abs_tol=1e-12),
                        'Grouped bootstrap changed a fixed-data contrast')
        for before in original['judge_comparisons']:
            after = next(c for c in alternative['judge_comparisons']
                         if {c['judge_a'], c['judge_b']} == {before['judge_a'], before['judge_b']})
            require(math.isclose(before['kendall_tau_b'], after['kendall_tau_b'], abs_tol=1e-12),
                    'Grouped bootstrap changed a fixed-data tau')


def render(calibration, ranking, grouped, root):
    primary = ranking['conditions']['fleurs']['policies']
    corpus = lambda row: estimate(row['corpus_cer_percent'], row['ci95_corpus_speaker_bootstrap'])
    sentence = lambda row: estimate(row['mean_cer_percent'], row['ci95_mean_speaker_bootstrap'])
    small = calibration['asr']['arena_raw']['asr_small']
    taus = [c['kendall_tau_b'] for c in primary['legacy_transliterated']['judge_comparisons']]
    lines = [
        '# Arena judge calibration and local rank sensitivity', '',
        f'Generated {datetime.datetime.now(datetime.timezone.utc).isoformat()}, Codex `/root` '
        '(generator authored by `/root/docs_reconcile`), '
        'job `arena_judges_v1`; owner Ampixa / Arbiter `nenglish-73`. Automatic measurements for review. '
        '**No public submission audio was accessed or rescored; private audio and D32 weights remain local.**', '',
        f'Across five local-system judges, legacy-policy Kendall τ-b spans {min(taus):.2f}–{max(taus):.2f}. '
        'This measures instrument sensitivity on fixed saved outputs, not an overall TTS winner.', '',
        '## Real speech', '',
        'CER %, estimate [95% CI]. PV: 100 read recordings / four speakers. Fresh: 96 silver-transcribed '
        '`real_yt` recordings / twelve voices. All targets included. Corpus CER weights characters; sentence '
        f'CER weights utterances. CIs: {calibration["bootstraps"]:,} paired speaker-cluster draws, '
        f'seed {calibration["seed"]}, conditional on fixed texts/sessions.', '',
    ]
    rows = []
    for stage in STAGES:
        for cohort in ['pv', 'fresh']:
            raw = calibration['asr']['raw'][stage][cohort]['real']
            translit = calibration['asr']['translit'][stage][cohort]['real']
            rows.append([NAMES[stage], 'PV' if cohort == 'pv' else 'Fresh',
                         corpus(raw), sentence(raw), corpus(translit), sentence(translit)])
    lines += [table(['Judge', 'Set', 'Raw corpus', 'Raw sentence', 'Legacy corpus', 'Legacy sentence'], rows), '',
              'Raw uses frozen phase1b suite scoring, including spaces. Legacy adds exact `score2.py` conditional '
              'transliteration: convert Latin-containing hypotheses only when the reference has no ASCII Latin. '
              'This compatibility rule has no new native accepted-script validation.', '',
              '### Current arena: Whisper small', '',
              '`whisper-small-ne-048909d9dafd`: **uncalibrated for spontaneous Nepali**, with measured numbers. '
              'Arena normalization removes punctuation, maps digits to ASCII, lowercases, excludes spaces from '
              'character CER and does not transliterate loanwords. Its macro sentence metric is separate from phase1b.', '',
              table(['Set', 'Arena corpus CER', 'Arena sentence CER'],
                    [[label, corpus(small[cohort]['real']), sentence(small[cohort]['real'])]
                     for cohort, label in [('pv', 'PV'), ('fresh', 'Fresh')]]), '',
              'Weights, package versions and CPU decoding options match the historical profile; v3 turbo is a '
              'distinct candidate and cannot calibrate small. The archived profile does not record FFmpeg, '
              'the host architecture or all transitive dependencies; matching its declared settings is not '
              'proof of cross-host bitwise transcript equality. The measured environment is recorded in '
              '[environment_small.json](calibration/environment_small.json).', '',
              'D32 streaming uses the existing int8 c16/left-128 ONNX export; full context uses the strict-loaded '
              'FP32 PyTorch checkpoint with chunk/left context −1. Their difference includes context, precision '
              'and execution implementation, so it cannot isolate the causal effect of context alone.', '',
              '## Controls and cold retests', '',
              'Paired **corpus-CER Δpp**, control minus real [95% CI]; worse/n. Small uses arena raw, others legacy. '
              'Degradation and within-cohort text derangement exactly reuse phase1b; ASRs receive audio only.', '']
    rows = []
    for stage in STAGES:
        policy = 'arena_raw' if stage == 'asr_small' else 'translit'
        for cohort in ['pv', 'fresh']:
            result = calibration['asr'][policy][stage][cohort]
            cells = []
            for condition in ['degraded', 'shuffled_text']:
                v = result[condition]['paired']
                cells.append(estimate(v['corpus_delta_cer_pp'], v['ci95_corpus_delta_speaker'], True)
                             +f'; {v["worse_count"]}/{v["n"]}')
            rows.append([NAMES[stage], 'PV' if cohort == 'pv' else 'Fresh', *cells])
    lines += [table(['Judge', 'Set', 'Degraded Δ; worse/n', 'Shuffled Δ; worse/n'], rows), '',
              '≥90% worsening plus positive paired lower bounds is a coarse sanity screen, not subtle-TTS '
              'qualification. Corpus/sentence conclusions may differ. CER >100% is retained.', '']
    rows = []
    for stage in STAGES:
        t = calibration['retest'][stage]
        rows.append([NAMES[stage], *(f'{v["changed_hypotheses"]}/{v["n"]}' for v in t['passes']),
                     f'{t["cold_to_cold_changed"]}/32'])
    lines += [table(['Judge', 'Pass 1 changed', 'Pass 2 changed', 'Cold-to-cold changed'], rows), '',
              'One real/degraded input per voice: 32/392 unique inputs (8.16%), below the prospective 10% '
              'sampling gate. These recognizer retests do not measure TTS seed spread or new sessions.', '',
              '## Five systems × 205 prompts', '',
              '**FLEURS reference condition**, fixed saved takes. Vox1/Vox2: VoxCPM2 best8 run1/run2; '
              'Cosy: CosyVoice3 neutral; Stock: stock VoxCPM2. Best8 uses a LONG1C selector, making LONG1C '
              'dependent evidence for selected outputs. No synthesis/selection was rerun.', '',
              'Legacy **mean sentence CER % [95% CI]**. Paired prompt bootstrap conditions on saved takes, '
              'reference voice and settings; it does not cover new voices/seeds.', '']
    rows = []
    for system in SYSTEMS:
        cells = []
        for judge in JUDGES:
            v = primary['legacy_transliterated']['aggregates'][judge][system]
            cells.append(estimate(v['mean_sentence_cer_percent'], v['ci95_mean_sentence']))
        rows.append([SYSTEM_NAMES[system], *cells])
    lines += [table(['System', *(SHORT_JUDGES[j] for j in JUDGES)], rows), '',
              'Point order runs from lower to higher CER (`→`); no statistically distinct winner is implied.', '']
    rows = [[SHORT_JUDGES[judge],
             *(' → '.join(SYSTEM_NAMES[s] for s in primary[p]['ranks'][judge]) for p in POLICIES)]
            for judge in JUDGES]
    lines += [table(['Judge', 'Suite raw order', 'Legacy order', 'Arena raw order'], rows), '',
              f'Tie-aware Kendall τ-b [95% CI], {ranking["bootstrap_draws"]:,} paired prompt draws, '
              f'seed {ranking["seed"]}. These are percentile intervals, not simultaneous multiple-comparison bounds.', '']
    rows = []
    for a, b in itertools.combinations(JUDGES, 2):
        values = []
        for policy in POLICIES:
            v = next(c for c in primary[policy]['judge_comparisons']
                     if {c['judge_a'], c['judge_b']} == {a, b})
            values.append(estimate(v['kendall_tau_b'], v['ci95_tau'])
                          +(f'; undef {v["undefined_bootstraps"]}' if v['undefined_bootstraps'] else ''))
        rows.append([SHORT_JUDGES[a]+' / '+SHORT_JUDGES[b], *values])
    lines += [table(['Judge pair', 'Suite raw τ-b', 'Legacy τ-b', 'Arena raw τ-b'], rows), '',
              '**Legacy paired system-minus-Swar mean CER Δpp [95% CI]**; negative means fewer ASR errors '
              'under that judge, not an isolated best-of-N effect or human preference.', '']
    rows = [[SHORT_JUDGES[judge], SYSTEM_NAMES[system],
             baseline_delta(primary['legacy_transliterated'], judge, system)]
            for judge in JUDGES for system in SYSTEMS if system != 'swar']
    lines += [table(['Judge', 'System − Swar', 'Legacy Δpp'], rows), '',
              '### Minimal-pair dependence sensitivity', '',
              'The 205 prompts contain **133 groups: 72 complete minimal pairs and 61 singleton prompts**. '
              'Resampling these groups keeps paired prompts together and preserves utterance-weighted estimands. '
              'Point scores/ranks are unchanged. This checks one known text dependence structure; it is not '
              'a voice/seed CI or proof that all prompt dependence is captured.', '',
              'Legacy **Vox1 minus Swar** mean CER Δpp [95% CI], comparing the 205-prompt bootstrap '
              'with the 133-group bootstrap:', '',
              table(['Judge', '205 prompts', '133 groups'], [
                  [SHORT_JUDGES[judge], baseline_delta(primary['legacy_transliterated'], judge, 'vox2_beat1'),
                   baseline_delta(grouped['conditions']['fleurs']['policies']['legacy_transliterated'],
                                  judge, 'vox2_beat1')]
                  for judge in JUDGES]), '',
              'Legacy τ-b [95% CI] versus LONG1C, using the same two resampling units:', '']
    rows = []
    for judge in JUDGES:
        if judge == 'long1c':
            continue
        cells = []
        for result in [ranking, grouped]:
            comparison = next(c for c in result['conditions']['fleurs']['policies']
                              ['legacy_transliterated']['judge_comparisons']
                              if {c['judge_a'], c['judge_b']} == {judge, 'long1c'})
            cells.append(estimate(comparison['kendall_tau_b'], comparison['ci95_tau'])
                         +(f'; undef {comparison["undefined_bootstraps"]}'
                           if comparison['undefined_bootstraps'] else ''))
        rows.append([SHORT_JUDGES[judge]+' / LONG1C', *cells])
    lines += [table(['Judge pair', '205 prompts', '133 groups'], rows), '',
              '[ranking/numbers.json](ranking/numbers.json) contains **all three policies’** system-pair and '
              'per-system judge-change paired CIs, corpus CIs, aggregates and τ-b/tie diagnostics. '
              '[numbers_pair_clusters.json](ranking/numbers_pair_clusters.json) contains the corresponding '
              'complete grouped sensitivity. '
              '[calibration/numbers.json](calibration/numbers.json) contains every raw/legacy/arena control '
              'sentence/corpus CI. Rasa is separate and never pooled as 410 independent prompts.', '',
              '## Limits and evidence', '',
              'Real ASR error is not a hard 10% discrimination floor and is not subtracted from synthetic CER. '
              'Controls do not establish native intelligibility, voice identity, style or overall quality. '
              '**No new human listening/MOS, transcription/accepted-script review or independent speaker/session '
              'qualification was performed.** Native-rater work remains separate.', '',
              '[REVIEW_GATES.md](REVIEW_GATES.md) and [ranking/contamination.json](ranking/contamination.json) '
              'record finite corpus screens, known benchmark tuning, Fresh silver/admission bias and private-training '
              'unknowns. Text absence does not prove decontamination or new speakers; 205-track short-phrase '
              'overlap and reference-source exposure remain confounds.', '',
              '**D32 is call-trained; a public 32M judge profile depends on the owner’s open publication decision.** '
              'No weights were uploaded. Other candidates await executable public worker/import support; only '
              'the existing small profile is enabled. Separate ranks and required take/N/selector/judge-family '
              'disclosures avoid silent mixing, but declarations are not execution attestation.', '',
              'Calibration JSON records source/runtime/model hashes and verified input/normalizer pins. Every '
              'reused raw/legacy score and paired interval reproduces frozen phase1b. D32 strict loading and '
              'checkpoint/tokenizer hashes are recorded without private text or weights.', '',
              'Reproduce after complete aggregates: `python3 benchmark/judge_calibration_v1/calibration/report.py`. '
              'Missing six-candidate coverage, five-judge primary results, paired CIs or finite values are fatal. '
              '[arena.yml](../../.github/workflows/arena.yml) specifies Python/Ruff/JS checks and receiver compilation; '
              'actual runs belong in the PR and verify software only. '
              '[Source-install evidence](../../neptts-eval/source_install_validation.json) is separate.', '',
              table(['Input', 'SHA256'], [
                  ['calibration/numbers.json', '`'+sha(root/'calibration/numbers.json')+'`'],
                  ['ranking/numbers.json', '`'+sha(root/'ranking/numbers.json')+'`'],
                  ['ranking/numbers_pair_clusters.json', '`'+sha(root/'ranking/numbers_pair_clusters.json')+'`'],
                  ['calibration/report.py', '`'+sha(Path(__file__))+'`'],
              ]), '',
              '[README.md](README.md) owns provenance, inventory, copy verification and retention. '
              'These conditional automatic measurements do not qualify a universal TTS leaderboard.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root/'REPORT.md'
    require(not output.exists(), 'Refuse to overwrite an existing report: '+str(output))
    calibration = read(root/'calibration/numbers.json')
    ranking = read(root/'ranking/numbers.json')
    grouped = read(root/'ranking/numbers_pair_clusters.json')
    validate(calibration, ranking)
    validate_group_sensitivity(calibration, ranking, grouped)
    report = render(calibration, ranking, grouped, root)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(report)
    print('WROTE', output, len(report.encode()), 'bytes; complete aggregate evidence only', flush=True)


if __name__ == '__main__':
    main()
