"""Render the completed five-judge ranking evidence without exposing hypotheses/audio."""
import datetime
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
NAMES=dict(swar='Swar',vox2_beat1='VoxCPM2 run1',vox2_beat2='VoxCPM2 run2',
           cosy3_elx_neutral='CosyVoice3 neutral',vox2_clone='Stock VoxCPM2')
JUDGES=dict(whisper_small='Whisper-small',mms='MMS npi',long1c='LONG1C',
            d32_streaming='D32 320 ms',d32_full='D32 full context')


def interval(point,limits):
    return f'{point:.2f} [{limits[0]:.2f}, {limits[1]:.2f}]'


def main():
    raw=json.loads((ROOT/'numbers.json').read_text())
    families=json.loads((ROOT/'numbers_pair_clusters.json').read_text())
    p=raw['conditions']['fleurs']['policies']['legacy_transliterated']
    q=families['conditions']['fleurs']['policies']['legacy_transliterated']
    assert set(p['aggregates'])==set(JUDGES)==set(q['aggregates'])
    assert len(p['paired_system_contrasts'])==len(q['paired_system_contrasts'])==50
    assert len(p['judge_comparisons'])==len(q['judge_comparisons'])==10
    lines=['# Judge choice changes the local 205-prompt ranking','',
        f'Generated {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")} by Codex `/root` using a report generator prepared by `/root/rank_assets`, job `arena_judges_v1`. Owner: Ampixa / Arbiter nenglish-73. Reference evidence for PR review; next review 2026-11-11. No audio or weights are included. NOT on HF.','',
        'These are five saved local systems on the same **205 arena prompt IDs/texts**, under the head-to-head FLEURS-reference generation condition. The five approved public arena reports were not rescored or accessed. Rasa-reference outputs were inventoried but not rescored; the two conditions are not pooled as 410 independent prompts.','',
        'The main comparison applies the frozen phase1b legacy loanword rule and suite normalizer to every judge. Both corpus CER and mean sentence CER are percentages. Lower is better. The unchanged arena normalization is separately available under `arena_raw`; it removes spaces from character counting and does not transliterate Latin tokens. Consequently this table uses the arena Whisper decoding instrument with an explicitly different, common comparison policy.','',
        '## Mean sentence CER with paired-prompt 95% intervals','',
        '| System | '+' | '.join(JUDGES.values())+' |',
        '|---|'+'---|'*len(JUDGES)]
    for system,name in NAMES.items():
        cells=[]
        for judge in JUDGES:
            row=p['aggregates'][judge][system]
            cells.append(interval(row['mean_sentence_cer_percent'],row['ci95_mean_sentence']))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## Corpus CER with paired-prompt 95% intervals','',
            '| System | '+' | '.join(JUDGES.values())+' |','|---|'+'---|'*len(JUDGES)]
    for system,name in NAMES.items():
        cells=[]
        for judge in JUDGES:
            row=p['aggregates'][judge][system]
            cells.append(interval(row['corpus_cer_percent'],row['ci95_corpus']))
        lines.append('| '+name+' | '+' | '.join(cells)+' |')
    lines+=['','## Rank sensitivity','',
        'Kendall tau-b compares the five system means. Intervals resample the same prompts or groups jointly across every system and judge; they are conditional rank uncertainty on this panel, not population uncertainty over all TTS systems.','',
        '| Judges | Tau-b | 205-prompt 95% CI | 133-group 95% CI |','|---|---:|---|---|']
    grouped={(r['judge_a'],r['judge_b']):r for r in q['judge_comparisons']}
    for row in p['judge_comparisons']:
        other=grouped[(row['judge_a'],row['judge_b'])]
        a,b=row['ci95_tau'];c,d=other['ci95_tau']
        lines.append(f'| {JUDGES[row["judge_a"]]} / {JUDGES[row["judge_b"]]} | {row["kendall_tau_b"]:.2f} | [{a:.2f}, {b:.2f}] | [{c:.2f}, {d:.2f}] |')
    lines+=['','## Paired system contrasts','',
        '**Difference = Swar minus the named VoxCPM2 run**, in sentence-CER percentage points. Positive favors that VoxCPM2 run; negative favors Swar. All 50 system contrasts, including every other system pair, are retained per normalization in the two JSON artifacts. These are descriptive intervals without a multiple-comparison correction.','',
        '| Judge | Comparator | Difference | 205-prompt 95% CI | 133-group 95% CI |','|---|---|---:|---|---|']
    grouped={(r['judge'],r['a'],r['b']):r for r in q['paired_system_contrasts']}
    for row in p['paired_system_contrasts']:
        if row['a']!='swar' or row['b'] not in ['vox2_beat1','vox2_beat2']:
            continue
        other=grouped[(row['judge'],row['a'],row['b'])]
        a,b=row['ci95_sentence_delta'];c,d=other['ci95_sentence_delta']
        lines.append(f'| {JUDGES[row["judge"]]} | {NAMES[row["b"]]} | {row["sentence_delta_pp"]:+.2f} | [{a:.2f}, {b:.2f}] | [{c:.2f}, {d:.2f}] |')
    lines+=['',
        '## Interpretation and limits','',
        '- Both fine-tuned VoxCPM2 runs used eight candidates and chose the lowest **LONG1C streaming c16 CER against the target text**, with lower-seed tie breaking. Their frozen receipts explicitly say no MMS or ECAPA selector. LONG1C therefore evaluates outputs selected by that same judge family; its apparent advantage is optimistic by construction. The cross-judge reversal is evidence for separate judge profiles and selection disclosure, not proof of a universally superior synthesizer.',
        '- Stock VoxCPM2 had retry machinery enabled but all 205 primary receipt rows record zero bad-case retries. Swar and CosyVoice3 have saved one-take seed-zero generation receipts; the comparison conditions remain fixed and differ from the eight-candidate selection conditions. No new synthesis or seed study was performed.',
        '- Ten thousand paired draws use seed 20261104. The primary unit is the prompt. The sensitivity analysis resamples 133 known `pair_id` groups: 72 complete two-prompt minimal pairs and 61 singleton prompts. It preserves both members together, but does not capture every possible text dependence. Neither interval is a new-voice, new-reference, human-rater or synthesis-seed interval.',
        '- These are intelligibility proxies. Real-speech calibration, degraded/shuffled controls and cold-process repeatability are reported by the parent calibration package. Real-speech CER is not a correction term to subtract from synthetic CER. Voice identity, naturalness and overall quality are not established by these scores.',
        '- Public text contamination checks reuse the hash-pinned audit after verifying exact identity of all 205 queries, and add Common Voice 17 (1,446 rows, 703 normalized unique strings): zero exact/15-character-containment matches in that finite CV snapshot. Short common-phrase matches occur in Rasa, IndicVoices and OpenSLR; FLEURS/Rasa reference-source exposure and public test tuning remain confounds. No finite text screen proves speaker or training exclusion.',
        '- D32 is call-trained. Its weights were only loaded locally on k2 and were never uploaded. A public D32 judge profile depends on the owner’s unresolved publication decision.',
        '- Whisper reproduction matches the declared source/weights/package versions and decoding options. The original public profile omitted FFmpeg, OS and architecture, so cross-host bitwise equivalence is unverified. The current environment receipt records FFmpeg binary SHA/version and macOS arm64. Both earlier failed FFmpeg and extra-interop/seed attempts were explicitly excluded; no partial hypotheses from them enter final numbers.','',
        '## Artifacts and reproduction','',
        '- [numbers.json](numbers.json): all three normalization policies; five judges × five systems; corpus/macro CER and 50 paired contrasts plus 10 tau comparisons per policy, with 205-prompt CIs.',
        '- [numbers_pair_clusters.json](numbers_pair_clusters.json): the same point estimates and comparisons with paired 133-group CIs.',
        '- [contamination.json](contamination.json): compact finite public-text checks, query IDs and source hashes; no corpus transcript quotes.',
        '- [copy_verification.json](copy_verification.json): all 1,025 transported WAVs verified byte-identical to the frozen original synthesis receipts. No WAV is included here.',
        '- `prepare.py`, `mms_worker.py`, `whisper_worker.py`, `launch_whisper_exact.py`, `analyze.py`, and this `report.py` reproduce input admission, raw CPU inference, uncertainty and this document. Every legacy normalization/code/lexical dependency is checked against the phase1b frozen SHA256 manifest before scoring. Raw hypotheses and detailed execution receipts remain in the documented internal job directories; final aggregate files are hash-inventoried.',
        '', 'Source and runtime provenance, ownership, licences, retention and excluded-run details are in [README.md](README.md).']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
