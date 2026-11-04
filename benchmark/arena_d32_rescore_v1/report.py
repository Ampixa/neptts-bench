"""Render the review report exclusively from sanitized numerical evidence."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def val(row):
    return f"{row['percent']:.2f} [{row['ci95'][0]:.2f}, {row['ci95'][1]:.2f}]"


def main():
    data = json.loads((HERE / 'numbers.json').read_text())
    panel = data['analyses']['prompts']
    grouped = data['analyses']['pair_groups']
    gate = json.loads((HERE / 'calibration_gate.json').read_text())
    retest = json.loads((HERE / 'submission_retest.json').read_text())
    cleanup = json.loads((HERE / 'audio_cleanup.json').read_text())
    if not retest['passed'] or not cleanup['verified_zero_job_wav_zip']:
        raise ValueError('Retest/cleanup must complete before final report')
    lines = [
        '# D32 full-context rescore: five approved arena submissions', '',
        'All five approved submissions now have complete205-prompt D32 offline reports. **Nothing has been imported to the live arena.** Merge/deployment and separate reviewer import/approval remain with the Arbiter; [IMPORT.md](IMPORT.md) gives exact commands. The five existing Whisper reports and audio bindings were rechecked unchanged in [legacy_preservation.json](legacy_preservation.json).', '',
        f"Instrument: `{data['profile']['id']}`; 33,447,414 parameters, step60000, FP32 full context, CPU1thread, greedy, chunk/left context−1/−1, frozen legacy score2 conditional loanword transliteration. Call-trained private weights remain on k2. **Scores are reproducible only by Ampixa until the owner decides on weight release.** The profile card states this limitation; a public numeric report does not make the judge public.", '',
        '## Real-speech instrument error beside the arena scores', '',
        '| Exact judge and policy | PV100 real targets, corpus CER | Fresh96 real targets, corpus CER |',
        '|---|---:|---:|',
        '| D32 full / legacy transliteration | **7.88%** | **13.69%** |',
        '| Historical Whisper-small / arena normalization | **49.01%** | **115.45%** |', '',
        'The final executable reran every real calibration input: PV7.875536% and Fresh13.689444%, both **0.000000 percentage points** from the calibration record, against a maximum permitted difference0.05pt. All196 hypotheses were identical to the original full-context run. [calibration_gate.json](calibration_gate.json) binds these observations to the final code/profile hash; [instrument_verification.json](instrument_verification.json) records checkpoint, tokenizer and original five recipe pins. The profile additionally pins the21-module local recipe import closure, clean icefall revision, all15 normalizer/resource files and runtime versions. No weights were copied.', '',
        'These panels have4PV and12Fresh voices and fixed sessions/texts. The existing speaker-bootstrap95%CIs are PV[6.50,8.75]% and Fresh[10.97,16.24]% only if confirmed against the source catalog below; instrument error must never be subtracted from synthetic error or treated as a hard discrimination floor.', '',
        '## Approved submissions', '',
        'Every row covers the same205 required prompts. CER/WER below are **macro means of sentence error rates, percent**, matching arena aggregation; brackets are95% paired-prompt bootstrap intervals. The shared draws preserve pairing for differences. Real-speech D32 corpus CER is **PV7.88% / Fresh13.69%**; Whisper is **49.01% /115.45%** under its own historical policy.', '',
        '| D32 rank | Approved system | D32 CER [95%CI] | D32 WER [95%CI] | Whisper CER [95%CI] | Whisper rank |',
        '|---:|---|---:|---:|---:|---:|'
    ]
    # Pull exact calibration intervals rather than hand-copying them.
    catalog = json.loads((HERE.parent / 'judge_calibration_v1/calibration_public.json').read_text())
    full = next(p for p in catalog['profiles'] if p['stage'] == 'd32_full')
    pvci = full['calibration']['pv']['ci95_corpus_speaker_bootstrap']
    frci = full['calibration']['fresh']['ci95_corpus_speaker_bootstrap']
    lines[lines.index(next(s for s in lines if s.startswith('These panels have4PV')))] = (
        f'These panels have4PV and12Fresh voices and fixed sessions/texts. Existing speaker-bootstrap95%CIs are PV[{pvci[0]:.2f},{pvci[1]:.2f}]% and Fresh[{frci[0]:.2f},{frci[1]:.2f}]%. Instrument error must never be subtracted from synthetic error or treated as a hard discrimination floor.')
    for rank, sid in enumerate(panel['approved_d32_rank'], 1):
        row = panel['systems'][sid]
        lines.append(f"| {rank} | {row['name']} | {val(row['cer'])} | {val(row['wer'])} | {val(panel['whisper_scores'][sid]['cer'])} | {panel['approved_whisper_rank'].index(sid)+1} |")
    tau = panel['kendall_tau_b']
    lines.extend(['', f"The five-entry ranking changes from **Aayush → Sujata → WiseAI_TTS → Prakriti → Astadhwani** to **WiseAI_TTS → Aayush → Sujata → Prakriti → Astadhwani** (lowest meanCER first). Kendall tau-b is **{tau['point']:.2f}**, paired95%CI[{tau['ci95'][0]:.2f},{tau['ci95'][1]:.2f}]; the133-group minimal-pair sensitivity gives the same point and interval. WiseAI_TTS moves from third to first, reversing its order against Aayush and Sujata. These are observed instrument-specific rankings, not a human-quality result.", '',
        'The top two D32 systems are unresolved: WiseAI_TTS minus Aayush is−0.62CER points [−2.72,+1.36] and−2.16WER points [−5.61,+1.24]. Corpus weighting even reverses their point order: Aayush6.86% versus WiseAI_TTS7.22% corpusCER; their paired corpus difference+0.36pt [−0.90,+1.79] also includes zero. A first-place point estimate is not evidence of a winner.', '',
        'The comparison changes both recognizer and normalization: historical Whisper preserves Latin script and excludes spaces from CER; calibrated D32 conditionally transliterates Latin hypotheses when references contain no ASCII Latin and uses suite CER including spaces. The stored legacy reports contain numeric scores, not raw hypotheses, so this job cannot causally separate the ASR change from the loanword/space-policy change. Historical values remain untouched.', '',
        '## Reference context, not arena entries', '',
        'D32 real-speech corpus CER: **PV7.88% / Fresh13.69%**. Each synthetic reference below uses the same205 prompts and the existing FLEURS enrollment/reference condition. The615 original WAV hashes were rechecked on both Ubuntu and k2. The saved full-context hypotheses match the earlier committed source pin; CER/WER were recomputed through the final scorer normalizer. No additional synthesis or API was used.', '',
        '| Reference system | Mean CER [95%CI] | Mean WER [95%CI] | Corpus CER | Corpus WER |',
        '|---|---:|---:|---:|---:|'
    ])
    for sid in ['swar', 'vox2_clone', 'cosy3_elx_neutral']:
        row = panel['systems'][sid]
        name = {'swar':'Swar', 'vox2_clone':'VoxCPM2 stock, observed single attempt', 'cosy3_elx_neutral':'CosyVoice3 neutral'}[sid]
        lines.append(f"| {name} | {val(row['cer'])} | {val(row['wer'])} | {row['corpus_cer']['percent']:.2f}% | {row['corpus_wer']['percent']:.2f}% |")
    lines.extend(['',
        'VoxCPM2 stock has zero bad-case retries in all205 item receipts. The two finetuned VoxCPM2 beat runs used LONG1C-selected best-of-eight outputs and are excluded from this single-attempt context. Swar/Cosy receipts record no best-of-N selector but lack an explicit retry-count field. [REFERENCE_AUDIT.md](REFERENCE_AUDIT.md) records exact revisions, hashes and limits.', '',
        '**No real target recordings exist for the designed205 prompts** in the source manifest. The82 real recordings belong to a different cloning track; enrollment/reference audio is not target speech. Therefore no misleading same205 real row is added. PV/Fresh calibration is shown separately.', '',
        '## Paired differences and uncertainty', '',
        '10,000 shared resamples, seed20261104, pair identical prompt IDs across all eight systems. The main analysis resamples205 prompts; sensitivity resamples133 minimal-pair families/singletons (72pairs+61singletons) while keeping both pair members. Intervals condition on saved systems, enrollment voice and synthesis take; they do not cover new speakers, synthesis seeds or human judgments. No multiple-comparison correction is applied; all28 pairwise contrasts and both weighting conventions are retained in [numbers.json](numbers.json).', '',
        '| D32 difference (first minus second), points | Mean CER [paired95%CI] | Mean WER [paired95%CI] |133-group CER CI |',
        '|---|---:|---:|---:|'
    ])
    names = {r['name']: sid for sid, r in panel['systems'].items()}
    for a, b in [('WiseAI_TTS', 'WiseAI_TTS_Aayush'), ('WiseAI_TTS', 'WiseAI_TTS_Sujata'),
                 ('WiseAI_TTS', 'swar'), ('WiseAI_TTS', 'cosy3_elx_neutral'), ('swar', 'cosy3_elx_neutral')]:
        c = next(r for r in panel['paired_contrasts'] if r['a'] == names[a] and r['b'] == names[b])
        g = next(r for r in grouped['paired_contrasts'] if r['a'] == names[a] and r['b'] == names[b])
        def delta(x):
            return f"{x['delta_pp']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"
        lines.append(f"| {a} − {b} | {delta(c['cer'])} | {delta(c['wer'])} | [{g['cer']['ci95'][0]:+.2f}, {g['cer']['ci95'][1]:+.2f}] |")
    lines.extend(['',
        'The sample resolves several coarse differences, while WiseAI_TTS/Aayush and WiseAI_TTS/Swar include zero. Narrower fixed-prompt intervals do not repair judge bias or training exposure. Corpus and macro ranking sensitivity is explicitly retained.', '',
        '## Qualification, privacy and reproducibility', '',
        '- Known-good/bad evidence: original matched degradation worsened98/100PV and96/96Fresh items; transcript shuffles worsened all targets. Their paired corpus-CER lower bounds are positive. Reused calibration controls establish coarse content discrimination; ASR cannot score speaker identity when the words are correct. No wrong-speaker identity or human-preference claim is made.',
        '- Repeatability: this final decoder reproduced all196 original calibration hypotheses. A separate cold process redecoded21/205 prompts per approved submission (10.24%,105 total), with zero changed CER/WER/edit-count rows. No private submission hypotheses were stored, so this is numeric repeatability. Earlier calibration has two32-input cold hypothesis retests; no TTS seed spread was measured.',
        '- Contamination: the exact205-prompt public-text screen and input hashes are reused from [the calibration audit](../judge_calibration_v1/ranking/contamination.json), covering FLEURS, Rasa, IndicVoices/IVR, OpenSLR43/54 and Common Voice snapshots. Twelve short common-phrase matches and benchmark tuning remain known confounds; FLEURS/Rasa enrollment exposure is explicit. Actual private D32 training exclusion and submitter training membership are unverified. Fresh is silver-transcribed and historically ASR/speaker-selected. No sealed or universally uncontaminated ranking is claimed.',
        '- Scope corrections to inherited premises are in [REVIEW_GATES.md](../judge_calibration_v1/REVIEW_GATES.md):9.8% real MMS error is not a mathematical floor; historical82 recordings/50text clusters differ from the current advertised64prompt pairs; best-of-N comparisons are confounded and do not prove deliberate gaming.',
        '- Credentials were read only in VM memory. Submitted packages streamed VM→k2 without a Mac audio copy. All inference used k2 CPU, one thread, after memory-pressure checks; no live hat ports/processes, arena results or proper-voices service were changed. The initial missing-pydantic preflight failed before scoring; validator dependencies were installed only in this job directory, with the shared runtime unchanged.',
        f"- All five new ZIP copies and transient extracted WAVs were deleted after consumers finished and lsof found no open inputs. Package bytes removed:{cleanup['package_bytes_deleted']:,}; allocated package bytes:{cleanup['allocated_package_bytes_deleted']:,}; measured free-space increase:{cleanup['measured_free_delta_bytes']:,} bytes (concurrent host activity can affect the delta). Zero WAV/ZIP files remain in the job tree. [audio_cleanup.json](audio_cleanup.json) records IDs, hashes and deletion reasons; the full per-ID audio hash manifest remains on k2. Arena originals remain intact; ZIP-container byte history may not be restorable, while original WAV bytes can be re-downloaded by authorized reviewers.",
        '- Software validation:104 Python tests,2 Node tests, Ruff, JS syntax, receiver compilation and diff checks passed locally. Lightweight CI cannot load the private model; the actual196-input calibration gate and1,025 approved-prompt scores were run on k2.', '',
        'Regenerate the numbers with `PYTHONPATH=neptts-eval/src python3 benchmark/arena_d32_rescore_v1/analyze.py`, then this report with `python3 benchmark/arena_d32_rescore_v1/report.py`. These commands use public numeric evidence only. See [README.md](README.md) for artifact inventories/retention and [neptts-eval documentation](../../neptts-eval/README.md#private-d32-reviewer-judge) for private offline prerequisites. Initial effort estimate:8–16 agent-hours plus CPU/transfer waits, native human listening0; no human listening study was performed.', '',
        'Prepared2026-11-04UTC by Codex `/root`, job `arena_d32_rescore_v1`; owner Ampixa / Arbiter nenglish-73. Reference evidence for this PR; review by2026-11-11 or PR closure. NOT on HF. This document records its own provenance; source/report SHA256 inventory is in README.', ''
    ])
    text = '\n'.join(lines)
    for before, after in {
        'complete205': 'complete 205', 'step60000': 'step 60000', 'CPU1thread': 'CPU, one thread',
        'context−1/−1': 'context −1/−1', 'PV100': 'PV 100', 'Fresh96': 'Fresh 96',
        'PV7.': 'PV 7.', 'Fresh13.': 'Fresh 13.', 'difference0.05': 'difference 0.05',
        'All196': 'All 196', 'the21-module': 'the 21-module', 'all15': 'all 15',
        'have4PV and12Fresh': 'have 4 PV and 12 Fresh', 'speaker-bootstrap95%': 'speaker-bootstrap 95%',
        'same205': 'same 205', 'are95%': 'are 95%', 'paired95%': 'paired 95%',
        'the133-group': 'the 133-group', 'minus Aayush is−': 'minus Aayush is −',
        'and−2.16': 'and −2.16', 'Aayush6.86': 'Aayush 6.86', 'WiseAI_TTS7.22': 'WiseAI_TTS 7.22',
        'difference+0.36': 'difference +0.36', 'The615': 'The 615', 'all205': 'all 205',
        'designed205': 'designed 205', 'The82': 'The 82', 'seed20261104': 'seed 20261104',
        'resamples205': 'resamples 205', 'resamples133': 'resamples 133', '72pairs+61singletons': '72 pairs + 61 singletons',
        'all28': 'all 28', '|133-group': '| 133-group', 'worsened98/100PV and96/96Fresh': 'worsened 98/100 PV and 96/96 Fresh',
        'all196': 'all 196', 'redecoded21/205': 'redecoded 21/205', '(10.24%,105 total)': '(10.24%, 105 total)',
        'two32-input': 'two 32-input', 'OpenSLR43/54': 'OpenSLR 43/54', ':9.8%': ': 9.8%',
        'historical82': 'historical 82', '/50text': ' / 50 text', 'advertised64prompt': 'advertised 64 prompt',
        'removed:': 'removed: ', 'bytes:': 'bytes: ', 'increase:': 'increase: ',
        'validation:104': 'validation: 104', ',2 Node': ', 2 Node', 'actual196-input': 'actual 196-input',
        'and1,025': 'and 1,025', 'estimate:8–16': 'estimate: 8–16', 'listening0': 'listening 0',
        'Prepared2026-11-04UTC': 'Prepared 2026-11-04 UTC', 'by2026-11-11': 'by 2026-11-11',
        'PV[': 'PV [', 'Fresh[': 'Fresh [', 'meanCER': 'mean CER', 'corpusCER': 'corpus CER', '0.05pt': '0.05 pt', '0.62CER': '0.62 CER', '2.16WER': '2.16 WER', '0.36pt': '0.36 pt', 'exact205': 'exact 205', '95%CI': '95% CI', '95%CIs': '95% CIs', 'CI[': 'CI [', 'for4': 'for 4'
    }.items():
        text = text.replace(before, after)
    (HERE / 'REPORT.md').write_text(text)


if __name__ == '__main__':
    main()
