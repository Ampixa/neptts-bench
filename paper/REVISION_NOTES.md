# Revision notes since the initial draft

What changed in `paper/neptts_bench.tex` and the underlying analysis between the version Lekhnath reviewed and the current state.

The bibliography was the first concrete addition (commit `be14cc7`, Apr 18). Everything below is on top of that.

## Linguistic / phonological framing

The "four-way laryngeal contrast at five places of articulation" claim is now sourced and qualified. Inline citations `\citep{khatiwada2009, pokharel1989}` at §3.1 line 91. The original phrasing said "five places of articulation"; this is now "five stop and affricate series at the bilabial, dental, retroflex, palatal, and velar places of articulation" since the palatal series is affricate, not a stop. The full caveat paragraph at §3.1 explicitly notes:

- voiced aspirated (breathy) series is partially merged in casual speech
- retroflex–dental opposition tends to neutralize in coda position
- oral vs. nasal vowel and tatsama-driven gemination carry lower functional load in Standard Nepali than in Hindi
- breathy voice on sonorants (aspirated /m/, /n/, /l/) is *not* in the stimulus set; flagged for future work

Reframe at the end of §3.1: NepTTS-Bench is "an engineering benchmark with phonologically-motivated stimuli" rather than a phonological study. Linguistic-grade validation (trained-phonetician review of every pair, IPA transcription) is named as future work.

The categories table shrank to an appendix in the 8-page version (`neptts_bench_8pg.tex`, Appendix A). Counts unchanged: 6 minimal-pair categories, 7 additional categories, 205 sentences, 77 pairs.

A phantom citation was removed: `udupa2024indicmos` was not findable on arXiv or in the Interspeech 2024 archive. The dependent sentence in §2.1 was also dropped. Confirmed by `bd note nb-ec1.9`.

## Bibliography fixes

Concrete fixes to `paper/references.bib`:

- `khatiwada2009`: was `@phdthesis` with `school = {JIPA}`. Now `@article` with `journal`, `volume=39`, `number=3`, `pages=373--380`, `note={Illustration of the IPA}`.
- `pratap2023mms`: author list was wrong. `Tober → Tomasello`, `Kunber → Kundu`, `Babu, Anuroop → Babu, Arun`, `Graves, Alex` removed. Now lists all 16 authors verified against arXiv:2305.13516.
- `minixhofer2025ttsds`: title is "TTSDS2", body now says "TTSDS2 benchmark" instead of "TTSDS benchmark".
- Two new citations: `conneau2022fleurs` (FLEURS, IEEE SLT 2022) cited in §2.3 Low-Resource Language TTS; `huang2022voicemos` (VoiceMOS Challenge 2022, Interspeech) cited alongside UTMOS in §2.1.
- Three entries still need verification (out of scope for this round): `pokharel1989` exact title, `bandhu1971` co-author list, `gagan2023xlsr` author full name.

## Statistical rigor

Earlier draft: 95% CIs reported as `±0.10` etc. with no methodology stated. Krippendorff's α and significance tests absent.

Now:

- **Confidence intervals** in Table 2 are 95% cluster-bootstrap intervals over raters (2,000 resamples). Real intervals are 2–4× wider than the published Wald-style numbers. Top four TTS systems (Asmita 3.55, Piper 3.50, Subina 3.47, ElevenLabs v3 3.47) all have overlapping CIs and are not statistically separable.
- **Krippendorff's α** = 0.17 (interval), 0.16 (ordinal). Reported in the abstract, in §3.3 Inter-Rater Reliability, and in the Conclusion. This is below the conventional 0.667 threshold and below the 0.4–0.6 typical of lab MOS studies. The low α is itself written up as a finding rather than buried.
- **Steiger's Z** for "Chirp 2 vs Whisper-small in agreement with humans": Z = 1.09, p = 0.28. The paper now explicitly says we cannot statistically distinguish them at n = 9.
- **Benjamini–Hochberg correction** applied to Tab 4 (cross-metric correlation) asterisks. Only SCOREQ–Chirp 2 and Chirp 2–MMS survive at FDR 0.05. Was 3 pairs marked under raw p; now 2 under BH.
- **New Table 5** (`tab:metric_vs_human`): each automated metric's Spearman ρ with human MOS over n = 9 systems. NepaliMOS is now the only metric with raw p < 0.05; it also survives BH correction across the table. The ASR and SCOREQ rows remain nonsignificant.

The agreement-with-humans story is now reported separately from auto-vs-auto agreement. Earlier the paper conflated them: "Whisper unreliable (ρ = 0.38)" was actually ρ(Whisper, SCOREQ), not ρ(Whisper, human). Reframed throughout (abstract finding 3, §6.4, conclusion).

The full numerical analysis pipeline is at `scripts/compute_stat_rigor.py` (committed `3cd96b6`). It loads the production rating-app database snapshot (`benchmark/data/ratings_prod.db`, gitignored), computes everything above, and prints results. Reproducible end-to-end.

## Sample-size correction

The human-evaluation count is now honest: 6,962 native-rater ratings in the current snapshot of the rating-app DB, down from "7,003" in the initial draft. NepaliMOS predictor training is separately described as 1,697 matched TTS utterance-level targets derived from 5,865 matched-audio ratings (5,804 from native-speaker registrations). The 7,003 figure is retained only in the scaling table as an earlier pre-cleanup snapshot with a clarifying note.

Tab 2 means shifted by ≤0.02 with the corrected snapshot. The headline ranking is unchanged.

## NepaliMOS motivation and overfitting controls (Rupak review, 2026-04-29)

Rupak Ghimire flagged §4.6 as a likely reviewer objection: "How do you manage the overfitting issue with only 7K training samples?" and "If we are not using NepaliMOS, what is the point of mentioning it?" The first concern is addressed in §4.6 itself; the second is closed by applying NepaliMOS to the automated-metric table and metric-vs-human table (system-level ρ_human = 0.90 vs SCOREQ's 0.40). The draft now explicitly states that 0.90 is same-panel system-level agreement, not leave-one-system-out generalization.

§4.6 NepaliMOS Predictor expanded:

- **Motivation** rewritten. The earlier paragraph said "SCOREQ misranks systems → we train a Nepali-specific predictor." The new version names the representation gap explicitly: SCOREQ is trained on English speech-quality data and learns audio-fidelity features (signal cleanness, codec artifacts, additive noise) that are largely orthogonal to Nepali phonology (four-way laryngeal contrast, retroflex/dental opposition, schwa deletion). The same gap is expected to apply to UTMOS and DNSMOS, both English-trained. This frames why a small Nepali-supervised predictor is the right intervention rather than just a stronger general-purpose predictor.
- **New paragraph "Capacity and overfitting controls"** between Architecture/training and Results. Inventories the controls: 28.5M trainable parameters (top 4 of 12 transformer layers + 197K head) out of 95.0M backbone params total, differential LR (backbone 10⁻⁵, head 10⁻⁴), dropout 0.1 in the head, batch size 2 as implicit regularization, stratified 90/10 split by integer MOS bucket, and best-validation-loss checkpoint selection across 30 epochs (selects epoch 7; validation loss does not improve in the remaining 23 epochs, so the early-stopping criterion is binding). Two empirical signals support the no-memorization conclusion: (i) utterance-level Spearman ρ = 0.59 on n = 167 held-out utterances unseen during training (a memorizing model would predict near the training-set mean and yield ρ ≈ 0); (ii) earlier scaling snapshots show monotonic improvement in ρ as supervision grows. The final prose now avoids claiming those snapshots are the identical cleaned checkpoint.
- Param counts verified directly from the saved checkpoint (`model/checkpoints/neptts_mos_best.pt`): SSL backbone 95,044,608; top 4 encoder layers 28,351,488; head 197,121; trainable total 28,548,609; best epoch 7; n_train 1,530; n_val 167; val MSE 0.5804 (RMSE 0.762); val Spearman 0.589 (rounds to 0.59 in paper).

A train-vs-validation loss curve figure is the natural complement and is filed as future work; the current training script does not log per-epoch losses to disk and re-running on the GTX 1650 (4 GB) is tight at this batch size. The prose case (held-out ρ on unseen utterances + monotonic data-scaling) carries the no-overfit argument without the figure.

## NepaliMOS leave-one-system-out validation (2026-05-31)

Added `scripts/run_nepalimos_loso.py` and ran the strict LOSO experiment over all 10 matched TTS systems. Each fold holds out one system completely, retrains NepaliMOS from IndicWav2Vec on the remaining systems with top-four-layer unfreezing, batch size 2, differential learning rates, and internal stratified validation, then restores the best validation-loss checkpoint (30-epoch cap, patience 6) for held-out testing.

Output artifact: `benchmark/results/nepalimos_loso.json`.

Key results:

- All 10 systems, utterance level: Spearman ρ = 0.429, Pearson r = 0.426, n = 1,697, MAE = 0.687, RMSE = 0.861.
- All 10 systems, held-out system means against audio-label means: Spearman ρ = 0.018, p = 0.960.
- TTS-9 subset excluding Gemini Hindi, utterance level: Spearman ρ = 0.418, n = 1,558.
- TTS-9 subset against native-human system means: Spearman ρ = -0.333, p = 0.381.

Interpretation: the same-panel ρ = 0.90 is useful for showing that language-matched supervision recovers the evaluated panel ranking, but it does not transfer to calibrated held-out system ranking. The manuscript now reports LOSO as a negative/generalization-limiting result rather than leaving it as future work. NepaliMOS is framed as a useful utterance-level signal and same-panel analysis tool, not as a final MOS for new systems without calibration.

Follow-up unseen-system check (May 7):

- Investigated the existing `mms_tts` path first. The script referenced `facebook/mms-tts-npi`, but that repo does not exist, and the public `facebook/mms-tts` tree did not contain `models/npi` in the Hugging Face snapshot. The paper now avoids claiming that the public MMS-TTS release exposes Nepali TTS.
- Added a reproducible public checkpoint instead: `atul10/nepali_male_v1` (Apache-2.0 VITS; `procit001/nepali_male_v1` redirects there). Generated all 205 numeric NepTTS-Bench stimuli to `/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs/nepali_male_v1`.
- Ran NepaliMOS on that unseen system only. Result: mean predicted MOS `2.570`, std `0.347`, n `205`, saved in `benchmark/results/nepalimos_unseen_nepali_male_v1.json`.
- This is explicitly written as a smoke check, not validation. It proves the scorer can run on a checkpoint absent from the rated panel, but because no human MOS exists for this model it is excluded from Spearman/Steiger tables. A small human calibration set remains required for strong out-of-panel system-level claims.

## Methodological additions

- §3.2 TTS configuration table (`tab:tts_configs`) with explicit voice IDs, model versions, prompt strings, and output formats for all 10 TTS systems. Lekhnath had asked about Gemini-Hindi vs Gemini Flash specifically; this is now nailed down: Gemini Flash uses a Nepali-naming prompt prefix, Gemini (Hindi) uses raw Devanagari with no language hint. Both run `gemini-2.5-flash-preview-tts` voice "Kore". The audio sets are distinct (verified against the original `gemini/` and `gemini_hindi/` directories at `~/gt/bolne/crew/bolne/benchmark/data/tts_outputs/`).
- §3.3 ASR Round-Trip paragraph names every ASR's configuration: Chirp 2 (Speech-to-Text v2 API, model `chirp_2`, language `ne-NP`), MMS (`facebook/mms-1b-all` with Nepali adapter), XLS-R (`gagan3012/wav2vec2-xlsr-nepali` greedy CTC), NepConformer CTC-BPE (NeMo `EncDecCTCModelBPE`, 128-token SentencePiece tokenizer, citing Poudel et al. 2026), Whisper-small (`language="ne"`, fp32), Whisper large-v2 via faster-whisper, and Whisper large-v3-turbo via MLX.
- §5.4 new subsection documenting Whisper large-v3-turbo's hallucination behavior on Nepali. 2–10% of files per system hit CER > 1.0; max CERs reach 186 on Edge Hemkala and 1,115 on gTTS. Earlier draft claimed "median CER above 1.0 on four of nine systems" — this was wrong (median is 0.15–0.44 on every system). Corrected against the actual JSON data.
- Tab 3 now includes the usable ASR columns, including Whisper-small, Whisper large-v2, and the newly added NepConformer CTC-BPE run. Abstract/method claims now use "six usable ASRs" consistently.
- Tab 3 (ranking divergence) was using mixed ranking schemes for the Human column. ElevenLabs row used all-12 ranking, Asmita row used TTS-only ranking, gTTS row used yet another. Rebuilt with consistent TTS-9 ranking; ΔRank values shifted slightly (ElevenLabs −5 → −3, Piper unchanged at +6).
- §3.3 Statistical Inference paragraph now states the clustering scheme and the implications. The earlier "we don't report Krippendorff α" hedge is replaced by the actual α = 0.17.

## Limitations expansion

§Limitations was one paragraph in the initial draft. Now nine named paragraphs (full version in `neptts_bench_8pg.tex` Appendix E):

1. Rater pool and demographics (no demographic metadata collected)
2. Listening environment (no headphone or P.808 controls)
3. Rater fatigue and order effects (drift across session not audited)
4. Same-system bias (voice characteristics may identify systems despite blinding)
5. Statistical inference and IRR (Wald assumption; Krippendorff α reported)
6. NepaliMOS predictor (LOSO-CV run; weak held-out system ranking)
7. Pair discrimination (n = 54 < 1 rating per pair; no statistical conclusions drawn)
8. Automated MOS scope (only SCOREQ tested; UTMOS, DNSMOS not run)
9. Configuration sensitivity (results bound by Tab 1 settings)

## Conflict-of-interest statement

New §"Conflicts of Interest" after the Conclusion. Names Ampixa Labs as the affiliation, declares no commercial relationship with TingTing.io / ElevenLabs / Microsoft / Google / Piper community, no provider funding, independent system selection / rater recruitment / stimulus design, open-source release for verification.

## Page layout

Two versions in the repo:

- `paper/neptts_bench.tex` — master, currently 16 pages, all detail in body.
- `paper/neptts_bench_8pg.tex` — compressed draft, currently 12 pages including references and appendices. It still needs trimming before a strict 8-page submission target.

Author field in both versions reads `Draft passing to collaborators` for this round of sharing.

## What I still need from a linguist

These are the specific places where domain knowledge I do not have would change the paper:

1. **Phonetic validation of the 77 minimal pairs.** A trained phonetician's pass over the pair set, ideally with IPA transcription replacing the current ISO 15919-ish romanization (`k\=am` / `kh\=am`).
2. **Breathy voice on sonorants.** Whether to add `/m mʱ/`, `/n nʱ/`, `/l lʱ/` minimal pairs. Listed as future work but if the answer is "yes obviously add these" we can do it before submission.
3. **Schwa deletion modeling.** §3.1 says "Nepali schwa deletion differs from Hindi" but does not engage with Bal Krishna Bal or Yadava. If you have a preferred reference for the Nepali-specific deletion rules we should cite it directly.
4. **Categories with low functional load.** Whether to drop oral/nasal vowel and gemination from the headline contrast count or keep them with the merger caveat.
5. **Bibliography entries that need verification:** the exact `pokharel1989` work being cited (the bib entry currently says "Nepali Linguistics" but Pokharel's better-known phonological reference is *Experimental Analysis of Nepali Sound System*, Deccan College PhD), and the full author list for `bandhu1971` (currently uses "and others").
6. **Native-speaker definition.** The benchmark treats "native Nepali speaker" as self-declared. Whether the rater pool should be split by L1 (Khas-Kura vs. L1 Maithili / Bhojpuri / Newar / Tamang) for separate analysis.

Concrete pointers for each are in the GitHub repo. Happy to address any of these in the next revision.
