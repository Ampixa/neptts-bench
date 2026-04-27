# NepTTS-Bench Pre-Submission AI Review — Synthesis

Generated 2026-04-27 from a 5-reviewer simulated panel (EIC, R1 Methodology, R2 Domain, R3 Perspective, Devil's Advocate). Drives the multi-session revision plan tracked in beads under parent `nb-ec1`.

## Editorial Decision

**Major revision required** before submission. Target venue: Interspeech / LREC-COLING. Do not target IEEE TASLP — modeling contribution is too light for that scope.

The contribution is genuine and the artifact is valuable, but the paper as written cannot survive adversarial review without substantial methodological tightening. The single most damaging finding (raised independently by R1 and Devil's Advocate) is that with **n=10 systems, the headline ρ=0.66 NepaliMOS result has p≈0.04 and a CI spanning roughly [0, 0.9]**, and the train/test partitioning is undocumented. The same 207 raters generated both training labels and the test ratings the predictor is evaluated against. This alone could sink the paper at conference Q&A.

## Reviewer Verdicts

| Reviewer | Verdict | Headline concern |
|---|---|---|
| EIC | Minor revision | Inter-rater reliability + bootstrap CIs needed; pair-discrimination underpowered; COI statement needed |
| R1 Methodology | Major revision | n=10 ρ=0.66 not significant; CI method undocumented; no Steiger's Z on ρ comparisons; reproducibility 2/5 |
| R2 Domain | Major revision | "4×5 grid" overstates real Nepali; bib has 6+ errors; missing FLEURS / VoiceMOS / Acharya 1991 / Yadava 2003 |
| R3 Perspective | Minor revision | MOS-as-ground-truth unjustified; "Standard Nepali" excludes actual users; no adoption strategy |
| Devil's Advocate | Bombs | "Automated MOS predictors unreliable" generalized from n=1 (only SCOREQ run); CI overlap on rankings; ρ=0.05 is noise not "bias"; Gemini (Hindi) listed as Nepali TTS |

## Consensus Issues (3+ of 5 reviewers)

| # | Issue | Raised by | Severity |
|---|---|---|---|
| 1 | No inter-rater reliability (Krippendorff α / ICC) | EIC, R1, R3, Devil's | P0 |
| 2 | CIs computed without cluster-robust SEs (34 ratings/rater) | R1, Devil's | P0 |
| 3 | ρ comparisons not significance-tested (Steiger's Z absent) | R1, Devil's | P0 |
| 4 | TTS / ASR configurations missing | R1, EIC | P0 |
| 5 | NepaliMOS train/test split undocumented; likely overlap | R1, Devil's | P0 |
| 6 | Conflict-of-interest disclosure missing | EIC, Devil's | P0 |
| 7 | Pair discrimination (54 ratings) underpowered but central to framing | EIC, R1, Devil's | P1 |

## Devil's Advocate Bombs (issues unique to one reviewer but high impact)

- "Automated MOS predictors are unreliable for Nepali" generalizes from n=1 (only SCOREQ run). UTMOS and DNSMOS are cited but never executed. → Either run them, or hedge to "SCOREQ specifically".
- Gemini (Hindi) listed as a Nepali TTS, then admitted to be Hindi-accented, then included in the ranking anyway.
- Tab 3 caption says "three best-performing ASR systems" but abstract says 5; quietly drops 2 ASRs. Tab 4 has 7 systems vs Tab 2's 12. Counts inconsistent throughout.
- ρ=0.05 with n=10 has p≈0.89 ("zero evidence of disagreement, just noise") yet paper interprets it as "bias toward particular pronunciation patterns."
- Scaling law claim confounds rating density with dataset size: 1,166 → 1,696 unique samples, but ratings grow 1,166 → 7,003 — later rows have 4 ratings/sample, earlier rows have 1.
- TingTing Asmita 3.55 ±0.09 vs ElevenLabs 3.47 ±0.11 — every CI overlaps. No paired test.

## R2 Domain-Expert Findings (Phonology)

- "Four-way laryngeal contrast at five places of articulation" overstates: voiced-aspirated series is partially merged in casual Nepali; "five places" should be "five stop/affricate series" (palatal is affricate not stop).
- Retroflex/dental neutralizes in coda; minimal pairs should target onset position only — disclose.
- Nasal vowels have low functional load; Kathmandu younger speakers underdifferentiate.
- Gemination is loanword-conditioned (Sanskrit/tatsama), not core Nepali.
- Schwa deletion in Nepali differs from Hindi; Bal Krishna Bal / Yadava work uncited.
- Breathy voice on sonorants (/m mʱ/, /n nʱ/, /l lʱ/) missing from contrast categories.

## R2 Bibliography Issues (concrete fixes)

- `khatiwada2009` is `@phdthesis` but should be `@article` (JIPA vol 39(3), pp 373–380).
- `pratap2023mms` author list has "Tober" (likely Tobin) and "Kunber" (likely Kumar) — verify.
- `pokharel1989` title may be wrong; the foundational Pokharel work is *Experimental Analysis of Nepali Sound System* (Deccan College).
- `bandhu1971` uses "and others" — give full author list.
- `gagan2023xlsr` author is just "Gagan"; HF model card is not a stable citation.
- `udupa2024indicmos` uses "and others" — list co-authors.
- `minixhofer2025ttsds` title says TTSDS2 but body says TTSDS; reconcile.

## R2 Citation Gaps

- Acharya, J. (1991) *A Descriptive Grammar of Nepali* (Georgetown UP)
- FLEURS (Conneau et al., 2022); XTREME-S (Conneau et al., 2022)
- VoiceMOS Challenge 2022 / 2023 (Huang et al.; Cooper et al.)
- TTS-Portuguese / SOMOS (Maniati et al., 2022)
- Indic Parler-TTS / AI4Bharat IndicTTS (Kumar et al., 2023)
- Yadava, Y. P. (2003) "Language" in Population Monograph of Nepal — for the 32M-speaker claim
- Bal Krishna Bal et al. (Madan Puraskar Pustakalaya / PAN-Localization) for Nepali NLP / schwa deletion

## R3 Socio-Technical Concerns

- MOS as ground truth never justified; preference / CMOS / comprehension-based eval are the moving target.
- "Standard Nepali" stimuli exclude the actual TTS users (rural, L2, Maithili/Bhojpuri/Tharu speakers).
- 77 minimal pairs as ecologically valid: questionable. Intelligibility-under-load would be more relevant.
- No adoption strategy: hosted leaderboard, deployment-tier eval (latency, cost), outreach to ABC News Nepal / Hamro Patro / vision-impaired user community.
- Future-proofing: lead with what ages well (stimulus set, 7,003-rating dataset, predictor recipe, ASR-choice finding); de-emphasize absolute system numbers.

## Prioritized Revision Roadmap

### P0 — Blockers

| ID | Action | bd issue |
|---|---|---|
| P0.1 | Inter-rater reliability: Krippendorff's α (interval) + ICC(2,k) with bootstrap CIs | nb-ec1.2 |
| P0.2 | State CI methodology; switch to cluster-bootstrap (cluster by rater AND sentence) | nb-ec1.2 |
| P0.3 | Steiger's Z (or paired bootstrap) for ρ comparisons; Benjamini–Hochberg on Tab 5 asterisks | nb-ec1.2 |
| P0.4 | TTS Configuration table: voice IDs, model versions, parameters, sampling rates | nb-8cm |
| P0.5 | Document NepaliMOS train/val/test split; report leave-one-system-out CV ρ; clarify system-level vs utterance-level ρ | nb-ec1.2 |
| P0.6 | Conflict-of-interest / independence statement | new |
| P0.7 | Reconcile internal count contradictions (5 ASRs vs 3; 12 vs 10 vs 7 systems across tables) | new |
| P0.8 | Run UTMOS + DNSMOS (n=3 predictors) or hedge "automated MOS predictors" → "SCOREQ specifically" | new |
| P0.9 | Mechanical bib fixes (khatiwada2009 → @article; pratap2023mms author list; etc.) | new |

### P1 — Significant

| ID | Action | bd issue |
|---|---|---|
| P1.1 | Soften phonology framing: engineering benchmark with motivated stimuli, not Nepali linguistic contribution | new |
| P1.2 | Drop pair-discrimination claims (n=54 < 1 rating per pair) or add power analysis | nb-ec1.5 |
| P1.3 | Rater demographics: geography, dialect, L1, headphones, IRB/consent, compensation | nb-ec1.3 |
| P1.4 | Add missing citations (Acharya 1991, FLEURS, VoiceMOS Challenges, IndicTTS, Yadava 2003) | new |
| P1.5 | Bracket Gemini (Hindi) as control or remove from Nepali ranking | new |
| P1.6 | Hedge "first Nepali MOS predictor" or strengthen with leave-one-system-out CV | new |

### P2 — Polish

| ID | Action |
|---|---|
| P2.1 | IPA transcription instead of ISO 15919 romanization |
| P2.2 | Justify MOS-vs-CMOS choice |
| P2.3 | Soften scaling-law extrapolation; drop "data not architecture is the bottleneck" |
| P2.4 | Cross-disciplinary connections (accessibility, EdTech, sociolinguistics) |
| P2.5 | Adoption section: hosted leaderboard, deployment-tier eval, outreach plan |

## Reviewer Output Files

Full text of each review is in the conversation transcript. Key quotes are preserved above.
