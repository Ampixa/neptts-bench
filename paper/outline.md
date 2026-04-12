# NepTTS-Bench: A Comprehensive Benchmark for Nepali Text-to-Speech Evaluation

## Paper Outline

### 1. Introduction
- No standardized benchmark exists for evaluating Nepali TTS systems
- Growing availability of Nepali TTS (Edge TTS, gTTS, Gemini, Piper, ElevenLabs) but no systematic comparison
- We present NepTTS-Bench: a phonologically-informed evaluation benchmark
- Contributions:
  1. Curated sentence set covering 66 minimal pairs across 9 phonological contrast categories
  2. Multi-system evaluation of N TTS systems + human speech baselines
  3. Automated + human MOS evaluation with cross-validation
  4. ASR round-trip intelligibility metric
  5. Open-source benchmark toolkit and rating app

### 2. Related Work
- TTS evaluation: MOS, MUSHRA, comparative evaluation
- Automated MOS prediction: UTMOS, SCOREQ, DNSMOS, IndicMOS
- Low-resource language TTS: challenges, existing benchmarks
- Nepali NLP landscape: resources, tools, prior TTS work
- Indic TTS: AI4Bharat, IndicVoices, IIT systems

### 3. NepTTS-Bench Design

#### 3.1 Sentence Set
- 193 benchmark sentences across 9 categories:
  - Phonological minimal pairs (108 sentences, 66 pairs)
    - Aspiration: velar, palatal, retroflex, dental, labial
    - Retroflex vs dental
    - Oral vs nasal vowels
    - Gemination (single vs doubled consonants)
    - Schwa deletion
    - Nasal consonants
  - Question intonation (10)
  - Emotion (8)
  - Contrastive stress (8)
  - Homographs (6)
  - Newspaper ambiguities (4)
  - Robustness / tongue twisters (10)
  - Phrases / greetings (10)

#### 3.2 Human Reference Audio
- Recording app: 35 speakers, 1,821 recordings from native Nepali speakers
- Natural speech: 160 utterances from Chirp2-transcribed YouTube Nepali speech
- Both included as blinded reference systems in MOS evaluation

#### 3.3 Evaluation Methodology
- **Human MOS**: Web-based rating app, 1-5 scale, blinded, randomized
  - Time-flexible (5/10/15/20 min)
  - Equal proportioning across systems
  - Text shown alongside audio
- **ASR Round-trip**: Whisper small → CER/WER
- **Automated MOS**: SCOREQ (NeurIPS 2024)
- Cross-validation: Spearman correlation between all three metrics

### 4. TTS Systems Evaluated

| System | Provider | Type | Voices |
|--------|----------|------|--------|
| Edge TTS (Hemkala) | Microsoft | Cloud neural | Female |
| Edge TTS (Sagar) | Microsoft | Cloud neural | Male |
| gTTS | Google Translate | Cloud | Default |
| Gemini 2.5 Flash | Google | Cloud neural | Kore |
| Piper | Open source | Local VITS | ne_NP-google-medium |
| [ElevenLabs] | ElevenLabs | Cloud neural | TBD |
| [OpenAI TTS] | OpenAI | Cloud neural | TBD |
| [AI4Bharat] | AI4Bharat | Open source | TBD |

### 5. Results

#### 5.1 Automated MOS (SCOREQ)
| System | SCOREQ MOS |
|--------|-----------|
| Gemini Flash | 4.10 |
| Edge TTS Hemkala | 3.74 |
| Edge TTS Sagar | 3.63 |
| Piper | 3.49 |
| gTTS | 3.02 |

#### 5.2 ASR Round-trip (Whisper small)
| System | CER | WER |
|--------|-----|-----|
| Gemini Flash | 0.368 | 0.940 |
| Edge TTS Sagar | 0.429 | 1.003 |
| Edge TTS Hemkala | 0.451 | 0.997 |
| gTTS | 0.644 | 1.193 |
| Piper | TBD | TBD |

#### 5.3 Human MOS
- TBD (collecting ratings)
- Per-system MOS with 95% CI
- Per-category breakdown
- Human ceiling (natural speech MOS)

#### 5.4 Cross-metric Correlation
- SCOREQ vs Human MOS (Spearman)
- ASR CER vs Human MOS
- Agreement analysis

#### 5.5 Phonological Analysis
- Per-contrast-category accuracy
- Which systems handle aspiration better?
- Retroflex vs dental discrimination
- Nasalization accuracy

### 6. Discussion
- Gemini leads across all metrics but is API-only
- Open-source options (Piper) lag significantly
- gTTS quality is notably poor for Nepali
- Edge TTS provides a solid mid-range option
- Human ceiling analysis: how close are best systems?
- Limitations of automated MOS for Nepali
- Recommendations for Nepali TTS development

### 7. Conclusion
- First comprehensive Nepali TTS benchmark
- Open-source toolkit for ongoing evaluation
- Identified quality gap between commercial and open-source
- Call for more Nepali speech data and model development

### 8. References

### Appendices
- A. Full sentence list
- B. Minimal pair inventory
- C. Rating app design
- D. Per-sentence scores
