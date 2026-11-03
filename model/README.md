# NepaliMOS — Nepali Speech Quality Predictor

NepaliMOS predicts mean opinion scores using an IndicWav2Vec backbone and a regression head. It is an auxiliary research metric. Same-panel agreement does not establish calibrated ranking of unseen TTS systems.

## Architecture and study scope

- IndicWav2Vec base representation; the training recipe unfreezes the top four transformer layers.
- Head: Linear(768, 256) → ReLU → Dropout(0.1) → Linear(256, 1).
- The [paper](../paper/neptts_bench.tex) distinguishes its **6,962-rating / 180-contributor human study** from the predictor's **1,697 matched TTS utterance targets**, derived from 5,865 matched-audio ratings (5,804 from native-speaker registrations). These totals are not interchangeable training-set sizes.
- Reported held-out-utterance Spearman ρ ≈ 0.59 and same-panel system ρ = 0.90 have different scopes. [Leave-one-system-out evidence](../benchmark/results/nepalimos_loso.json) gives pooled utterance ρ = 0.429 and native-human system-mean ρ = −0.394 (10 systems, p = 0.260).

## Checkpoints and actual default

The [evaluator constant](../neptts-eval/src/neptts_eval/nepalimos_eval.py) selects **`model/neptts_mos_v9_best.pt`**, a head-only checkpoint, from the **dataset repository** [ampixa/neptts-bench](https://huggingface.co/datasets/ampixa/neptts-bench/tree/d07d5495968f5e1ec4e18e852bb926f8babc1335/model). The loader uses the vanilla `SYSPIN/IndicMOS` backbone when no `ssl_state_dict` is present. Its prediction scale differs from a fully fine-tuned model; do not compare these predictions silently with the saved full-model baselines.

Public metadata at revision `d07d5495968f5e1ec4e18e852bb926f8babc1335` lists:

| File | Bytes | SHA256 (LFS object) | Role |
|------|------:|--------------------|------|
| `model/neptts_mos_v9_best.pt` | 791,435 | `4bcec9e617446ceea4fccbfa84e8e3a4315b1c2b6057c14006a262e3dde781c5` | Current head-only default |
| `model/neptts_mos_best.pt` | 381,055,459 | `96579da41a3cbd20406e2f4980bc9a391540e1f868354fd478c2a8031ea48fdc` | Public full-state candidate; explicit local override |

The full-state candidate is publicly listed. This documentation update did **not** download/load either state, verify tensor/config compatibility, or rerun its score calibration. Metadata and filenames alone do not prove paper reproduction. After verifying a local file and its intended calibration, select it with `--nepalimos-ckpt /path/to/neptts_mos_best.pt` or `NEPALIMOS_CKPT`; the CLI flag takes precedence. Downloads currently follow the repository default revision, so record the resolved revision and file hash for reproducible results.

Install the optional predictor dependencies from the repository root:

```bash
python -m pip install -e './neptts-eval[nepalimos]'
neptts-eval --wav-dir ./my_outputs/ --nepalimos-ckpt /path/to/verified_checkpoint.pt
```

The first inference also needs the `SYSPIN/IndicMOS` `indicw2v_base_pretrained.pt` backbone. Keep model/source licences and permissions with any retained copy.

## Training

```bash
python model/train_nepali_mos.py \
  --ratings_db /path/to/authorized/ratings.db \
  --tts_dir /path/to/tts_outputs \
  --output_dir ./checkpoints \
  --device cuda \
  --epochs 50 \
  --batch_size 2 \
  --lr 1e-4 \
  --unfreeze_layers 4 \
  --init_from_indicmos
```

Training requires authorized private study inputs; they are not supplied by installing the evaluator. See [release scopes](../neptts-eval/RELEASE_SCOPE.md) for the distinction between the historical study, stored media and current arena.
