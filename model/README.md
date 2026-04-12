# NepaliMOS — Nepali Speech Quality Predictor

Fine-tuned IndicWav2Vec base model for predicting Mean Opinion Score of Nepali speech.

## Architecture

- **Backbone:** IndicWav2Vec base (363M params, top 4 transformer layers unfrozen)
- **Head:** Linear(768, 256) → ReLU → Dropout(0.1) → Linear(256, 1)
- **Training data:** 5,760+ human ratings from 164 native Nepali speakers
- **Best Spearman:** 0.587

## Training

```bash
python train_nepali_mos.py \
  --ratings_db ratings.db \
  --tts_dir /path/to/tts_outputs \
  --output_dir ./checkpoints \
  --device cuda \
  --epochs 50 \
  --batch_size 2 \
  --lr 1e-4 \
  --unfreeze_layers 4 \
  --init_from_indicmos
```

## Requirements

```
torch
torchaudio
s3prl
huggingface_hub
soundfile
scipy
```

## Checkpoint

Download from [HuggingFace](https://huggingface.co/ampixa/neptts-bench).
