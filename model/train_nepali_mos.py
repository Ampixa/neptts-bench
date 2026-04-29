#!/usr/bin/env python3
"""Fine-tune IndicMOS on Nepali human ratings from NepTTS-Bench.

Architecture: IndicWav2Vec base (frozen) → MLP head (trainable) → MOS [1-5]

Usage:
    python train_nepali_mos.py --ratings_db /path/to/ratings.db --tts_dir /path/to/tts_outputs/

Requirements:
    pip install torch torchaudio s3prl huggingface_hub
"""

import argparse
import json
import os
import random
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
from huggingface_hub import hf_hub_download

REPO_ID = "SYSPIN/IndicMOS"
SSL_NAME = "indicw2v_base_pretrained.pt"
BASE_PREDICTOR = "joint_indicw2v_base.pt"


class NepaliMOSDataset(Dataset):
    """Dataset of (audio_path, mos_score) pairs from rating app."""

    def __init__(self, samples: list[dict], max_len: int = 16000 * 15):
        self.samples = samples
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load_audio(path):
        """Load audio via soundfile, fallback to ffmpeg for MP3/webm."""
        import soundfile as sf
        try:
            data, sr = sf.read(str(path), dtype="float32")
            t = torch.from_numpy(data)
            if t.ndim == 1:
                t = t.unsqueeze(0)
            else:
                t = t.T
            return t, sr
        except Exception:
            import subprocess
            ffmpeg_bin = "/tmp/ffmpeg" if os.path.exists("/tmp/ffmpeg") else "ffmpeg"
            cmd = [ffmpeg_bin, "-i", str(path), "-f", "s16le", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", "-"]
            r = subprocess.run(cmd, capture_output=True, timeout=30)
            audio = np.frombuffer(r.stdout, dtype=np.int16).astype(np.float32) / 32768.0
            return torch.from_numpy(audio).unsqueeze(0), 16000

    def __getitem__(self, idx):
        s = self.samples[idx]
        audio, sr = self._load_audio(s["audio_path"])
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio.squeeze(0)
        # Truncate/pad
        if audio.shape[0] > self.max_len:
            audio = audio[: self.max_len]
        return audio, torch.tensor(s["mos"], dtype=torch.float32)


def collate_fn(batch):
    audios, scores = zip(*batch)
    lengths = [a.shape[0] for a in audios]
    max_len = max(lengths)
    padded = torch.zeros(len(audios), max_len)
    for i, a in enumerate(audios):
        padded[i, : a.shape[0]] = a
    return padded, torch.stack(scores), torch.tensor(lengths)


class NepaliMOSPredictor(nn.Module):
    """IndicWav2Vec base + MLP head for MOS prediction."""

    def __init__(self, ssl_model, hidden_dim: int = 256, unfreeze_layers: int = 0):
        super().__init__()
        self.ssl_model = ssl_model
        self.unfreeze_layers = unfreeze_layers

        # Freeze entire backbone first
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        # Unfreeze top N transformer layers if requested
        if unfreeze_layers > 0:
            # s3prl wav2vec2 has model.encoder.layers
            try:
                encoder_layers = self.ssl_model.model.encoder.layers
                total = len(encoder_layers)
                for layer in encoder_layers[total - unfreeze_layers:]:
                    for param in layer.parameters():
                        param.requires_grad = True
                print(f"  Unfroze top {unfreeze_layers}/{total} transformer layers")
            except AttributeError:
                # Try alternative path
                try:
                    encoder_layers = self.ssl_model.model.model.encoder.layers
                    total = len(encoder_layers)
                    for layer in encoder_layers[total - unfreeze_layers:]:
                        for param in layer.parameters():
                            param.requires_grad = True
                    print(f"  Unfroze top {unfreeze_layers}/{total} transformer layers")
                except AttributeError:
                    print("  WARNING: Could not find transformer layers to unfreeze")

        self.head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, lengths=None):
        if self.unfreeze_layers > 0:
            feats = self.ssl_model(x)["hidden_states"][-1]
        else:
            with torch.no_grad():
                feats = self.ssl_model(x)["hidden_states"][-1]

        # Mean pooling with length masking
        if lengths is not None:
            max_t = feats.shape[1]
            ratio = x.shape[1] / max_t
            ssl_lengths = (lengths.float() / ratio).long().clamp(min=1, max=max_t)
            mask = (
                torch.arange(max_t, device=x.device).unsqueeze(0)
                < ssl_lengths.unsqueeze(1)
            ).float()
            feats = (feats * mask.unsqueeze(-1)).sum(1) / mask.sum(-1, keepdim=True)
        else:
            feats = feats.mean(1)

        return self.head(feats).squeeze(-1)


def load_ratings(db_path: str, tts_dir: str) -> tuple[list[dict], dict]:
    """Load human ratings from the rating app DB, match to audio files.

    Returns (samples, rater_map) where rater_map maps sample index to rater_ids.
    """
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row

    # Get individual ratings (not averaged) to track raters
    # Exclude Admin ratings if --exclude_admin flag is set
    exclude_clause = "WHERE rater_id NOT IN (SELECT id FROM raters WHERE name LIKE 'Admin%')" if os.environ.get("EXCLUDE_ADMIN") else ""
    rows = db.execute(f"""
        SELECT system_name, sent_id, AVG(score) as avg_score, COUNT(*) as n_ratings,
               GROUP_CONCAT(rater_id) as rater_ids
        FROM ratings
        {exclude_clause}
        GROUP BY system_name, sent_id
        HAVING n_ratings >= 1
    """).fetchall()

    tts_base = Path(tts_dir)
    samples = []

    for r in rows:
        sys_name = r["system_name"]
        sent_id = r["sent_id"]
        mos = r["avg_score"]

        # Find audio file
        for ext in (".wav", ".mp3", ".flac", ".webm"):
            audio_path = tts_base / sys_name / f"{sent_id}{ext}"
            if audio_path.exists():
                raters = set(r["rater_ids"].split(",")) if r["rater_ids"] else set()
                samples.append({
                    "audio_path": str(audio_path),
                    "system": sys_name,
                    "sent_id": sent_id,
                    "mos": mos,
                    "n_ratings": r["n_ratings"],
                    "raters": raters,
                })
                break

    db.close()
    return samples


def train(args):
    device = torch.device(args.device)

    # Load data
    print("Loading ratings...")
    samples = load_ratings(args.ratings_db, args.tts_dir)
    print(f"  {len(samples)} rated audio files")

    if len(samples) < 50:
        print(f"WARNING: Only {len(samples)} samples. Need at least 200 for reliable training.")
        print("Collect more ratings at tts.ampixa.com/rating")
        if len(samples) < 20:
            print("Too few samples. Exiting.")
            return

    # Stratified split by MOS score — ensures val has diverse scores
    random.seed(42)

    # Bucket by score range
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, s in enumerate(samples):
        bucket = max(1, min(5, round(s["mos"])))
        buckets[bucket].append(i)

    val_set = set()
    for bucket, indices in buckets.items():
        random.shuffle(indices)
        n_val = max(1, len(indices) // 10)  # 10% per bucket
        val_set.update(indices[:n_val])

    val_samples = [samples[i] for i in sorted(val_set)]
    train_samples = [samples[i] for i in range(len(samples)) if i not in val_set]

    val_scores = [s["mos"] for s in val_samples]
    print(f"  Val score range: {min(val_scores):.1f} - {max(val_scores):.1f} (std={np.std(val_scores):.2f})")
    print(f"  Train: {len(train_samples)}, Val: {len(val_samples)}")

    train_ds = NepaliMOSDataset(train_samples)
    val_ds = NepaliMOSDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, collate_fn=collate_fn)

    # Load IndicWav2Vec backbone
    print("Loading IndicWav2Vec backbone...")
    import s3prl.hub as hub

    ssl_path = hf_hub_download(repo_id=REPO_ID, filename=SSL_NAME)
    ssl_model = getattr(hub, "wav2vec2_custom")(ckpt=ssl_path)
    print("  Backbone loaded")

    # Optionally initialize from pretrained IndicMOS head
    model = NepaliMOSPredictor(ssl_model, hidden_dim=args.hidden_dim, unfreeze_layers=args.unfreeze_layers)

    if args.init_from_indicmos:
        print("  Initializing from pretrained IndicMOS head...")
        predictor_path = hf_hub_download(repo_id=REPO_ID, filename=BASE_PREDICTOR)
        pretrained = torch.load(predictor_path, map_location="cpu")
        # Load the linear layer weights as initialization for first layer of our MLP
        model.head[0].weight.data[:768] = pretrained["linear.weight"].repeat(args.hidden_dim // 1 + 1, 1)[:args.hidden_dim]

    model = model.to(device)

    # Differential learning rates: backbone gets 10x lower lr
    backbone_params = [p for n, p in model.ssl_model.named_parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    n_backbone = sum(p.numel() for p in backbone_params)
    n_head = sum(p.numel() for p in head_params)
    print(f"  Trainable: backbone={n_backbone:,} head={n_head:,} total={n_backbone+n_head:,}")

    if backbone_params:
        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": args.lr / 10},
            {"params": head_params, "lr": args.lr},
        ])
    else:
        optimizer = torch.optim.Adam(head_params, lr=args.lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0
        for audio, scores, lengths in train_loader:
            audio, scores, lengths = audio.to(device), scores.to(device), lengths.to(device)
            pred = model(audio, lengths)
            loss = criterion(pred, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for audio, scores, lengths in val_loader:
                audio, scores, lengths = audio.to(device), scores.to(device), lengths.to(device)
                pred = model(audio, lengths)
                val_loss += criterion(pred, scores).item()
                val_preds.extend(pred.cpu().numpy())
                val_targets.extend(scores.cpu().numpy())

        val_loss /= len(val_loader)

        # Correlation
        from scipy.stats import spearmanr, pearsonr
        spearman = spearmanr(val_targets, val_preds)[0]
        pearson = pearsonr(val_targets, val_preds)[0]

        print(f"  Epoch {epoch+1}/{args.epochs}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} spearman={spearman:.3f} pearson={pearson:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            # Save best model
            out_path = Path(args.output_dir) / "neptts_mos_best.pt"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            # Save full state: head + any unfrozen backbone params, so
            # inference can reproduce the val ρ from the saved artifact alone.
            backbone_state = {
                k: v for k, v in model.ssl_model.state_dict().items()
                if any(p.requires_grad for n, p in model.ssl_model.named_parameters() if n == k or k.startswith(n.rsplit(".", 1)[0]))
            } if args.unfreeze_layers > 0 else {}
            # Simpler: save the whole ssl_model state dict if any unfreezing happened.
            full_ssl = model.ssl_model.state_dict() if args.unfreeze_layers > 0 else None
            torch.save({
                "head_state_dict": model.head.state_dict(),
                "ssl_state_dict": full_ssl,
                "unfreeze_layers": args.unfreeze_layers,
                "hidden_dim": args.hidden_dim,
                "val_loss": val_loss,
                "spearman": spearman,
                "pearson": pearson,
                "epoch": epoch + 1,
                "n_train": len(train_samples),
                "n_val": len(val_samples),
            }, out_path)

    print(f"\nBest model at epoch {best_epoch} (val_loss={best_val_loss:.4f})")
    print(f"Saved to {args.output_dir}/neptts_mos_best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratings_db", required=True, help="Path to ratings.db from rating app")
    parser.add_argument("--tts_dir", required=True, help="Path to tts_outputs/ directory")
    parser.add_argument("--output_dir", default="./checkpoints", help="Output directory")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--unfreeze_layers", type=int, default=0, help="Number of top transformer layers to unfreeze (0=frozen)")
    parser.add_argument("--init_from_indicmos", action="store_true", help="Initialize from pretrained IndicMOS")
    args = parser.parse_args()
    train(args)
