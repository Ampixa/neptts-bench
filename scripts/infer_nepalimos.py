"""Run NepaliMOS predictor on TTS audio and write per-system mean scores.

Inputs:
  - Checkpoint: model/checkpoints/neptts_mos_best.pt (full fine-tuned state,
    also hosted at huggingface.co/datasets/ampixa/neptts-bench)
  - Audio root: /home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs/

Output: benchmark/results/nepalimos_predictions.json
        Two views:
          - per_file: {system: {sent_id: predicted_mos}}
          - per_system: {system: {mean, n, std}}
"""
import json
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "model"))

CKPT = ROOT / "model" / "checkpoints" / "neptts_mos_best.pt"
AUDIO_ROOT = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs")
OUT = ROOT / "benchmark" / "results" / "nepalimos_predictions.json"

SYSTEMS = [
    "edge_tts/hemkala",
    "edge_tts/sagar",
    "gtts",
    "gemini",
    "gemini_hindi",
    "piper",
    "tingting_asmita",
    "tingting_sambriddhi",
    "tingting_subina",
    "elevenlabs",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=CKPT)
    parser.add_argument("--audio-root", type=Path, default=AUDIO_ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument(
        "--systems",
        nargs="+",
        default=SYSTEMS,
        help="System directories relative to --audio-root.",
    )
    return parser.parse_args()


def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    import soundfile as sf
    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception:
        # MP3/m4a fallback via torchaudio + ffmpeg
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        data = wav.mean(0).numpy()
    if data.ndim > 1:
        data = data.mean(axis=1)
    t = torch.from_numpy(data.astype(np.float32))
    if sr != target_sr:
        import torchaudio
        t = torchaudio.functional.resample(t, sr, target_sr)
    # Cap at 15 seconds (matches training)
    if t.shape[0] > 16000 * 15:
        t = t[: 16000 * 15]
    return t


class NepaliMOSPredictor(nn.Module):
    def __init__(self, ssl_model, hidden_dim: int = 256):
        super().__init__()
        self.ssl_model = ssl_model
        for p in self.ssl_model.parameters():
            p.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        feats = self.ssl_model(x)["hidden_states"][-1]
        feats = feats.mean(1)
        return self.head(feats).squeeze(-1)


def build_ssl():
    """Download IndicWav2Vec base from SYSPIN/IndicMOS and load via s3prl."""
    from huggingface_hub import hf_hub_download
    print("Downloading IndicWav2Vec backbone from SYSPIN/IndicMOS...")
    path = hf_hub_download(repo_id="SYSPIN/IndicMOS", filename="indicw2v_base_pretrained.pt")
    print(f"  Backbone at {path}")
    import s3prl.hub as hub
    return hub.wav2vec2_local(ckpt=path)


def main():
    args = parse_args()
    print(f"Device: {DEVICE}")
    ssl = build_ssl().to(DEVICE)
    model = NepaliMOSPredictor(ssl).to(DEVICE)

    state = torch.load(str(args.checkpoint), map_location=DEVICE, weights_only=False)
    head_key = "head_state_dict" if "head_state_dict" in state else "head"
    model.head.load_state_dict(state[head_key])
    if state.get("ssl_state_dict"):
        model.ssl_model.load_state_dict(state["ssl_state_dict"], strict=False)
        print(f"Loaded full state (head + backbone) from {CKPT.name}")
    else:
        print(f"Loaded HEAD-ONLY from {CKPT.name} (no backbone state in checkpoint)")
    sp = state.get("spearman") or state.get("sp", "?")
    epoch = state.get("epoch", "?")
    print(f"  epoch={epoch}, val_spearman={sp}")
    model.eval()

    per_file = {}
    per_system = {}

    with torch.no_grad():
        for sys_name in args.systems:
            d = args.audio_root / sys_name
            if not d.exists():
                print(f"  SKIP {sys_name} (no dir)")
                continue
            audio_files = sorted(d.glob("*.mp3")) + sorted(d.glob("*.wav"))
            scores = []
            file_preds = {}
            print(f"  {sys_name}: {len(audio_files)} files...", end=" ", flush=True)
            for audio in audio_files:
                try:
                    wav = load_audio(audio).unsqueeze(0).to(DEVICE)
                    pred = float(model(wav).cpu().item())
                    pred = max(1.0, min(5.0, pred))  # clip to MOS range
                    scores.append(pred)
                    file_preds[audio.stem] = pred
                except Exception as e:
                    print(f"\n    ERROR {audio.name}: {e}")
            scores = np.array(scores)
            per_system[sys_name] = {
                "mean": float(scores.mean()) if len(scores) else None,
                "std": float(scores.std()) if len(scores) > 1 else 0.0,
                "n": len(scores),
            }
            per_file[sys_name] = file_preds
            print(f"mean={per_system[sys_name]['mean']:.3f} n={len(scores)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"per_system": per_system, "per_file": per_file}, indent=2))
    print(f"\nWrote: {args.output}")


if __name__ == "__main__":
    main()
