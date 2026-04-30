"""NepaliMOS auto-MOS evaluation.

NepaliMOS is a Nepali-specific MOS predictor, fine-tuned from IndicWav2Vec base
on 6,962 native-rater MOS scores collected via the NepTTS-Bench rating app.
On the paper's TTS-9 panel it reaches system-level Spearman rho=0.90 against
human MOS, vs SCOREQ's rho=0.40 (Steiger's Z, p=0.028).

Architecture (must match training in model/train_nepali_mos.py):
  IndicWav2Vec base (top 4 transformer layers fine-tuned)
    -> mean-pooled hidden states (768-d)
    -> Linear(768, 256) -> ReLU -> Dropout(0.1) -> Linear(256, 1)

Checkpoints (hosted at huggingface.co/datasets/ampixa/neptts-bench):
  - model/neptts_mos_v9_best.pt  - head only (~800 KB), public default.
    The backbone reverts to vanilla IndicWav2Vec, so predictions are
    on a different scale than the paper's full fine-tuned model.
  - model/neptts_mos_best.pt     - full state (~381 MB), reproduces
    paper baselines exactly. Currently not public; pass via
    --nepalimos-ckpt or NEPALIMOS_CKPT env var.

Optional dep group: pip install neptts-eval[nepalimos]
  Adds: torch, torchaudio, s3prl, huggingface_hub, soundfile.
"""

import os
import sys
from pathlib import Path

HF_REPO = "ampixa/neptts-bench"
HF_REPO_TYPE = "dataset"
CKPT_FILENAME = "model/neptts_mos_v9_best.pt"
SSL_REPO = "SYSPIN/IndicMOS"
SSL_FILENAME = "indicw2v_base_pretrained.pt"

# Cap matches training (15s @ 16kHz). Longer audio is truncated.
MAX_SAMPLES = 16000 * 15


def _build_predictor(ckpt_path: str, device: str):
    """Construct the NepaliMOSPredictor and load weights.

    The checkpoint stores both the fine-tuned backbone (ssl_state_dict) and
    the regression head (head_state_dict). We load both so the artifact
    reproduces the paper's validation rho from the saved file alone.
    """
    import torch
    import torch.nn as nn
    import s3prl.hub as hub
    from huggingface_hub import hf_hub_download

    ssl_local = hf_hub_download(repo_id=SSL_REPO, filename=SSL_FILENAME)
    ssl_model = hub.wav2vec2_local(ckpt=ssl_local)

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

    model = NepaliMOSPredictor(ssl_model).to(device)

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    head_key = "head_state_dict" if "head_state_dict" in state else "head"
    model.head.load_state_dict(state[head_key])
    if state.get("ssl_state_dict"):
        model.ssl_model.load_state_dict(state["ssl_state_dict"], strict=False)
    model.eval()
    return model


def _load_audio_16k(path: Path):
    """Load mono 16kHz float32 tensor, truncated to 15s."""
    import numpy as np
    import torch
    import soundfile as sf

    try:
        data, sr = sf.read(str(path), dtype="float32")
    except Exception:
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        data = wav.mean(0).numpy()
    if data.ndim > 1:
        data = data.mean(axis=1)
    t = torch.from_numpy(data.astype(np.float32))
    if sr != 16000:
        import torchaudio
        t = torchaudio.functional.resample(t, sr, 16000)
    if t.shape[0] > MAX_SAMPLES:
        t = t[:MAX_SAMPLES]
    return t


def evaluate_nepalimos(
    audio_files: dict[str, Path],
    verbose: bool = False,
    ckpt_path: str | None = None,
    device: str | None = None,
) -> dict:
    """Run NepaliMOS on audio files. Returns per-file and aggregate scores.

    Args:
        audio_files: {sent_id: Path} mapping.
        verbose: print progress every 50 files.
        ckpt_path: local checkpoint path. Default: download from HF
            (env override: NEPALIMOS_CKPT).
        device: 'cpu' or 'cuda'. Default: cuda if available, else cpu.

    Returns:
        {avg_mos, n_scored, n_errors, per_file: {sent_id: score}}.
        Scores are clipped to the MOS range [1.0, 5.0].
    """
    import torch
    from huggingface_hub import hf_hub_download

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if ckpt_path is None:
        ckpt_path = os.environ.get("NEPALIMOS_CKPT")
    if ckpt_path is None:
        if verbose:
            print(f"  NepaliMOS: downloading checkpoint from {HF_REPO}...", file=sys.stderr)
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO, filename=CKPT_FILENAME, repo_type=HF_REPO_TYPE,
        )

    if verbose:
        print(f"  NepaliMOS: loading model on {device}...", file=sys.stderr)
    model = _build_predictor(ckpt_path, device)

    scores: dict[str, float] = {}
    errors = 0

    with torch.no_grad():
        for i, (sent_id, path) in enumerate(sorted(audio_files.items())):
            try:
                wav = _load_audio_16k(Path(path)).unsqueeze(0).to(device)
                pred = float(model(wav).cpu().item())
                pred = max(1.0, min(5.0, pred))
                scores[sent_id] = round(pred, 3)
            except Exception as e:
                errors += 1
                if verbose and errors <= 5:
                    print(f"  NepaliMOS error {sent_id}: {e}", file=sys.stderr)

            if verbose and (i + 1) % 50 == 0:
                avg = sum(scores.values()) / len(scores) if scores else 0
                print(f"  NepaliMOS: {i+1}/{len(audio_files)} - avg: {avg:.3f}", file=sys.stderr)

    avg_mos = sum(scores.values()) / len(scores) if scores else 0

    return {
        "avg_mos": round(avg_mos, 3),
        "n_scored": len(scores),
        "n_errors": errors,
        "per_file": scores,
    }
