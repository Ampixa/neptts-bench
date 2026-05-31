#!/usr/bin/env python3
"""Leave-one-system-out validation for NepaliMOS.

Each fold holds out one TTS system completely, trains NepaliMOS on the
remaining systems, selects the checkpoint by an internal validation split, and
evaluates only on the held-out system.
"""

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from model.train_nepali_mos import (  # noqa: E402
    BASE_PREDICTOR,
    REPO_ID,
    SSL_NAME,
    NepaliMOSDataset,
    NepaliMOSPredictor,
    collate_fn,
    load_ratings,
)

DEFAULT_TTS_DIR = Path("/home/cdjk/gt/bolne/crew/bolne/benchmark/data/tts_outputs")
DEFAULT_OUTPUT = ROOT / "benchmark" / "results" / "nepalimos_loso.json"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ratings-db",
        type=Path,
        default=ROOT / "benchmark" / "data" / "ratings_prod.db",
    )
    parser.add_argument("--tts-dir", type=Path, default=DEFAULT_TTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--fold-dir", type=Path, default=None)
    parser.add_argument("--systems", nargs="+", default=None)
    parser.add_argument("--exclude-systems", nargs="+", default=[])
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--unfreeze-layers", type=int, default=4)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit-train-samples", type=int, default=None)
    parser.add_argument("--limit-val-samples", type=int, default=None)
    parser.add_argument("--limit-test-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop a fold after this many non-improving epochs; 0 disables.",
    )
    parser.add_argument("--init-from-indicmos", action="store_true")
    parser.add_argument("--save-fold-checkpoints", action="store_true")
    parser.add_argument(
        "--no-clip-predictions",
        action="store_true",
        help="Do not clip held-out predictions to the 1-5 MOS range.",
    )
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_internal_split(samples: list[dict], val_frac: float, seed: int):
    buckets = {1: [], 2: [], 3: [], 4: [], 5: []}
    for i, sample in enumerate(samples):
        bucket = max(1, min(5, round(float(sample["mos"]))))
        buckets[bucket].append(i)

    rng = random.Random(seed)
    val_idx = set()
    for indices in buckets.values():
        if not indices:
            continue
        rng.shuffle(indices)
        n_val = max(1, int(round(len(indices) * val_frac)))
        if len(indices) > 1:
            n_val = min(n_val, len(indices) - 1)
        val_idx.update(indices[:n_val])

    train = [sample for i, sample in enumerate(samples) if i not in val_idx]
    val = [sample for i, sample in enumerate(samples) if i in val_idx]
    return train, val


def safe_corr(fn, targets, preds):
    if len(targets) < 2:
        return None, None
    if len(set(np.round(targets, 8))) < 2 or len(set(np.round(preds, 8))) < 2:
        return None, None
    rho, p = fn(targets, preds)
    if math.isnan(rho):
        return None, None
    return float(rho), float(p) if not math.isnan(p) else None


def regression_metrics(targets, preds):
    from scipy.stats import pearsonr, spearmanr

    targets = np.asarray(targets, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    err = preds - targets
    spearman, spearman_p = safe_corr(spearmanr, targets, preds)
    pearson, pearson_p = safe_corr(pearsonr, targets, preds)
    return {
        "n": int(len(targets)),
        "spearman": spearman,
        "spearman_p": spearman_p,
        "pearson": pearson,
        "pearson_p": pearson_p,
        "mae": float(np.mean(np.abs(err))) if len(err) else None,
        "rmse": float(np.sqrt(np.mean(err**2))) if len(err) else None,
        "target_mean": float(np.mean(targets)) if len(targets) else None,
        "pred_mean": float(np.mean(preds)) if len(preds) else None,
        "target_std": float(np.std(targets)) if len(targets) > 1 else 0.0,
        "pred_std": float(np.std(preds)) if len(preds) > 1 else 0.0,
    }


def system_summary(samples: list[dict]):
    by_system = defaultdict(list)
    ratings = defaultdict(int)
    for sample in samples:
        by_system[sample["system"]].append(float(sample["mos"]))
        ratings[sample["system"]] += int(sample["n_ratings"])
    return {
        system: {
            "clips": len(vals),
            "ratings": ratings[system],
            "mean_mos_label": float(np.mean(vals)),
            "std_mos_label": float(np.std(vals)) if len(vals) > 1 else 0.0,
        }
        for system, vals in sorted(by_system.items())
    }


def load_native_human_mos(db_path: Path):
    import sqlite3

    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        """
        SELECT r.system_name, r.score
        FROM ratings r
        JOIN raters rr ON r.rater_id = rr.id
        WHERE rr.native_speaker = 1
        """
    ).fetchall()
    conn.close()
    by_system = defaultdict(list)
    for system, score in rows:
        by_system[system].append(float(score))
    return {
        system: {"mean": float(np.mean(scores)), "n": len(scores)}
        for system, scores in by_system.items()
    }


def build_ssl_model():
    from huggingface_hub import hf_hub_download
    import s3prl.hub as hub

    ssl_path = hf_hub_download(repo_id=REPO_ID, filename=SSL_NAME)
    return getattr(hub, "wav2vec2_custom")(ckpt=ssl_path)


def maybe_init_from_indicmos(model, hidden_dim: int):
    from huggingface_hub import hf_hub_download

    predictor_path = hf_hub_download(repo_id=REPO_ID, filename=BASE_PREDICTOR)
    pretrained = torch.load(predictor_path, map_location="cpu", weights_only=False)
    model.head[0].weight.data[:768] = pretrained["linear.weight"].repeat(
        hidden_dim // 1 + 1, 1
    )[:hidden_dim]


def make_model(args):
    ssl_model = build_ssl_model()
    model = NepaliMOSPredictor(
        ssl_model,
        hidden_dim=args.hidden_dim,
        unfreeze_layers=args.unfreeze_layers,
    )
    if args.init_from_indicmos:
        maybe_init_from_indicmos(model, args.hidden_dim)
    return model.to(args.device)


def capture_trainable_state(model):
    return {
        "head": {
            key: value.detach().cpu().clone()
            for key, value in model.head.state_dict().items()
        },
        "ssl_params": {
            key: value.detach().cpu().clone()
            for key, value in model.ssl_model.named_parameters()
            if value.requires_grad
        },
    }


def restore_trainable_state(model, state):
    model.head.load_state_dict(state["head"])
    ssl_params = dict(model.ssl_model.named_parameters())
    for key, value in state["ssl_params"].items():
        ssl_params[key].data.copy_(value.to(ssl_params[key].device))


def save_fold_checkpoint(path: Path, model, args, fold_result: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "head_state_dict": model.head.state_dict(),
            "ssl_state_dict": model.ssl_model.state_dict()
            if args.unfreeze_layers > 0
            else None,
            "unfreeze_layers": args.unfreeze_layers,
            "hidden_dim": args.hidden_dim,
            "fold": fold_result,
        },
        path,
    )


def train_one_fold(holdout: str, samples: list[dict], args, fold_dir: Path):
    from scipy.stats import pearsonr, spearmanr

    fold_seed = args.seed + sum(ord(c) for c in holdout)
    set_seed(fold_seed)

    train_pool = [s for s in samples if s["system"] != holdout]
    test_samples = [s for s in samples if s["system"] == holdout]
    train_samples, val_samples = stratified_internal_split(
        train_pool, args.val_frac, fold_seed
    )
    if args.limit_train_samples is not None:
        train_samples = train_samples[: args.limit_train_samples]
    if args.limit_val_samples is not None:
        val_samples = val_samples[: args.limit_val_samples]
    if args.limit_test_samples is not None:
        test_samples = test_samples[: args.limit_test_samples]

    train_loader = DataLoader(
        NepaliMOSDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        NepaliMOSDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        NepaliMOSDataset(test_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )

    print(
        f"\n=== Hold out {holdout}: train={len(train_samples)} "
        f"val={len(val_samples)} test={len(test_samples)} ===",
        flush=True,
    )
    model = make_model(args)

    backbone_params = [p for p in model.ssl_model.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    if backbone_params:
        optimizer = torch.optim.Adam(
            [
                {"params": backbone_params, "lr": args.lr / 10},
                {"params": head_params, "lr": args.lr},
            ]
        )
    else:
        optimizer = torch.optim.Adam(head_params, lr=args.lr)
    criterion = nn.MSELoss()

    best_state = None
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_metrics = {}
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (audio, scores, lengths) in enumerate(train_loader, start=1):
            audio = audio.to(args.device)
            scores = scores.to(args.device)
            lengths = lengths.to(args.device)
            pred = model(audio, lengths)
            loss = criterion(pred, scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if args.log_every and batch_idx % args.log_every == 0:
                print(
                    f"    batch {batch_idx}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        val_targets = []
        val_preds = []
        with torch.no_grad():
            for audio, scores, lengths in val_loader:
                audio = audio.to(args.device)
                scores = scores.to(args.device)
                lengths = lengths.to(args.device)
                pred = model(audio, lengths)
                val_loss += criterion(pred, scores).item()
                val_targets.extend(scores.cpu().numpy().tolist())
                val_preds.extend(pred.cpu().numpy().tolist())

        val_loss /= max(1, len(val_loader))
        val_spearman, val_spearman_p = safe_corr(spearmanr, val_targets, val_preds)
        val_pearson, val_pearson_p = safe_corr(pearsonr, val_targets, val_preds)
        epoch_record = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_spearman": val_spearman,
            "val_spearman_p": val_spearman_p,
            "val_pearson": val_pearson,
            "val_pearson_p": val_pearson_p,
        }
        history.append(epoch_record)
        print(
            f"  epoch {epoch:02d}/{args.epochs}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"rho={val_spearman if val_spearman is not None else float('nan'):.3f}",
            flush=True,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_metrics = epoch_record
            best_state = capture_trainable_state(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if args.patience and stale_epochs >= args.patience:
                print(
                    f"  stopping after {stale_epochs} non-improving epochs",
                    flush=True,
                )
                break

    if best_state is not None:
        restore_trainable_state(model, best_state)

    model.eval()
    test_targets = []
    test_preds = []
    test_records = []
    cursor = 0
    with torch.no_grad():
        for audio, scores, lengths in test_loader:
            audio = audio.to(args.device)
            lengths = lengths.to(args.device)
            pred = model(audio, lengths).cpu().numpy().tolist()
            if not args.no_clip_predictions:
                pred = [max(1.0, min(5.0, float(p))) for p in pred]
            targets = scores.numpy().tolist()
            for target, prediction in zip(targets, pred):
                sample = test_samples[cursor]
                test_records.append(
                    {
                        "system": sample["system"],
                        "sent_id": sample["sent_id"],
                        "target_mos": float(target),
                        "predicted_mos": float(prediction),
                        "n_ratings": int(sample["n_ratings"]),
                    }
                )
                cursor += 1
            test_targets.extend(float(t) for t in targets)
            test_preds.extend(float(p) for p in pred)

    fold_result = {
        "heldout_system": holdout,
        "seed": fold_seed,
        "n_train": len(train_samples),
        "n_internal_val": len(val_samples),
        "n_test": len(test_samples),
        "best_epoch": best_epoch,
        "best_val_loss": float(best_val_loss),
        "best_val_metrics": best_val_metrics,
        "test_metrics": regression_metrics(test_targets, test_preds),
        "test_predictions": test_records,
        "history": history,
    }
    print(
        f"  held-out {holdout}: pred_mean={fold_result['test_metrics']['pred_mean']:.3f} "
        f"target_mean={fold_result['test_metrics']['target_mean']:.3f} "
        f"test_rho={fold_result['test_metrics']['spearman']}",
        flush=True,
    )

    if args.save_fold_checkpoints:
        save_fold_checkpoint(
            fold_dir / f"{holdout.replace('/', '__')}_best.pt",
            model,
            args,
            fold_result,
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return fold_result


def aggregate_results(folds: list[dict], native_human_mos: dict):
    all_targets = []
    all_preds = []
    systems = []
    label_means = []
    pred_means = []
    native_means = []

    for fold in folds:
        metrics = fold["test_metrics"]
        if metrics["n"] == 0:
            continue
        systems.append(fold["heldout_system"])
        label_means.append(metrics["target_mean"])
        pred_means.append(metrics["pred_mean"])
        native = native_human_mos.get(fold["heldout_system"])
        if native is not None:
            native_means.append(native["mean"])
        for record in fold["test_predictions"]:
            all_targets.append(record["target_mos"])
            all_preds.append(record["predicted_mos"])

    aggregate = {
        "n_folds": len(folds),
        "systems": systems,
        "utterance_level": regression_metrics(all_targets, all_preds),
        "system_level_label_means": regression_metrics(label_means, pred_means),
    }
    if len(native_means) == len(pred_means):
        aggregate["system_level_native_human_means"] = regression_metrics(
            native_means, pred_means
        )
    return aggregate


def write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def main():
    args = parse_args()
    args.device = str(torch.device(args.device))
    if args.fold_dir is None:
        args.fold_dir = args.output.parent / f"{args.output.stem}_folds"

    samples = load_ratings(str(args.ratings_db), str(args.tts_dir))
    samples = sorted(samples, key=lambda s: (s["system"], str(s["sent_id"])))
    excluded = set(args.exclude_systems)
    available_systems = sorted({sample["system"] for sample in samples} - excluded)
    systems = args.systems or available_systems
    missing = sorted(set(systems) - set(available_systems))
    if missing:
        raise SystemExit(f"Unknown or excluded system(s): {', '.join(missing)}")
    if args.max_folds is not None:
        systems = systems[: args.max_folds]

    native_human_mos = load_native_human_mos(args.ratings_db)
    config = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "ratings_db": str(args.ratings_db),
        "tts_dir": str(args.tts_dir),
        "systems": systems,
        "excluded_systems": sorted(excluded),
        "device": args.device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "unfreeze_layers": args.unfreeze_layers,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "patience": args.patience,
        "clip_predictions": not args.no_clip_predictions,
        "init_from_indicmos": args.init_from_indicmos,
        "limit_train_samples": args.limit_train_samples,
        "limit_val_samples": args.limit_val_samples,
        "limit_test_samples": args.limit_test_samples,
        "log_every": args.log_every,
    }

    result = {
        "config": config,
        "data_summary": system_summary(samples),
        "folds": [],
        "aggregate": {},
    }
    if args.resume and args.output.exists():
        result = json.loads(args.output.read_text())
        result["config"] = config
        result["data_summary"] = system_summary(samples)

    print(json.dumps({"config": config, "data_summary": result["data_summary"]}, indent=2))

    if args.dry_run:
        write_json(args.output, result)
        print(f"\nDry run wrote fold plan to {args.output}")
        return

    completed = {fold["heldout_system"] for fold in result.get("folds", [])}
    for holdout in systems:
        if holdout in completed:
            print(f"\n=== Hold out {holdout}: already complete, skipping ===", flush=True)
            continue
        fold_result = train_one_fold(holdout, samples, args, args.fold_dir)
        result.setdefault("folds", []).append(fold_result)
        result["aggregate"] = aggregate_results(result["folds"], native_human_mos)
        write_json(args.output, result)
        print(f"  wrote partial results to {args.output}", flush=True)

    result["aggregate"] = aggregate_results(result["folds"], native_human_mos)
    write_json(args.output, result)
    print("\nAggregate:")
    print(json.dumps(result["aggregate"], indent=2))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
