"""Phonological ABX discrimination test using Whisper encoder embeddings.

For each minimal pair (a, b), extracts Whisper encoder hidden states
and tests whether embeddings of matching items are closer than
embeddings of non-matching items.

Usage:
    python phonological_abx.py --manifest benchmark/results/edge-tts-hemkala/manifest.json
"""

import argparse
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import whisper

from utils import load_json, save_json


# Thresholds from phonological_accuracy_protocol.json
THRESHOLDS = {
    "aspiration_velar": {"pass": 90, "minimum": 80},
    "aspiration_palatal": {"pass": 90, "minimum": 80},
    "aspiration_retroflex": {"pass": 90, "minimum": 80},
    "aspiration_dental": {"pass": 90, "minimum": 80},
    "aspiration_labial": {"pass": 90, "minimum": 80},
    "oral_vs_nasal_vowel": {"pass": 85, "minimum": 75},
    "retroflex_vs_dental": {"pass": 90, "minimum": 80},
    "gemination": {"pass": 85, "minimum": 75},
    "schwa_deletion": {"pass": 80, "minimum": 70},
    "nasal_consonants": {"pass": 85, "minimum": 75},
}

# Map subcategories to threshold groups
THRESHOLD_GROUP = {
    "aspiration_velar": "aspiration_4way",
    "aspiration_palatal": "aspiration_4way",
    "aspiration_retroflex": "aspiration_4way",
    "aspiration_dental": "aspiration_4way",
    "aspiration_labial": "aspiration_4way",
    "oral_vs_nasal_vowel": "nasal_vowels",
    "retroflex_vs_dental": "retroflex_vs_dental",
    "gemination": "gemination",
    "schwa_deletion": "schwa_deletion",
    "nasal_consonants": "nasal_vowels",
}


def extract_whisper_embedding(model, audio_path: str) -> np.ndarray:
    """Extract mean-pooled Whisper encoder hidden states as embedding."""
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    with torch.no_grad():
        encoder_output = model.encoder(mel.unsqueeze(0))
        # Mean-pool over time dimension -> (1, d_model) -> (d_model,)
        embedding = encoder_output.mean(dim=1).squeeze(0).cpu().numpy()

    return embedding


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance (1 - cosine_similarity)."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def run_abx(manifest_path: Path, model_size: str = "medium") -> dict:
    """Run ABX discrimination test on phonological minimal pairs."""
    manifest = load_json(manifest_path)
    output_dir = manifest_path.parent

    # Filter to minimal pair items with audio
    mp_items = [m for m in manifest
                if m["category"] == "phonological_minimal_pairs"
                and m.get("audio_path")
                and m["status"] in ("ok", "cached")]

    # Group into pairs by pair_idx
    pairs_by_idx = defaultdict(dict)
    for item in mp_items:
        pair_idx = item["metadata"]["pair_idx"]
        side = item["metadata"]["side"]
        pairs_by_idx[pair_idx][side] = item

    # Only keep complete pairs
    complete_pairs = {k: v for k, v in pairs_by_idx.items()
                      if "a" in v and "b" in v}
    print(f"Found {len(complete_pairs)} complete minimal pairs (of {len(pairs_by_idx)} total)")

    # Load Whisper model
    print(f"Loading Whisper {model_size} for embedding extraction...")
    t0 = time.time()
    model = whisper.load_model(model_size)
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Extract embeddings for all items
    print("Extracting embeddings...")
    embeddings = {}
    all_items = []
    for pair_idx, sides in complete_pairs.items():
        all_items.extend([sides["a"], sides["b"]])

    for i, item in enumerate(all_items):
        embeddings[item["id"]] = extract_whisper_embedding(model, item["audio_path"])
        if (i + 1) % 20 == 0 or (i + 1) == len(all_items):
            print(f"  [{i+1}/{len(all_items)}] embeddings extracted")

    # Run ABX test
    # For each pair (a_i, b_i), and for each other pair (a_j, b_j) in the same category:
    # Check: is embedding(a_i) closer to embedding(a_j) than to embedding(b_j)?
    # And: is embedding(b_i) closer to embedding(b_j) than to embedding(a_j)?
    print("\nRunning ABX discrimination...")

    # Group pairs by subcategory
    pairs_by_subcat = defaultdict(list)
    for pair_idx, sides in complete_pairs.items():
        subcat = sides["a"]["subcategory"]
        pairs_by_subcat[subcat].append((pair_idx, sides))

    abx_results = {}
    per_pair_results = []

    for subcat, pairs in sorted(pairs_by_subcat.items()):
        correct = 0
        total = 0

        for i, (idx_i, sides_i) in enumerate(pairs):
            emb_a_i = embeddings[sides_i["a"]["id"]]
            emb_b_i = embeddings[sides_i["b"]["id"]]

            # Within-pair discrimination: is a closer to other a's than to b's?
            for j, (idx_j, sides_j) in enumerate(pairs):
                if i == j:
                    continue

                emb_a_j = embeddings[sides_j["a"]["id"]]
                emb_b_j = embeddings[sides_j["b"]["id"]]

                # Test A: is a_i closer to a_j than to b_j?
                dist_aa = cosine_distance(emb_a_i, emb_a_j)
                dist_ab = cosine_distance(emb_a_i, emb_b_j)
                if dist_aa < dist_ab:
                    correct += 1
                total += 1

                # Test B: is b_i closer to b_j than to a_j?
                dist_bb = cosine_distance(emb_b_i, emb_b_j)
                dist_ba = cosine_distance(emb_b_i, emb_a_j)
                if dist_bb < dist_ba:
                    correct += 1
                total += 1

            # Also direct within-pair: a_i should be far from b_i
            direct_dist = cosine_distance(emb_a_i, emb_b_i)
            per_pair_results.append({
                "pair_idx": idx_i,
                "subcategory": subcat,
                "word_a": sides_i["a"]["metadata"]["word"],
                "word_b": sides_i["b"]["metadata"]["word"],
                "cosine_distance": round(float(direct_dist), 4),
            })

        accuracy = (correct / total * 100) if total > 0 else 0.0
        thresholds = THRESHOLDS.get(subcat, {"pass": 85, "minimum": 75})

        if accuracy >= thresholds["pass"]:
            judgment = "pass"
        elif accuracy >= thresholds["minimum"]:
            judgment = "marginal"
        else:
            judgment = "fail"

        abx_results[subcat] = {
            "accuracy": round(accuracy, 1),
            "correct": correct,
            "total": total,
            "n_pairs": len(pairs),
            "threshold_pass": thresholds["pass"],
            "threshold_minimum": thresholds["minimum"],
            "judgment": judgment,
        }

    # Overall
    total_correct = sum(r["correct"] for r in abx_results.values())
    total_tests = sum(r["total"] for r in abx_results.values())
    overall_accuracy = (total_correct / total_tests * 100) if total_tests > 0 else 0.0

    output = {
        "model": model_size,
        "total_pairs": len(complete_pairs),
        "total_abx_tests": total_tests,
        "overall_accuracy": round(overall_accuracy, 1),
        "per_category": abx_results,
        "per_pair_distances": per_pair_results,
    }

    out_path = output_dir / "abx_results.json"
    save_json(out_path, output)
    print(f"\nABX results saved to {out_path}")

    print_summary(abx_results, overall_accuracy)
    return output


def print_summary(abx_results: dict, overall: float):
    """Print human-readable ABX results."""
    print(f"\n=== Phonological ABX Discrimination ===")
    print(f"Overall accuracy: {overall:.1f}%")
    print(f"{'Category':<25} {'Pairs':>5} {'Accuracy':>8} {'Threshold':>9} {'Judge':>8}")
    print("-" * 65)
    for cat in sorted(abx_results.keys()):
        r = abx_results[cat]
        print(f"{cat:<25} {r['n_pairs']:>5} {r['accuracy']:>7.1f}% "
              f"{r['threshold_pass']:>8}% {r['judgment']:>8}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Phonological ABX test")
    parser.add_argument("--manifest", type=Path, required=True,
                        help="Path to manifest.json")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size for embedding extraction")
    args = parser.parse_args()

    run_abx(args.manifest, args.model)


if __name__ == "__main__":
    main()
