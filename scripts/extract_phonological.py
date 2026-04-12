#!/usr/bin/env python3
"""Extract phonologically-relevant utterances from transcribed audio datasets.

Searches chirp2, whisper, and OpenSLR-54 transcripts for utterances containing
words from our phonological minimal pairs. Selects a balanced subset per
contrast category and outputs a manifest for audio extraction.

Usage:
    python scripts/extract_phonological.py

Outputs:
    benchmark/data/phonological_manifest.json  — full manifest with utterance IDs,
        text, matched words, contrast categories, and source dataset info
    benchmark/data/extraction_stats.json — coverage statistics
"""

import json
import os
import re
import random
from collections import defaultdict
from pathlib import Path

random.seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
TRANSCRIPTS = ROOT / "benchmark" / "data" / "transcripts"
OUTPUT = ROOT / "benchmark" / "data"

# ---------- Configuration ----------

# Max utterances per contrast category per dataset
MAX_PER_CATEGORY = 200
# Min utterances per category (warn if below)
MIN_PER_CATEGORY = 50
# Max utterances per speaker within a category (for diversity)
MAX_PER_SPEAKER = 10


def load_minimal_pairs():
    """Load phonological minimal pairs and extract target words per category."""
    mp_path = DATA / "phonological_minimal_pairs.json"
    with open(mp_path) as f:
        mp = json.load(f)

    category_words = {}
    category_pairs = {}
    for key, val in mp.items():
        if not isinstance(val, list):
            continue
        words = set()
        pairs = []
        for pair in val:
            words.add(pair["word_a"])
            words.add(pair["word_b"])
            pairs.append({
                "word_a": pair["word_a"],
                "word_b": pair["word_b"],
                "meaning_a": pair.get("meaning_a", ""),
                "meaning_b": pair.get("meaning_b", ""),
                "contrast_type": pair.get("contrast_type", key),
            })
        category_words[key] = words
        category_pairs[key] = pairs

    return category_words, category_pairs


def load_tsv(path, has_speaker_col=True):
    """Load a TSV file. Returns list of (utt_id, speaker_id, text, audio_path)."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if has_speaker_col and len(parts) >= 3:
                utt_id, spk_id, text = parts[0], parts[1], parts[2]
                audio_path = None  # OpenSLR doesn't have audio path in TSV
            elif not has_speaker_col and len(parts) >= 2:
                utt_id, text = parts[0], parts[1]
                spk_id = "unknown"
                # chirp2/whisper TSVs have audio path in column 2 or derivable from ID
                audio_path = parts[2] if len(parts) > 2 else None
            else:
                continue
            rows.append({
                "utt_id": utt_id,
                "speaker_id": spk_id,
                "text": text,
                "audio_path": audio_path,
            })
    return rows


def search_utterances(rows, category_words):
    """Search utterances for target words. Returns dict[category] -> list of matches."""
    results = defaultdict(list)

    for row in rows:
        text = row["text"]
        for cat, words in category_words.items():
            matched = []
            for word in words:
                # Match word in text - use word boundary-ish matching for Devanagari
                # Devanagari doesn't have \b word boundaries, so check if word appears
                # as a substring (which is how Nepali words typically work in continuous text)
                if word in text:
                    matched.append(word)
            if matched:
                results[cat].append({
                    **row,
                    "matched_words": matched,
                    "category": cat,
                })

    return results


def select_balanced_subset(matches, available_audio=None,
                          max_per_cat=MAX_PER_CATEGORY, max_per_speaker=MAX_PER_SPEAKER):
    """Select a balanced subset ensuring speaker diversity.

    Prefers utterances with locally available audio (available_audio set).
    """
    selected = {}

    for cat, utterances in matches.items():
        # Only keep utterances with locally available audio
        if available_audio:
            utterances = [u for u in utterances if u["utt_id"] in available_audio]

        # Group by speaker
        by_speaker = defaultdict(list)
        for utt in utterances:
            by_speaker[utt["speaker_id"]].append(utt)

        # Round-robin across speakers for diversity
        chosen = []
        speaker_counts = defaultdict(int)
        speakers = list(by_speaker.keys())
        random.shuffle(speakers)

        # Multiple passes to fill up to max
        for _ in range(max_per_speaker):
            for spk in speakers:
                if len(chosen) >= max_per_cat:
                    break
                if speaker_counts[spk] >= max_per_speaker:
                    continue
                remaining = [u for u in by_speaker[spk]
                             if u["utt_id"] not in {c["utt_id"] for c in chosen}]
                if remaining:
                    chosen.append(random.choice(remaining))
                    speaker_counts[spk] += 1
            if len(chosen) >= max_per_cat:
                break

        selected[cat] = chosen

    return selected


def main():
    print("Loading phonological minimal pairs...")
    category_words, category_pairs = load_minimal_pairs()

    print(f"  {len(category_words)} contrast categories")
    for cat, words in category_words.items():
        print(f"    {cat}: {len(words)} target words — {', '.join(sorted(words)[:5])}...")

    # Load all transcript datasets
    datasets = {}

    chirp2_path = TRANSCRIPTS / "chirp2_train.tsv"
    if chirp2_path.exists():
        print(f"\nLoading chirp2 transcripts...")
        datasets["chirp2"] = load_tsv(chirp2_path, has_speaker_col=False)
        print(f"  {len(datasets['chirp2'])} utterances")

    whisper_path = TRANSCRIPTS / "whisper_train.tsv"
    if whisper_path.exists():
        print(f"Loading whisper transcripts...")
        datasets["whisper"] = load_tsv(whisper_path, has_speaker_col=False)
        print(f"  {len(datasets['whisper'])} utterances")

    whisper_next_path = TRANSCRIPTS / "whisper_next_train.tsv"
    if whisper_next_path.exists():
        print(f"Loading whisper-next transcripts...")
        datasets["whisper_next"] = load_tsv(whisper_next_path, has_speaker_col=False)
        print(f"  {len(datasets['whisper_next'])} utterances")

    openslr_path = TRANSCRIPTS / "openslr54_train.tsv"
    if openslr_path.exists():
        print(f"Loading OpenSLR-54 transcripts...")
        datasets["openslr54"] = load_tsv(openslr_path, has_speaker_col=True)
        print(f"  {len(datasets['openslr54'])} utterances")

    if not datasets:
        print("ERROR: No transcript files found!")
        return

    # Search each dataset for phonological matches
    all_matches = defaultdict(list)
    dataset_stats = {}

    for ds_name, rows in datasets.items():
        print(f"\nSearching {ds_name} for phonological target words...")
        matches = search_utterances(rows, category_words)

        ds_stats = {}
        for cat, utts in matches.items():
            # Tag each utterance with its source dataset
            for u in utts:
                u["source_dataset"] = ds_name
            all_matches[cat].extend(utts)
            ds_stats[cat] = len(utts)

        dataset_stats[ds_name] = ds_stats
        total = sum(ds_stats.values())
        print(f"  {total} total matches across {len(ds_stats)} categories")
        for cat, count in sorted(ds_stats.items()):
            print(f"    {cat}: {count}")

    # Load available audio IDs (shard 0 + hetzner recordings)
    available_audio = set()
    openslr_base = Path("/home/cdjk/gt/bolne/crew/iggy/data/openslr54")
    for shard_dir in sorted(openslr_base.glob("shard*")):
        if shard_dir.is_dir():
            for root, dirs, files in os.walk(shard_dir):
                for fname in files:
                    if fname.endswith(".flac"):
                        available_audio.add(fname.replace(".flac", ""))
    print(f"\nLocally available audio: {len(available_audio)} files (OpenSLR shards)")

    # Select balanced subset, preferring utterances with available audio
    print(f"Selecting balanced subset (max {MAX_PER_CATEGORY}/category, "
          f"max {MAX_PER_SPEAKER}/speaker, preferring available audio)...")
    selected = select_balanced_subset(all_matches, available_audio=available_audio)

    # Build manifest
    manifest = {
        "description": "Phonologically-relevant utterances extracted from transcribed Nepali audio",
        "extraction_config": {
            "max_per_category": MAX_PER_CATEGORY,
            "max_per_speaker": MAX_PER_SPEAKER,
            "seed": 42,
        },
        "category_pairs": category_pairs,
        "categories": {},
    }

    stats = {
        "total_utterances_searched": sum(len(rows) for rows in datasets.values()),
        "total_matches_found": sum(len(utts) for utts in all_matches.values()),
        "total_selected": 0,
        "per_category": {},
        "per_dataset": dataset_stats,
        "coverage_warnings": [],
    }

    for cat in sorted(category_words.keys()):
        utts = selected.get(cat, [])
        speakers = set(u["speaker_id"] for u in utts)
        sources = defaultdict(int)
        for u in utts:
            sources[u["source_dataset"]] += 1

        manifest["categories"][cat] = {
            "target_words": sorted(category_words[cat]),
            "num_pairs": len(category_pairs.get(cat, [])),
            "selected_utterances": utts,
        }

        stats["per_category"][cat] = {
            "total_matches": len(all_matches.get(cat, [])),
            "selected": len(utts),
            "unique_speakers": len(speakers),
            "by_source": dict(sources),
        }
        stats["total_selected"] += len(utts)

        status = "OK" if len(utts) >= MIN_PER_CATEGORY else "LOW"
        if status == "LOW":
            stats["coverage_warnings"].append(
                f"{cat}: only {len(utts)} utterances (min: {MIN_PER_CATEGORY})")
        print(f"  {cat}: {len(utts)} selected from {len(all_matches.get(cat, []))} matches "
              f"({len(speakers)} speakers, {dict(sources)}) [{status}]")

    # Write outputs
    manifest_path = OUTPUT / "phonological_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nManifest written to {manifest_path}")

    stats_path = OUTPUT / "extraction_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"Stats written to {stats_path}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Datasets searched: {len(datasets)}")
    print(f"Total utterances searched: {stats['total_utterances_searched']:,}")
    print(f"Total matches found: {stats['total_matches_found']:,}")
    print(f"Selected for extraction: {stats['total_selected']}")
    print(f"Contrast categories: {len(manifest['categories'])}")
    if stats["coverage_warnings"]:
        print(f"\nWARNINGS:")
        for w in stats["coverage_warnings"]:
            print(f"  ⚠ {w}")

    # Generate audio file list for download
    audio_files = []
    for cat, utts in selected.items():
        for u in utts:
            if u.get("audio_path"):
                audio_files.append({
                    "source": u["source_dataset"],
                    "audio_path": u["audio_path"],
                    "utt_id": u["utt_id"],
                })

    audio_list_path = OUTPUT / "audio_download_list.json"
    with open(audio_list_path, "w") as f:
        json.dump(audio_files, f, ensure_ascii=False, indent=2)
    print(f"\nAudio download list ({len(audio_files)} files) written to {audio_list_path}")


if __name__ == "__main__":
    main()
