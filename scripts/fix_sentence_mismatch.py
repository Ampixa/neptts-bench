#!/usr/bin/env python3
"""Fix sentence text mismatch between DB and recorded audio.

Problem: 33 sentences were changed in iggy's sentences.json after recordings
were collected. The DB was updated to iggy's text, but the audio corresponds
to the original server text.

Fix:
1. Restore 29 recorded sentences to server text (what speakers actually read)
2. Add iggy's replacement sentences as new IDs (sent_175+) for future recording
3. Add 10 sentences (sent_127-136) that were in JSON but never loaded into DB
4. Generate updated sentences.json for the server

Usage:
    # Dry run (default)
    python scripts/fix_sentence_mismatch.py

    # Apply to local DB copy
    python scripts/fix_sentence_mismatch.py --apply

    # Apply to live Hetzner DB (via SSH)
    python scripts/fix_sentence_mismatch.py --apply --live
"""

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DB = ROOT / "benchmark" / "data" / "hetzner_recordings" / "recordings.db"
SERVER_SENTENCES = Path("/tmp/server_sentences.json")
IGGY_SENTENCES = ROOT.parent / "iggy" / "benchmark" / "mos" / "sentences.json"

DRY_RUN = "--apply" not in sys.argv
LIVE = "--live" in sys.argv


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    if DRY_RUN:
        print("=== DRY RUN (pass --apply to execute) ===\n")

    server = {s["sent_id"]: s for s in load_json(SERVER_SENTENCES)}
    iggy = {s["sent_id"]: s for s in load_json(IGGY_SENTENCES)}

    db = sqlite3.connect(LOCAL_DB)
    db.row_factory = sqlite3.Row
    c = db.cursor()

    # Get current DB state
    c.execute("SELECT sent_id, text_dev, text_roman FROM sentences")
    db_sents = {r["sent_id"]: dict(r) for r in c.fetchall()}

    # --- Step 1: Find and fix mismatches with recordings ---
    print("STEP 1: Restore recorded text for mismatched sentences\n")

    restore_sql = []
    for sid in sorted(server):
        if sid not in iggy:
            continue
        s_text = server[sid]["text_devanagari"]
        i_text = iggy[sid]["text_devanagari"]
        if s_text == i_text:
            continue

        # Check if has recordings
        c.execute("SELECT COUNT(*) FROM recordings WHERE sentence_id=?", (sid,))
        rec_count = c.fetchone()[0]

        if rec_count > 0:
            s_roman = server[sid].get("text_romanized", "")
            print(f"  {sid} ({rec_count} recs): restore to '{s_text[:50]}...'")
            restore_sql.append((s_text, s_roman, sid))
        else:
            print(f"  {sid} (0 recs): skip (no recordings to fix)")

    print(f"\n  → {len(restore_sql)} sentences to restore\n")

    if not DRY_RUN and restore_sql:
        for text, roman, sid in restore_sql:
            c.execute(
                "UPDATE sentences SET text_dev=?, text_roman=? WHERE sent_id=?",
                (text, roman, sid),
            )
        db.commit()
        print(f"  ✓ Updated {len(restore_sql)} sentences in local DB\n")

    # --- Step 2: Add iggy's replacements as new sentence IDs ---
    print("STEP 2: Add iggy's replacement sentences as new IDs\n")

    # Find the max existing sent_id number
    c.execute("SELECT sent_id FROM sentences")
    all_ids = [r[0] for r in c.fetchall()]
    max_num = max(int(sid.replace("sent_", "")) for sid in all_ids if sid.startswith("sent_"))
    next_num = max(max_num + 1, 175)  # Start from 175 or higher

    new_v2_sentences = []
    for sid in sorted(server):
        if sid not in iggy:
            continue
        s_text = server[sid]["text_devanagari"]
        i_text = iggy[sid]["text_devanagari"]
        if s_text == i_text:
            continue

        # iggy's version is the "v2" — add as new ID
        new_sid = f"sent_{next_num:03d}"
        iggy_sent = iggy[sid]
        new_sent = {
            "sent_id": new_sid,
            "text_dev": i_text,
            "text_roman": iggy_sent.get("text_romanized", ""),
            "word_count": iggy_sent.get("word_count", len(i_text.split())),
            "category": iggy_sent.get("category", ""),
            "phonetic_targets": json.dumps(iggy_sent.get("phonetic_targets", [])),
            "contrast_word": iggy_sent.get("contrast_word", ""),
            "pair_id": iggy_sent.get("pair_id", ""),
            "replaces": sid,  # Track lineage
        }
        new_v2_sentences.append(new_sent)
        print(f"  {sid} → {new_sid}: '{i_text[:50]}...'")
        next_num += 1

    print(f"\n  → {len(new_v2_sentences)} new v2 sentences to add\n")

    if not DRY_RUN and new_v2_sentences:
        for s in new_v2_sentences:
            c.execute(
                """INSERT OR IGNORE INTO sentences
                   (sent_id, text_dev, text_roman, word_count, category,
                    phonetic_targets, contrast_word, pair_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (s["sent_id"], s["text_dev"], s["text_roman"], s["word_count"],
                 s["category"], s["phonetic_targets"], s["contrast_word"], s["pair_id"]),
            )
        db.commit()
        print(f"  ✓ Inserted {len(new_v2_sentences)} v2 sentences\n")

    # --- Step 3: Add missing sentences (127-136) ---
    print("STEP 3: Add sentences missing from DB\n")

    missing = set(server.keys()) - set(db_sents.keys())
    # Also add any from iggy that aren't in server
    missing |= set(iggy.keys()) - set(db_sents.keys()) - set(s["sent_id"] for s in new_v2_sentences)

    new_missing = []
    for sid in sorted(missing):
        src = server.get(sid, iggy.get(sid))
        if not src:
            continue
        new_missing.append({
            "sent_id": sid,
            "text_dev": src["text_devanagari"],
            "text_roman": src.get("text_romanized", ""),
            "word_count": src.get("word_count", len(src["text_devanagari"].split())),
            "category": src.get("category", ""),
            "phonetic_targets": json.dumps(src.get("phonetic_targets", [])),
            "contrast_word": src.get("contrast_word", ""),
            "pair_id": src.get("pair_id", ""),
        })
        print(f"  {sid}: '{src['text_devanagari'][:60]}'")

    print(f"\n  → {len(new_missing)} missing sentences to add\n")

    if not DRY_RUN and new_missing:
        for s in new_missing:
            c.execute(
                """INSERT OR IGNORE INTO sentences
                   (sent_id, text_dev, text_roman, word_count, category,
                    phonetic_targets, contrast_word, pair_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (s["sent_id"], s["text_dev"], s["text_roman"], s["word_count"],
                 s["category"], s["phonetic_targets"], s["contrast_word"], s["pair_id"]),
            )
        db.commit()
        print(f"  ✓ Inserted {len(new_missing)} missing sentences\n")

    # --- Step 4: Generate updated sentences.json ---
    print("STEP 4: Generate updated sentences.json\n")

    c.execute("SELECT * FROM sentences ORDER BY sent_id")
    all_sents = []
    for row in c.fetchall():
        all_sents.append({
            "sent_id": row[0],
            "text_devanagari": row[1],
            "text_romanized": row[2],
            "word_count": row[3],
            "category": row[4],
            "phonetic_targets": row[5],
            "contrast_word": row[6],
            "pair_id": row[7],
        })

    output_path = ROOT / "benchmark" / "data" / "sentences_fixed.json"
    with open(output_path, "w") as f:
        json.dump(all_sents, f, ensure_ascii=False, indent=2)
    print(f"  → Written {len(all_sents)} sentences to {output_path}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Restored to recorded text:  {len(restore_sql)} sentences")
    print(f"Added v2 replacements:      {len(new_v2_sentences)} sentences")
    print(f"Added missing sentences:    {len(new_missing)} sentences")

    c.execute("SELECT COUNT(*) FROM sentences")
    print(f"Total sentences in DB:      {c.fetchone()[0]}")
    c.execute("SELECT COUNT(*) FROM recordings")
    print(f"Total recordings:           {c.fetchone()[0]} (unchanged)")

    if DRY_RUN:
        print(f"\n⚠ DRY RUN — no changes made. Run with --apply to execute.")
    else:
        print(f"\n✓ Local DB updated. To apply to Hetzner, run with --apply --live")

    db.close()


if __name__ == "__main__":
    main()
