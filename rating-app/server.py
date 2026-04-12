"""NepTTS-Bench MOS Rating App — collect human quality ratings for speech.

Raters listen to audio (TTS + human recordings, blinded, randomized) and
rate on a 1-5 MOS scale. Each rater picks how much time they can afford.
"""

import json
import os
import random
import sqlite3
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import secrets

from fastapi import FastAPI, Request, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("DATA_DIR", str(APP_DIR / "data")))
TTS_DIR = Path(os.environ.get("TTS_DIR", str(APP_DIR.parent.parent / "data" / "tts_outputs")))
SENTENCES_PATH = Path(os.environ.get("SENTENCES_PATH", str(APP_DIR.parent.parent / "data" / "sentences_fixed.json")))
DB_PATH = DATA_DIR / "ratings.db"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "changeme")

app = FastAPI(title="NepTTS-Bench MOS Rating")
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))

# Cache sentence texts on startup
_sentences: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db():
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()


def init_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.executescript("""
        CREATE TABLE IF NOT EXISTS raters (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            native_speaker INTEGER DEFAULT 1,
            device_info TEXT,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rater_id TEXT REFERENCES raters(id),
            system_name TEXT NOT NULL,
            sent_id TEXT NOT NULL,
            score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rater_assignments (
            rater_id TEXT,
            system_name TEXT,
            sent_id TEXT,
            sort_order INTEGER,
            PRIMARY KEY (rater_id, system_name, sent_id)
        );

        CREATE TABLE IF NOT EXISTS pair_ratings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rater_id TEXT,
            system_name TEXT NOT NULL,
            pair_id TEXT NOT NULL,
            sent_id_a TEXT NOT NULL,
            sent_id_b TEXT NOT NULL,
            can_distinguish INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
    """)
    db.commit()
    db.close()


def load_sentences():
    global _sentences
    if SENTENCES_PATH.exists():
        with open(SENTENCES_PATH, encoding="utf-8") as f:
            for s in json.load(f):
                _sentences[s["sent_id"]] = {
                    "text_dev": s["text_devanagari"],
                    "text_roman": s.get("text_romanized", ""),
                    "category": s.get("category", ""),
                    "pair_id": s.get("pair_id", ""),
                    "contrast_word": s.get("contrast_word", ""),
                    "word_count": s.get("word_count", 0),
                }


# ---------------------------------------------------------------------------
# Audio discovery
# ---------------------------------------------------------------------------

def discover_systems():
    """Find all audio sources: TTS systems + human speakers."""
    systems = {}

    # TTS systems
    if TTS_DIR.exists():
        for system_dir in sorted(TTS_DIR.iterdir()):
            if not system_dir.is_dir():
                continue
            name = system_dir.name
            if name == "edge_tts":
                for voice_dir in sorted(system_dir.iterdir()):
                    if voice_dir.is_dir():
                        key = f"edge_tts/{voice_dir.name}"
                        files = {f.stem: str(f) for f in sorted(voice_dir.glob("*.mp3"))}
                        if files:
                            systems[key] = files
            else:
                files = {}
                for ext in ("*.mp3", "*.wav", "*.webm", "*.flac"):
                    for f in sorted(system_dir.glob(ext)):
                        files[f.stem] = str(f)
                if files:
                    systems[name] = files

    return systems


# Seconds per stimulus (listen + think + rate)
SECS_PER_STIMULUS = 12


def assign_stimuli(db, rater_id, minutes: int):
    """Assign stimuli based on available time.

    Each system gets an equal number of stimuli. Systems that share sentences
    (TTS + human) use common sentence pools; systems with their own sentences
    (natural_speech/chirp2) draw from their own pool.
    """
    systems = discover_systems()
    n_systems = len(systems)
    if n_systems == 0:
        return []

    total_budget = (minutes * 60) // SECS_PER_STIMULUS
    n_per_system = max(1, total_budget // n_systems)

    # Separate systems into shared-sentence pool vs own-sentence pool.
    # "natural_speech" has chirp_XXX IDs; all others share sent_XXX IDs.
    shared_systems = {}
    own_systems = {}
    for name, files in systems.items():
        sample_id = next(iter(files), "")
        if sample_id.startswith("chirp_"):
            own_systems[name] = files
        else:
            shared_systems[name] = files

    assignments = []

    # Shared pool: pick common sentences, assign across all shared systems
    if shared_systems:
        all_sets = [set(f.keys()) for f in shared_systems.values()]
        common = sorted(set.intersection(*all_sets)) if all_sets else []
        selected = random.sample(common, min(n_per_system, len(common)))
        for sent_id in selected:
            for sys_name in shared_systems:
                assignments.append((sys_name, sent_id))

    # Own pool: each system picks from its own sentences independently
    for sys_name, files in own_systems.items():
        available = sorted(files.keys())
        selected = random.sample(available, min(n_per_system, len(available)))
        for sent_id in selected:
            assignments.append((sys_name, sent_id))

    # Shuffle ensuring same sentence doesn't appear back-to-back
    random.shuffle(assignments)
    for _ in range(200):
        changed = False
        for i in range(1, len(assignments)):
            if assignments[i][1] == assignments[i - 1][1]:
                swap_start = min(i + 3, len(assignments) - 1)
                if swap_start < len(assignments):
                    swap = random.randint(swap_start, len(assignments) - 1)
                    assignments[i], assignments[swap] = assignments[swap], assignments[i]
                    changed = True
        if not changed:
            break

    for i, (sys_name, sent_id) in enumerate(assignments):
        db.execute(
            "INSERT OR IGNORE INTO rater_assignments (rater_id, system_name, sent_id, sort_order) VALUES (?,?,?,?)",
            (rater_id, sys_name, sent_id, i),
        )
    db.commit()
    return assignments


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

@app.on_event("startup")
def startup():
    init_db()
    load_sentences()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("rate.html", {"request": request})


@app.get("/long-rating", response_class=HTMLResponse)
async def long_rating(request: Request):
    return templates.TemplateResponse("rate_long.html", {"request": request})


@app.get("/voices", response_class=HTMLResponse)
async def voices_page(request: Request):
    return templates.TemplateResponse("voices.html", {"request": request})


@app.get("/pairs", response_class=HTMLResponse)
async def pair_test_page(request: Request):
    return templates.TemplateResponse("pair_test.html", {"request": request})


@app.get("/pairs/all", response_class=HTMLResponse)
async def pairs_browse_page(request: Request):
    return templates.TemplateResponse("pairs.html", {"request": request})


@app.get("/api/pairs")
def get_pairs():
    """Get all valid minimal pairs with their sentences and available systems."""
    systems = discover_systems()
    sys_names = sorted(s for s in systems if s not in ("mms_tts",))

    # Group sentences by pair_id
    pair_groups = {}
    for sid, info in _sentences.items():
        pid = info.get("pair_id", "")
        if not pid:
            continue
        if pid not in pair_groups:
            pair_groups[pid] = []
        pair_groups[pid].append({"sent_id": sid, **info})

    # Only valid pairs (exactly 2 sentences)
    pairs = []
    categories = set()
    for pid in sorted(pair_groups):
        sents = pair_groups[pid]
        if len(sents) != 2:
            continue

        a, b = sents[0], sents[1]
        # Find systems that have BOTH sentences
        available = [s for s in sys_names
                     if a["sent_id"] in systems.get(s, {}) and b["sent_id"] in systems.get(s, {})]
        if not available:
            continue

        # Extract category from pair_id
        parts = pid.split("/")
        cat = "/".join(parts[:2]) if len(parts) >= 2 else pid
        categories.add(cat)

        # Contrast label
        contrast = parts[1] if len(parts) >= 2 else "unknown"
        contrast_label = contrast.replace("_", " ").title()

        pairs.append({
            "pair_id": pid,
            "category": cat,
            "contrast_label": contrast_label,
            "sent_a": {
                "sent_id": a["sent_id"],
                "text_dev": a.get("text_dev", ""),
                "text_roman": a.get("text_roman", ""),
                "contrast_word": a.get("contrast_word", ""),
            },
            "sent_b": {
                "sent_id": b["sent_id"],
                "text_dev": b.get("text_dev", ""),
                "text_roman": b.get("text_roman", ""),
                "contrast_word": b.get("contrast_word", ""),
            },
            "systems": available,
        })

    return {"pairs": pairs, "categories": sorted(categories)}


@app.get("/api/voices/ratings")
def voices_ratings(db=Depends(get_db)):
    """Get average ratings per (system, sent_id)."""
    rows = db.execute("""
        SELECT system_name, sent_id, ROUND(AVG(score), 2) as avg_score, COUNT(*) as n
        FROM ratings
        GROUP BY system_name, sent_id
    """).fetchall()
    ratings = {}
    for r in rows:
        key = f"{r['system_name']}/{r['sent_id']}"
        ratings[key] = {"avg": r["avg_score"], "n": r["n"]}
    return ratings


@app.get("/api/voices/sentences")
def voices_sentences():
    systems = discover_systems()
    sys_names = sorted(s for s in systems if s != "mms_tts")

    # Build per-system sentence availability
    sys_sents = {s: set(files.keys()) for s, files in systems.items() if s != "mms_tts"}

    # Only include sentences that exist in at least one system
    all_sids = sorted(set().union(*sys_sents.values()))

    sents = []
    for sid in all_sids:
        info = _sentences.get(sid, {})
        available_in = [s for s in sys_names if sid in sys_sents.get(s, set())]
        sents.append({
            "sent_id": sid,
            "text_dev": info.get("text_dev", sid),
            "text_roman": info.get("text_roman", ""),
            "word_count": 0,
            "category": "",
            "systems": available_in,
        })

    return {"sentences": sents, "systems": sys_names}


class RaterCreate(BaseModel):
    name: str
    native_speaker: bool = True
    minutes: int = 10
    device_info: str | None = None


@app.post("/api/raters")
def create_rater(body: RaterCreate, db=Depends(get_db)):
    rater_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()

    db.execute(
        "INSERT INTO raters (id, name, native_speaker, device_info, created_at) VALUES (?,?,?,?,?)",
        (rater_id, body.name, 1 if body.native_speaker else 0, body.device_info, now),
    )

    minutes = max(3, min(body.minutes, 120))
    assignments = assign_stimuli(db, rater_id, minutes)
    systems = discover_systems()

    return {
        "rater_id": rater_id,
        "total_stimuli": len(assignments),
        "num_systems": len(systems),
    }


@app.get("/api/next/{rater_id}")
def get_next_stimulus(rater_id: str, db=Depends(get_db)):
    row = db.execute("""
        SELECT a.system_name, a.sent_id, a.sort_order
        FROM rater_assignments a
        LEFT JOIN ratings r ON r.rater_id = a.rater_id
            AND r.system_name = a.system_name
            AND r.sent_id = a.sent_id
        WHERE a.rater_id = ? AND r.id IS NULL
        ORDER BY a.sort_order
        LIMIT 1
    """, (rater_id,)).fetchone()

    if not row:
        total = db.execute(
            "SELECT COUNT(*) FROM ratings WHERE rater_id=?", (rater_id,)
        ).fetchone()[0]
        return {"done": True, "progress": total}

    total = db.execute(
        "SELECT COUNT(*) FROM rater_assignments WHERE rater_id=?", (rater_id,)
    ).fetchone()[0]
    done = db.execute(
        "SELECT COUNT(*) FROM ratings WHERE rater_id=?", (rater_id,)
    ).fetchone()[0]

    sent_info = _sentences.get(row["sent_id"], {})

    return {
        "done": False,
        "system_name": row["system_name"],
        "sent_id": row["sent_id"],
        "text_dev": sent_info.get("text_dev", ""),
        "text_roman": sent_info.get("text_roman", ""),
        "progress": done,
        "total": total,
    }


@app.get("/api/audio/{system_name:path}/{sent_id}")
def serve_audio(system_name: str, sent_id: str):
    systems = discover_systems()
    if system_name not in systems or sent_id not in systems[system_name]:
        raise HTTPException(status_code=404, detail="Audio not found")

    path = Path(systems[system_name][sent_id])
    media_types = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".webm": "audio/webm", ".flac": "audio/flac"}
    return FileResponse(path, media_type=media_types.get(path.suffix, "audio/mpeg"))


class RatingSubmit(BaseModel):
    rater_id: str
    system_name: str
    sent_id: str
    score: int


@app.post("/api/ratings")
def submit_rating(body: RatingSubmit, db=Depends(get_db)):
    if body.score < 1 or body.score > 5:
        raise HTTPException(status_code=400, detail="Score must be 1-5")

    assignment = db.execute(
        "SELECT * FROM rater_assignments WHERE rater_id=? AND system_name=? AND sent_id=?",
        (body.rater_id, body.system_name, body.sent_id),
    ).fetchone()
    if not assignment:
        # Auto-assign for admin/voices-page ratings
        db.execute(
            "INSERT OR IGNORE INTO rater_assignments (rater_id, system_name, sent_id, sort_order) VALUES (?,?,?,0)",
            (body.rater_id, body.system_name, body.sent_id),
        )

    db.execute(
        "DELETE FROM ratings WHERE rater_id=? AND system_name=? AND sent_id=?",
        (body.rater_id, body.system_name, body.sent_id),
    )

    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        "INSERT INTO ratings (rater_id, system_name, sent_id, score, created_at) VALUES (?,?,?,?,?)",
        (body.rater_id, body.system_name, body.sent_id, body.score, now),
    )
    db.commit()
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Pair discrimination test
# ---------------------------------------------------------------------------

class PairTestStart(BaseModel):
    name: str
    native_speaker: bool = True
    minutes: int = 10


@app.post("/api/pair-test/start")
def start_pair_test(body: PairTestStart, db=Depends(get_db)):
    rater_id = uuid.uuid4().hex[:8]
    now = datetime.now(timezone.utc).isoformat()

    db.execute(
        "INSERT INTO raters (id, name, native_speaker, device_info, created_at) VALUES (?,?,?,?,?)",
        (rater_id, body.name, 1 if body.native_speaker else 0, "pair-test", now),
    )
    db.commit()

    # Build assignments: pick pairs × systems
    systems = discover_systems()
    sys_names = sorted(s for s in systems if s not in ("mms_tts", "natural_speech"))

    # Get valid pairs
    pair_groups = {}
    for sid, info in _sentences.items():
        pid = info.get("pair_id", "")
        if not pid:
            continue
        if pid not in pair_groups:
            pair_groups[pid] = []
        pair_groups[pid].append({"sent_id": sid, **info})

    valid_pairs = []
    for pid, sents in pair_groups.items():
        if len(sents) != 2:
            continue
        a, b = sents[0], sents[1]
        available = [s for s in sys_names
                     if a["sent_id"] in systems.get(s, {}) and b["sent_id"] in systems.get(s, {})]
        if available:
            valid_pairs.append({"pair_id": pid, "a": a, "b": b, "systems": available})

    # Budget: ~15 seconds per trial
    total_budget = (body.minutes * 60) // 15
    n_pairs = max(1, total_budget // len(sys_names)) if sys_names else 0
    selected_pairs = random.sample(valid_pairs, min(n_pairs, len(valid_pairs)))

    # Create assignments: each selected pair × each available system
    assignments = []
    for p in selected_pairs:
        for sys in p["systems"]:
            assignments.append({
                "system": sys,
                "pair_id": p["pair_id"],
                "sent_a": p["a"]["sent_id"],
                "sent_b": p["b"]["sent_id"],
                "text_a": p["a"].get("text_dev", ""),
                "text_b": p["b"].get("text_dev", ""),
                "word_a": p["a"].get("contrast_word", ""),
                "word_b": p["b"].get("contrast_word", ""),
            })

    random.shuffle(assignments)

    return {
        "rater_id": rater_id,
        "assignments": assignments,
        "total": len(assignments),
    }


class PairSubmit(BaseModel):
    rater_id: str
    system_name: str
    pair_id: str
    sent_id_a: str
    sent_id_b: str
    can_distinguish: int


@app.post("/api/pair-test/submit")
def submit_pair_rating(body: PairSubmit, db=Depends(get_db)):
    # Delete existing rating for same rater/system/pair
    db.execute(
        "DELETE FROM pair_ratings WHERE rater_id=? AND system_name=? AND pair_id=?",
        (body.rater_id, body.system_name, body.pair_id),
    )

    now = datetime.now(timezone.utc).isoformat()
    db.execute(
        """INSERT INTO pair_ratings (rater_id, system_name, pair_id, sent_id_a, sent_id_b, can_distinguish, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (body.rater_id, body.system_name, body.pair_id, body.sent_id_a, body.sent_id_b, body.can_distinguish, now),
    )
    db.commit()
    return {"status": "ok"}


@app.get("/api/pair-test/results")
def pair_test_results(db=Depends(get_db)):
    rows = db.execute("""
        SELECT system_name, pair_id,
               SUM(can_distinguish) as yes_count,
               COUNT(*) as total,
               ROUND(AVG(can_distinguish) * 100, 1) as pct_distinguish
        FROM pair_ratings
        GROUP BY system_name, pair_id
    """).fetchall()

    # Per-system aggregate
    sys_rows = db.execute("""
        SELECT system_name,
               SUM(can_distinguish) as yes_count,
               COUNT(*) as total,
               ROUND(AVG(can_distinguish) * 100, 1) as pct_distinguish
        FROM pair_ratings
        GROUP BY system_name
        ORDER BY pct_distinguish DESC
    """).fetchall()

    return {
        "per_pair": [dict(r) for r in rows],
        "per_system": [dict(r) for r in sys_rows],
        "total_ratings": db.execute("SELECT COUNT(*) FROM pair_ratings").fetchone()[0],
    }


@app.get("/api/results")
def get_results(db=Depends(get_db)):
    rows = db.execute("""
        SELECT system_name,
               COUNT(*) as n_ratings,
               ROUND(AVG(score), 2) as mean_mos,
               ROUND(AVG(score * score) - AVG(score) * AVG(score), 3) as variance
        FROM ratings
        GROUP BY system_name
        ORDER BY mean_mos DESC
    """).fetchall()

    total_raters = db.execute("SELECT COUNT(*) FROM raters").fetchone()[0]
    total_ratings = db.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]

    return {
        "total_raters": total_raters,
        "total_ratings": total_ratings,
        "systems": [dict(r) for r in rows],
    }


# ---------------------------------------------------------------------------
# Admin dashboard
# ---------------------------------------------------------------------------

def verify_password(password: str):
    if not secrets.compare_digest(password, ADMIN_PASSWORD):
        raise HTTPException(status_code=401, detail="Wrong password")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


@app.delete("/api/dashboard/delete-rater")
def delete_rater(password: str = Query(...), rater_id: str = Query(...), db=Depends(get_db)):
    verify_password(password)

    # Count ratings before delete
    n = db.execute("SELECT COUNT(*) FROM ratings WHERE rater_id=?", (rater_id,)).fetchone()[0]

    # Delete ratings, assignments, and rater
    db.execute("DELETE FROM ratings WHERE rater_id=?", (rater_id,))
    db.execute("DELETE FROM rater_assignments WHERE rater_id=?", (rater_id,))
    db.execute("DELETE FROM pair_ratings WHERE rater_id=?", (rater_id,))
    db.execute("DELETE FROM raters WHERE id=?", (rater_id,))
    db.commit()

    return {"status": "deleted", "ratings_deleted": n}


@app.get("/api/dashboard")
def dashboard_data(password: str = Query(...), db=Depends(get_db)):
    verify_password(password)

    # Overall stats
    total_raters = db.execute("SELECT COUNT(*) FROM raters").fetchone()[0]
    total_ratings = db.execute("SELECT COUNT(*) FROM ratings").fetchone()[0]
    total_assigned = db.execute("SELECT COUNT(*) FROM rater_assignments").fetchone()[0]

    # Per-system MOS
    systems = db.execute("""
        SELECT system_name,
               COUNT(*) as n_ratings,
               ROUND(AVG(score), 2) as mean_mos,
               ROUND(AVG(score * score) - AVG(score) * AVG(score), 3) as variance,
               MIN(score) as min_score,
               MAX(score) as max_score
        FROM ratings
        GROUP BY system_name
        ORDER BY mean_mos DESC
    """).fetchall()

    # Per-rater breakdown
    raters = db.execute("""
        SELECT r.id, r.name, r.native_speaker, r.created_at,
               COUNT(rt.id) as n_rated,
               (SELECT COUNT(*) FROM rater_assignments WHERE rater_id = r.id) as n_assigned
        FROM raters r
        LEFT JOIN ratings rt ON rt.rater_id = r.id
        GROUP BY r.id
        ORDER BY r.created_at DESC
    """).fetchall()

    # Score distribution per system
    distributions = {}
    for row in db.execute("""
        SELECT system_name, score, COUNT(*) as cnt
        FROM ratings
        GROUP BY system_name, score
        ORDER BY system_name, score
    """).fetchall():
        sys = row["system_name"]
        if sys not in distributions:
            distributions[sys] = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        distributions[sys][row["score"]] = row["cnt"]

    # Per-rater per-system scores
    rater_scores = {}
    for row in db.execute("""
        SELECT rater_id, system_name, ROUND(AVG(score), 2) as avg_score
        FROM ratings
        GROUP BY rater_id, system_name
    """).fetchall():
        rid = row["rater_id"]
        if rid not in rater_scores:
            rater_scores[rid] = {}
        rater_scores[rid][row["system_name"]] = row["avg_score"]

    return {
        "total_raters": total_raters,
        "total_ratings": total_ratings,
        "total_assigned": total_assigned,
        "completion_pct": round(total_ratings / total_assigned * 100, 1) if total_assigned else 0,
        "systems": [dict(r) for r in systems],
        "raters": [dict(r) for r in raters],
        "distributions": distributions,
        "rater_scores": rater_scores,
    }
