"""Blind pairwise human-preference arena.

A listener hears the same prompt from two anonymous systems (A and B) and answers
"Which sounds more natural, as though spoken by a real person?": A, B, both
natural, or neither natural. Names are revealed only after the vote. Ratings are
Bradley-Terry with listener-cluster bootstrap intervals (see bt.py).

Quality controls:
- A listening check before the first vote: hear a clip, pick its Devanagari text
  from four options, including its minimal-pair partner. Three of four must be correct.
- A vote counts only after both clips were fetched and enough time has passed
  to hear the longer one.
- Hidden checks, about one in eight battles after warm-up: an identical clip on
  both sides (expect no preference), or a clip against a deliberately degraded
  copy of itself. Listeners who fail most of them are excluded from the ratings.
- A per-listener cap on votes involving any one system.

Only approved submissions whose owners allowed listener playback ever appear.
Clips are re-encoded to one format and loudness (arena_audio.py) and served only
under opaque per-battle tokens.
"""

import csv
import hashlib
import io
import json
import os
import random
import secrets
import shutil
import time
import uuid
from typing import Literal

from fastapi import Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import BaseModel, ConfigDict, Field

from . import arena_audio, bt
from .contract import SENTENCES
from .storage import db

COOKIE = "arena_listener"
# Issue tags shown after a vote, per side. Nepali-specific failures are listed explicitly.
TAGS = [
    {"id": "mispronunciation", "ne": "गलत उच्चारण", "en": "Mispronounced word"},
    {"id": "schwa", "ne": "अकार लोप गल्ती", "en": "Wrong schwa deletion"},
    {"id": "numbers", "ne": "अङ्क/मिति गलत", "en": "Numbers or dates read wrong"},
    {"id": "code_switch", "ne": "अङ्ग्रेजी शब्द गलत", "en": "English word wrong"},
    {"id": "prosody", "ne": "अस्वाभाविक लय", "en": "Unnatural rhythm or intonation"},
    {"id": "robotic", "ne": "यान्त्रिक आवाज", "en": "Robotic or metallic"},
    {"id": "missing_words", "ne": "शब्द छुटेको/थपिएको", "en": "Words missing or added"},
    {"id": "cut_off", "ne": "बीचमै काटिएको", "en": "Cut off"},
    {"id": "noise", "ne": "आवाजमा खसखस", "en": "Noise or glitches"},
]
TAG_IDS = {t["id"] for t in TAGS}
SCORES = {"a": 1.0, "b": 0.0, "both_good": 0.5, "both_bad": 0.5}
SCREEN_ITEMS, SCREEN_PASS, SCREEN_ATTEMPTS = 4, 3, 3
SCREEN_MAX_CER = 0.15
CHECK_RATE, CHECK_WARMUP = 0.12, 3
PER_SYSTEM_CAP = 60
# One person cannot move the ranking by voting hundreds of times:
DAILY_CAP = 50  # comparisons per listener per 24 h, so attention stays fresh
LIFETIME_CAP = 200  # comparisons per listener in total
IP_DAILY_CAP = 300  # comparisons per network address per 24 h (fresh browser sessions)
VOTE_WEIGHT_CAP = 50  # a listener's votes count for at most this many in the ranking
MIN_GAP = 1.0
MIN_CHECKS, MAX_CHECK_FAIL = 2, 0.5
PROVISIONAL_VOTES = 30
# The public board is recomputed at most this often (seconds). A vote no longer forces a
# full bootstrap on the next page view; consent changes and new systems still do.
RATINGS_REFRESH = 30
MIN_CATEGORY_VOTES = 40  # a prompt category gets its own (point-estimate) ratings from here
# Optional, self-declared. Nepali differs by region; this lets the ranking be split later.
REGIONS = {
    "kathmandu_valley": ("काठमाडौं उपत्यका", "Kathmandu Valley"),
    "eastern": ("पूर्वी पहाड/हिमाल", "Eastern hills or mountains"),
    "central_western": ("गण्डकी/पश्चिम पहाड", "Gandaki or western hills"),
    "karnali_far_west": ("कर्णाली/सुदूरपश्चिम", "Karnali or Far-West"),
    "madhesh_terai": ("मधेस/तराई", "Madhesh or Terai"),
    "diaspora_south_asia": ("भारत/भुटानका नेपालीभाषी", "Nepali speaker in India or Bhutan"),
    "diaspora_other": ("अन्य देशमा", "Elsewhere abroad"),
}
KINDS = {"baseline": "Ampixa baseline", "human_reference": "Human reference"}


def _sample_sentence():
    """One fixed prompt for public samples. It never appears in battles or screening, so
    a sample's bytes can't be matched against a battle clip to unblind it."""
    ids = sorted(SENTENCES)
    phrases = [k for k in ids if SENTENCES[k].get("category") == "phrases" and not SENTENCES[k].get("pair_id")]
    return (phrases or ids)[0]


SAMPLE_SENTENCE = _sample_sentence()


def export_public():
    return os.environ.get("NEPTTS_ARENA_PUBLIC_EXPORT", "off").strip().lower() in ("on", "1", "true")
METHOD = (
    "Blind pairwise preference on identical prompts. Question: which sounds more natural, "
    "as though spoken by a real person? Bradley-Terry ratings on the Elo scale (ties count "
    "half), 95% intervals and rank ranges from 200 listener-cluster bootstrap resamples. "
    "Only screened listeners who pass hidden checks count. Each listener counts for at most "
    f"{VOTE_WEIGHT_CAP} votes: a listener with more has every vote scaled down. "
    "Clips are loudness-matched and "
    "re-encoded identically. Ratings compare systems within this arena only."
)


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


def display(manifest):
    return {
        "system": manifest.get("system_name", "Unknown"),
        "organization": manifest.get("organization", ""),
        "voice": manifest.get("voice", ""),
        "model_version": manifest.get("model_version", ""),
    }


def kinds(c):
    """{submission id: 'baseline' | 'human_reference'} for first-party entries."""
    return {r["submission_id"]: r["kind"] for r in c.execute("SELECT submission_id, kind FROM system_kinds")}


def consent_granted(c, sid, manifest):
    row = c.execute("SELECT granted FROM arena_consent WHERE submission_id=?", (sid,)).fetchone()
    if row is not None:
        return bool(row["granted"])
    return manifest.get("permission_to_play_audio_to_listeners") is True


def eligible(c):
    """Approved submissions whose owners allow anonymous listener playback."""
    out = {}
    for row in c.execute("SELECT id, manifest, report FROM submissions WHERE state='published'"):
        manifest = json.loads(row["manifest"])
        if consent_granted(c, row["id"], manifest):
            out[row["id"]] = {"manifest": manifest, "report": json.loads(row["report"] or "{}")}
    return out


def check_passed(battle, choice):
    if battle["kind"] == "null":
        return choice in ("both_good", "both_bad")
    clean = "b" if battle["degraded_side"] == "a" else "a"
    return choice == clean


def install_arena(app, root, rate, origin, current_user, admin):
    secure = origin.startswith("https://")
    app.state.arena_clock = time.time
    app.state.arena_rng = random.SystemRandom()
    cache = {}

    def now():
        return app.state.arena_clock()

    def listener(request, c, required=True):
        token = request.cookies.get(COOKIE, "")
        row = None
        if token:
            row = c.execute(
                "SELECT * FROM arena_listeners WHERE token_hash=?", (digest(token),)
            ).fetchone()
        if row is None and required:
            raise HTTPException(401, "Start a listening session first.")
        return row

    def screened(request, c):
        row = listener(request, c)
        if not row["screened"]:
            raise HTTPException(403, "Complete the listening check first.")
        return row

    def clip(sid, sent_id, degraded=False):
        try:
            return arena_audio.playback(root, sid, sent_id, degraded)
        except (OSError, RuntimeError, ValueError):
            raise HTTPException(404, "Audio unavailable.") from None

    def audio_response(path):
        # No ETag/Last-Modified: identical cache files would otherwise link battles.
        return Response(path.read_bytes(), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    class SessionBody(BaseModel):
        model_config = ConfigDict(extra="forbid")
        native_speaker: Literal[True]
        headphones: bool
        region: Literal[tuple(REGIONS)] | None = None

    @app.post("/api/arena/session")
    def start_session(body: SessionBody, request: Request, response: Response):
        with db(root) as c:
            existing = listener(request, c, required=False)
        if existing:
            return state(request)
        rate("arena-session:" + (request.client.host if request.client else "?"), 30)
        token = secrets.token_urlsafe(32)
        with db(root) as c:
            c.execute(
                "INSERT INTO arena_listeners(id,token_hash,created,native_declared,region) VALUES(?,?,?,1,?)",
                (uuid.uuid4().hex, digest(token), now(), body.region),
            )
        response.set_cookie(COOKIE, token, max_age=180 * 86400, httponly=True,
                            samesite="lax", secure=secure, path="/")
        return {"listener": {"screened": False, "votes": 0, "attempts_left": SCREEN_ATTEMPTS},
                **public_state()}

    def public_state():
        with db(root) as c:
            systems = len(eligible(c))
        return {"systems": systems, "ready": systems >= 2, "tags": TAGS,
                "regions": [{"id": k, "ne": ne, "en": en} for k, (ne, en) in REGIONS.items()]}

    def allowance(c, lid):
        today = c.execute("SELECT COUNT(*) FROM arena_battles WHERE listener_id=? AND served>?",
                          (lid, now() - 86400)).fetchone()[0]
        total = c.execute("SELECT COUNT(*) FROM arena_battles WHERE listener_id=?", (lid,)).fetchone()[0]
        return {"left_today": max(0, min(DAILY_CAP - today, LIFETIME_CAP - total)),
                "left_total": max(0, LIFETIME_CAP - total)}

    @app.get("/api/arena/state")
    def state(request: Request):
        with db(root) as c:
            row = listener(request, c, required=False)
            info = None
            if row:
                votes = c.execute(
                    "SELECT COUNT(*) FROM arena_votes WHERE listener_id=?", (row["id"],)
                ).fetchone()[0]
                info = {"screened": bool(row["screened"]), "votes": votes,
                        "attempts_left": SCREEN_ATTEMPTS - row["screen_attempts"],
                        **allowance(c, row["id"])}
        return {"listener": info, **public_state()}

    # ------------------------------------------------------------------ screening

    @app.post("/api/arena/screening")
    def start_screening(request: Request):
        rng = app.state.arena_rng
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = listener(request, c)
            if row["screened"]:
                raise HTTPException(409, "Listening check already passed.")
            if row["screen_attempts"] >= SCREEN_ATTEMPTS:
                raise HTTPException(403, "No listening-check attempts left.")
            systems = eligible(c)
            partners = {}
            for sid, s in SENTENCES.items():
                if s.get("pair_id"):
                    partners.setdefault(s["pair_id"], []).append(sid)
            candidates = [ids for ids in partners.values()
                          if len(ids) == 2 and SAMPLE_SENTENCE not in ids]
            rng.shuffle(candidates)
            items = []
            for pair in candidates:
                target = rng.choice(pair)
                partner = pair[0] if pair[1] == target else pair[1]
                sources = sorted(
                    (info["report"].get("per_sentence", {}).get(target, {}).get("cer", 9), sid)
                    for sid, info in systems.items()
                )
                if not sources or sources[0][0] > SCREEN_MAX_CER:
                    continue
                others = rng.sample([k for k in SENTENCES if k not in pair and k != SAMPLE_SENTENCE], 2)
                options = [target, partner, *others]
                rng.shuffle(options)
                items.append((sources[0][1], target, options, options.index(target)))
                if len(items) == SCREEN_ITEMS:
                    break
            if len(items) < SCREEN_ITEMS:
                raise HTTPException(503, "Not enough approved audio for the listening check yet.")
            attempt = row["screen_attempts"] + 1
            c.execute("UPDATE arena_listeners SET screen_attempts=? WHERE id=?", (attempt, row["id"]))
            c.execute("DELETE FROM arena_screen_items WHERE listener_id=?", (row["id"],))
            out = []
            for sid, target, options, answer in items:
                token = secrets.token_urlsafe(24)
                c.execute(
                    "INSERT INTO arena_screen_items VALUES(?,?,?,?,?,?,?,?)",
                    (digest(token), row["id"], attempt, sid, target, json.dumps(options), answer, now()),
                )
                out.append({"token": token, "audio": f"/api/arena/screening/{token}/audio",
                            "options": [SENTENCES[o]["text_devanagari"] for o in options]})
        return {"items": out, "pass_mark": SCREEN_PASS}

    @app.get("/api/arena/screening/{token}/audio")
    def screening_audio(token: str, request: Request):
        with db(root) as c:
            row = listener(request, c)
            item = c.execute(
                "SELECT * FROM arena_screen_items WHERE token_hash=? AND listener_id=?",
                (digest(token), row["id"]),
            ).fetchone()
        if not item:
            raise HTTPException(404)
        path, _ = clip(item["submission_id"], item["sent_id"])
        return audio_response(path)

    class ScreeningAnswers(BaseModel):
        answers: dict[str, int] = Field(max_length=SCREEN_ITEMS)

    @app.post("/api/arena/screening/answers")
    def answer_screening(body: ScreeningAnswers, request: Request):
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row = listener(request, c)
            items = c.execute(
                "SELECT * FROM arena_screen_items WHERE listener_id=?", (row["id"],)
            ).fetchall()
            if len(items) != SCREEN_ITEMS:
                raise HTTPException(409, "Start the listening check first.")
            given = {digest(k): v for k, v in body.answers.items()}
            if set(given) != {i["token_hash"] for i in items}:
                raise HTTPException(422, "Answer every item.")
            correct = sum(given[i["token_hash"]] == i["answer"] for i in items)
            passed = correct >= SCREEN_PASS
            c.execute("DELETE FROM arena_screen_items WHERE listener_id=?", (row["id"],))
            if passed:
                c.execute("UPDATE arena_listeners SET screened=? WHERE id=?", (now(), row["id"]))
        return {"passed": passed, "correct": correct,
                "attempts_left": SCREEN_ATTEMPTS - row["screen_attempts"]}

    # ------------------------------------------------------------------ battles

    @app.post("/api/arena/battles")
    def new_battle(request: Request):
        rng = app.state.arena_rng
        with db(root) as c:
            row = screened(request, c)
        # Rate limiting uses its own write transaction; never nest it inside ours.
        rate("arena-battle:" + row["id"], 600)
        rate("arena-ip:" + (request.client.host if request.client else "?"), IP_DAILY_CAP, 86400)
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            left = allowance(c, row["id"])
            if left["left_total"] == 0:
                raise HTTPException(409, f"You have reached the limit of {LIFETIME_CAP} comparisons. Thank you for listening!")
            if left["left_today"] == 0:
                raise HTTPException(429, f"You have reached today's limit of {DAILY_CAP} comparisons. Please come back tomorrow.")
            systems = sorted(eligible(c))
            if len(systems) < 2:
                raise HTTPException(409, "Fewer than two systems are available to compare.")
            mine = c.execute(
                "SELECT sent_id, sub_a, sub_b, kind FROM arena_battles WHERE listener_id=?",
                (row["id"],),
            ).fetchall()
            used = {b["sent_id"] for b in mine}
            prompts = [s for s in SENTENCES if s != SAMPLE_SENTENCE]
            pool = [s for s in prompts if s not in used] or prompts
            sent_id = rng.choice(pool)
            kind, side = "compare", None
            if len(mine) >= CHECK_WARMUP and rng.random() < CHECK_RATE:
                kind = rng.choice(["null", "degraded"])
                sub_a = sub_b = rng.choice(systems)
                side = rng.choice(["a", "b"]) if kind == "degraded" else None
            else:
                seen = {}
                for b in mine:
                    if b["kind"] == "compare":
                        for s in (b["sub_a"], b["sub_b"]):
                            seen[s] = seen.get(s, 0) + 1
                open_systems = [s for s in systems if seen.get(s, 0) < PER_SYSTEM_CAP]
                counts = {}
                for r in c.execute(
                    "SELECT b.sub_a, b.sub_b, COUNT(*) AS n FROM arena_battles b "
                    "JOIN arena_votes v ON v.battle_id=b.id WHERE b.kind='compare' AND v.valid=1 "
                    "GROUP BY b.sub_a, b.sub_b"
                ):
                    key = tuple(sorted((r["sub_a"], r["sub_b"])))
                    counts[key] = counts.get(key, 0) + r["n"]
                pairs = [(x, y) for i, x in enumerate(open_systems) for y in open_systems[i + 1:]]
                if not pairs:
                    raise HTTPException(409, "You have reached the vote limit for every system. Thank you!")
                # Prefer pairs with few votes so every comparison gets evidence.
                weights = [1 / (1 + counts.get(p, 0)) ** 2 for p in pairs]
                sub_a, sub_b = rng.choices(pairs, weights)[0]
                if rng.random() < 0.5:
                    sub_a, sub_b = sub_b, sub_a
            token = secrets.token_urlsafe(24)
            c.execute(
                "INSERT INTO arena_battles(token_hash,listener_id,sent_id,sub_a,sub_b,kind,degraded_side,served) "
                "VALUES(?,?,?,?,?,?,?,?)",
                (digest(token), row["id"], sent_id, sub_a, sub_b, kind, side, now()),
            )
        return {
            "token": token,
            "left_today": left["left_today"] - 1,
            "text": SENTENCES[sent_id]["text_devanagari"],
            "audio": {s: f"/api/arena/battles/{token}/{s}" for s in ("a", "b")},
        }

    def battle_for(token, request, c):
        row = listener(request, c)
        battle = c.execute(
            "SELECT * FROM arena_battles WHERE token_hash=? AND listener_id=?",
            (digest(token), row["id"]),
        ).fetchone()
        if not battle:
            raise HTTPException(404, "Battle not found.")
        return row, battle

    @app.get("/api/arena/battles/{token}/{side}")
    def battle_audio(token: str, side: Literal["a", "b"], request: Request):
        with db(root) as c:
            _, battle = battle_for(token, request, c)
        sid = battle["sub_a"] if side == "a" else battle["sub_b"]
        path, duration = clip(sid, battle["sent_id"], battle["degraded_side"] == side)
        with db(root) as c:
            c.execute(
                f"UPDATE arena_battles SET fetched_{side}=COALESCE(fetched_{side},?), duration_{side}=? WHERE id=?",
                (now(), duration, battle["id"]),
            )
        return audio_response(path)

    class Vote(BaseModel):
        model_config = ConfigDict(extra="forbid")
        choice: Literal["a", "b", "both_good", "both_bad"]
        tags: dict[Literal["a", "b"], list[str]] = Field(default_factory=dict)

    @app.post("/api/arena/battles/{token}/vote")
    def vote(token: str, body: Vote, request: Request):
        tags = {k: sorted(set(v)) for k, v in body.tags.items()}
        if any(t not in TAG_IDS for v in tags.values() for t in v):
            raise HTTPException(422, "Unknown issue tag.")
        with db(root) as c:
            c.execute("BEGIN IMMEDIATE")
            row, battle = battle_for(token, request, c)
            if c.execute("SELECT 1 FROM arena_votes WHERE battle_id=?", (battle["id"],)).fetchone():
                raise HTTPException(409, "You already voted on this pair.")
            if battle["fetched_a"] is None or battle["fetched_b"] is None:
                raise HTTPException(409, "Play both clips before voting.")
            elapsed = now() - battle["served"]
            longest = max(battle["duration_a"] or 0, battle["duration_b"] or 0)
            valid, reason = 1, ""
            if elapsed < longest + MIN_GAP:
                valid, reason = 0, "too_fast"
            c.execute(
                "INSERT INTO arena_votes VALUES(?,?,?,?,?,?,?,?)",
                (battle["id"], row["id"], body.choice, json.dumps(tags), now(), elapsed, valid, reason),
            )
            manifests = {
                r["id"]: json.loads(r["manifest"])
                for r in c.execute(
                    "SELECT id, manifest FROM submissions WHERE id IN (?,?)",
                    (battle["sub_a"], battle["sub_b"]),
                )
            }
        reveal = {s: display(manifests.get(battle["sub_" + s], {})) for s in ("a", "b")}
        if battle["kind"] == "null":
            note = "Listening check: both clips were identical."
        elif battle["kind"] == "degraded":
            reveal[battle["degraded_side"]]["system"] += " (deliberately degraded)"
            note = "Listening check: one clip was deliberately degraded."
        else:
            note = ""
        return {"reveal": reveal, "kind": "check" if note else "compare", "note": note}

    # ------------------------------------------------------------------ results

    def ratings():
        with db(root) as c:
            systems_now = tuple(sorted(eligible(c)))
            key = tuple(c.execute("SELECT COUNT(*), MAX(battle_id) FROM arena_votes").fetchone()) + systems_now
            cached = cache.get("board")
            if cached and (cached["key"] == key or (
                    cached["key"][2:] == systems_now and now() - cached["at"] < RATINGS_REFRESH)):
                return cached["result"]
            rows = c.execute(
                "SELECT v.listener_id, v.choice, v.tags, v.valid, v.reason, v.created, b.kind, b.sent_id, "
                "b.sub_a, b.sub_b, b.degraded_side, l.screened, l.region FROM arena_votes v "
                "JOIN arena_battles b ON b.id=v.battle_id JOIN arena_listeners l ON l.id=v.listener_id "
                "ORDER BY v.battle_id"
            ).fetchall()
            manifests = {
                r["id"]: json.loads(r["manifest"])
                for r in c.execute("SELECT id, manifest FROM submissions WHERE state='published'")
            }
            entry_kinds = kinds(c)
        checks = {}
        for r in rows:
            if r["kind"] != "compare":
                passed, total = checks.get(r["listener_id"], (0, 0))
                checks[r["listener_id"]] = (passed + check_passed(r, r["choice"]), total + 1)
        excluded = {
            lid for lid, (passed, total) in checks.items()
            if total >= MIN_CHECKS and (total - passed) / total > MAX_CHECK_FAIL
        }
        votes, issues, appearances, invalid, counted, by_category = [], {}, {}, {}, [], {}
        for r in rows:
            if r["kind"] != "compare" or not r["screened"]:
                continue
            if r["sub_a"] not in manifests or r["sub_b"] not in manifests:
                continue
            if not r["valid"] or r["listener_id"] in excluded:
                reason = r["reason"] or "excluded_listener"
                invalid[reason] = invalid.get(reason, 0) + 1
                continue
            votes.append((r["listener_id"], r["sub_a"], r["sub_b"], SCORES[r["choice"]]))
            category = SENTENCES.get(r["sent_id"], {}).get("category", "unknown")
            by_category.setdefault(category, []).append(votes[-1])
            counted.append(r)
            tags = json.loads(r["tags"])
            for side in ("a", "b"):
                sid = r["sub_" + side]
                appearances[sid] = appearances.get(sid, 0) + 1
                for t in tags.get(side, []):
                    issues.setdefault(sid, {}).setdefault(t, 0)
                    issues[sid][t] += 1
        systems = sorted({v[1] for v in votes} | {v[2] for v in votes})
        categories = {}
        for category, cat_votes in sorted(by_category.items()):
            if len(cat_votes) < MIN_CATEGORY_VOTES:
                continue
            cat_systems = sorted({v[1] for v in cat_votes} | {v[2] for v in cat_votes})
            weighted, _ = bt.cap_listener_weight(cat_votes, VOTE_WEIGHT_CAP)
            categories[category] = {"votes": len(cat_votes),
                                    "elo": bt.fit(cat_systems, weighted) if len(cat_systems) > 1 else {}}
        votes, capped = bt.cap_listener_weight(votes, VOTE_WEIGHT_CAP)
        entries = []
        for r in bt.leaderboard(systems, votes):
            sid = r.pop("system")
            share = {t: n / appearances[sid] for t, n in issues.get(sid, {}).items()}
            entries.append({
                "id": sid, **display(manifests[sid]), **r, "kind": entry_kinds.get(sid),
                "provisional": r["votes"] < PROVISIONAL_VOTES,
                "issues": dict(sorted(share.items(), key=lambda x: -x[1])[:3]),
            })
        regions = {}
        for region in {r["listener_id"]: r["region"] for r in counted}.values():
            regions[region or "undeclared"] = regions.get(region or "undeclared", 0) + 1
        result = {
            "entries": entries,
            "updated": now(),
            "categories": categories,
            "category_note": f"Point estimates for prompt categories with at least {MIN_CATEGORY_VOTES} "
                             "counted votes; no intervals.",
            "listener_regions": regions,
            "sample_sentence": {"id": SAMPLE_SENTENCE, "text": SENTENCES[SAMPLE_SENTENCE]["text_devanagari"]},
            "kinds": KINDS,
            "export_public": export_public(),
            "_counted": counted,
            "votes": len(votes),
            "listeners": len({v[0] for v in votes}),
            "excluded_listeners": len(excluded),
            "capped_listeners": len(capped),
            "excluded_votes": invalid,
            "method": METHOD,
            "tags": TAGS,
        }
        cache["board"] = {"key": key, "at": now(), "result": result}
        return result

    @app.get("/api/arena/leaderboard")
    def arena_leaderboard():
        result = dict(ratings())
        result.pop("excluded_votes")
        result.pop("_counted")
        return result

    @app.get("/api/arena/samples/{sid}")
    def sample(sid: str):
        """A fixed, never-battled prompt from one listed system, for the leaderboard player."""
        with db(root) as c:
            if sid not in eligible(c):
                raise HTTPException(404, "Sample unavailable.")
        path, _ = clip(sid, SAMPLE_SENTENCE)
        return Response(path.read_bytes(), media_type="audio/wav",
                        headers={"Cache-Control": "public, max-age=3600"})

    @app.get("/api/arena/votes.csv")
    def export_votes(request: Request):
        """Counted votes, pseudonymised, so anyone can refit the ranking.

        Public only with NEPTTS_ARENA_PUBLIC_EXPORT=on; otherwise reviewers only.
        Listener IDs become per-export sequence numbers; times are rounded to the UTC day.
        """
        if not export_public():
            admin(request)
        result = ratings()
        with db(root) as c:
            names = {r["id"]: json.loads(r["manifest"]).get("system_name", "")
                     for r in c.execute("SELECT id, manifest FROM submissions")}
        pseudonym = {}
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(["listener", "region", "date_utc", "sent_id", "category", "system_a_id",
                         "system_a", "system_b_id", "system_b", "choice", "score_a", "tags_a", "tags_b"])
        for r in result["_counted"]:
            tags = json.loads(r["tags"])
            writer.writerow([
                pseudonym.setdefault(r["listener_id"], len(pseudonym) + 1), r["region"] or "",
                time.strftime("%Y-%m-%d", time.gmtime(r["created"])), r["sent_id"],
                SENTENCES.get(r["sent_id"], {}).get("category", ""),
                r["sub_a"], names.get(r["sub_a"], ""), r["sub_b"], names.get(r["sub_b"], ""),
                r["choice"], SCORES[r["choice"]], " ".join(tags.get("a", [])), " ".join(tags.get("b", [])),
            ])
        return PlainTextResponse(out.getvalue(), media_type="text/csv", headers={
            "Content-Disposition": 'attachment; filename="ampixa-arena-votes.csv"'})

    @app.get("/api/admin/arena", dependencies=[Depends(admin)])
    def arena_admin():
        result = ratings()
        with db(root) as c:
            listeners = c.execute(
                "SELECT COUNT(*) AS n, SUM(screened IS NOT NULL) AS screened FROM arena_listeners"
            ).fetchone()
            battles = c.execute("SELECT kind, COUNT(*) AS n FROM arena_battles GROUP BY kind").fetchall()
            systems = sorted(eligible(c))
        return {
            "listeners": listeners["n"], "screened": listeners["screened"] or 0,
            "battles": {b["kind"]: b["n"] for b in battles},
            "counted_votes": result["votes"], "excluded_votes": result["excluded_votes"],
            "excluded_listeners": result["excluded_listeners"], "eligible_systems": systems,
            "listener_regions": result["listener_regions"],
        }

    # ------------------------------------------------------------------ consent

    class Consent(BaseModel):
        granted: bool

    @app.post("/api/submissions/{sid}/listening-consent")
    def listening_consent(sid: str, body: Consent, request: Request):
        """Only the owning account or the private receipt holder may decide."""
        with db(root) as c:
            row = c.execute("SELECT * FROM submissions WHERE id=?", (sid,)).fetchone()
            if not row:
                raise HTTPException(404, "Submission not found.")
            identity = current_user(request)
            token = request.headers.get("x-submission-token", "")
            if identity and row["owner_id"] == identity["github_id"]:
                via = "owner"
            elif token and secrets.compare_digest(digest(token), row["token_hash"]):
                via = "receipt"
            else:
                raise HTTPException(404, "Submission not found.")
            c.execute(
                "INSERT OR REPLACE INTO arena_consent VALUES(?,?,?,?)",
                (sid, int(body.granted), now(), via),
            )
            c.execute(
                "INSERT INTO events(submission_id,time,state,message) VALUES(?,?,?,?)",
                (sid, time.time(), row["state"],
                 "Listener playback " + ("allowed." if body.granted else "withdrawn; existing votes are kept.")),
            )
        if not body.granted:
            shutil.rmtree(root / "arena_cache" / sid, ignore_errors=True)
        cache.clear()
        return {"id": sid, "listening_consent": body.granted}
