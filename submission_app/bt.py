"""Bradley-Terry ratings for blind pairwise votes, with listener-cluster bootstrap.

A vote is (listener, a, b, score_a) with score_a 1 (A preferred), 0 (B preferred)
or 0.5 ("both natural" / "neither natural"). Ties count as half a win each way,
as in Voice Arena's methodology.

Strengths are fitted with the Hunter (2004) minorise-maximise update. Every system
also plays one virtual tie against a fixed anchor of strength 1. That keeps a
system with zero wins finite and shrinks ratings with few votes towards the
mean. Ratings are reported on the Elo scale (400 points = 10x odds), centred on 1000.

Uncertainty resamples whole listeners, not single votes. A prolific listener
then cannot make the intervals look narrower than the evidence supports.

A vote may carry a fifth element, its weight (default 1). `cap_listener_weight`
scales a prolific listener's votes so their total weight never exceeds a cap:
500 votes from one person then move the ratings no more than 50 would.
"""

import math
import random

ELO_SCALE = 400 / math.log(10)
CENTER = 1000.0


def cap_listener_weight(votes, cap):
    """Weight each listener's votes so they sum to at most `cap`. Returns (votes, capped ids)."""
    totals = {}
    for vote in votes:
        totals[vote[0]] = totals.get(vote[0], 0) + 1
    capped = {lid for lid, n in totals.items() if n > cap}
    weighted = [(lid, a, b, score, cap / totals[lid] if lid in capped else 1.0)
                for lid, a, b, score, *_ in votes]
    return weighted, capped


def fit(systems, votes, iterations=2000, tol=1e-10):
    """Return {system: elo} for the given votes (listener id is ignored here)."""
    systems = list(systems)
    index = {s: i for i, s in enumerate(systems)}
    n = len(systems)
    wins = [0.5] * n  # half of the virtual tie against the anchor
    games = {}
    for _, a, b, score, *weight in votes:
        i, j = index[a], index[b]
        if i == j:
            continue
        w = weight[0] if weight else 1.0
        wins[i] += score * w
        wins[j] += (1 - score) * w
        key = (min(i, j), max(i, j))
        games[key] = games.get(key, 0) + w
    p = [1.0] * n
    for _ in range(iterations):
        denom = [1 / (x + 1) for x in p]  # one virtual game against the anchor
        for (i, j), count in games.items():
            share = count / (p[i] + p[j])
            denom[i] += share
            denom[j] += share
        new = [wins[i] / denom[i] for i in range(n)]
        mean_log = sum(math.log(x) for x in new) / n
        new = [x / math.exp(mean_log) for x in new]
        delta = max(abs(math.log(x) - math.log(y)) for x, y in zip(new, p))
        p = new
        if delta < tol:
            break
    return {s: CENTER + ELO_SCALE * math.log(p[index[s]]) for s in systems}


def _percentile(values, q):
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def _ranks(ratings):
    ordered = sorted(ratings, key=ratings.get, reverse=True)
    return {s: ordered.index(s) + 1 for s in ratings}


def leaderboard(systems, votes, resamples=200, seed=20261114):
    """Point ratings plus 95% bootstrap intervals and rank ranges per system."""
    systems = sorted(set(systems))
    if len(systems) < 2:
        return []
    point = fit(systems, votes)
    by_listener = {}
    for vote in votes:
        by_listener.setdefault(vote[0], []).append(vote)
    listeners = sorted(by_listener)
    rng = random.Random(seed)
    draws = {s: [] for s in systems}
    rank_draws = {s: [] for s in systems}
    for _ in range(resamples if listeners else 0):
        sample = [v for _ in listeners for v in by_listener[rng.choice(listeners)]]
        ratings = fit(systems, sample, iterations=500, tol=1e-8)
        ranks = _ranks(ratings)
        for s in systems:
            draws[s].append(ratings[s])
            rank_draws[s].append(ranks[s])
    counts = {s: {"votes": 0, "wins": 0.0} for s in systems}
    for _, a, b, score, *_ in votes:
        counts[a]["votes"] += 1
        counts[b]["votes"] += 1
        counts[a]["wins"] += score
        counts[b]["wins"] += 1 - score
    ranks = _ranks(point)
    rows = []
    for s in systems:
        row = {
            "system": s,
            "elo": point[s],
            "rank": ranks[s],
            "votes": counts[s]["votes"],
            "win_rate": counts[s]["wins"] / counts[s]["votes"] if counts[s]["votes"] else None,
            "ci95": None,
            "rank_range": None,
        }
        if draws[s]:
            row["ci95"] = [_percentile(draws[s], 0.025), _percentile(draws[s], 0.975)]
            row["rank_range"] = [round(_percentile(rank_draws[s], 0.025)),
                                 round(_percentile(rank_draws[s], 0.975))]
        rows.append(row)
    return sorted(rows, key=lambda r: r["rank"])
