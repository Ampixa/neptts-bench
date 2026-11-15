import math
import random

from submission_app import bt


def simulate(strengths, n=3000, listeners=40, seed=1):
    rng = random.Random(seed)
    names = list(strengths)
    votes = []
    for k in range(n):
        a, b = rng.sample(names, 2)
        p = 1 / (1 + 10 ** ((strengths[b] - strengths[a]) / 400))
        votes.append((f"l{k % listeners}", a, b, 1.0 if rng.random() < p else 0.0))
    return votes


def test_recovers_true_order_and_scale():
    truth = {"x": 1200, "y": 1050, "z": 950, "w": 800}
    ratings = bt.fit(truth, simulate(truth))
    assert sorted(ratings, key=ratings.get, reverse=True) == ["x", "y", "z", "w"]
    gap = ratings["x"] - ratings["w"]
    assert 330 < gap < 470  # true gap 400, slightly shrunk by the anchor prior
    assert math.isclose(sum(ratings.values()) / 4, bt.CENTER, abs_tol=1e-6)


def test_ties_are_half_wins_and_zero_win_systems_stay_finite():
    tied = bt.fit(["a", "b"], [("l", "a", "b", 0.5)] * 20)
    assert math.isclose(tied["a"], tied["b"])
    lopsided = bt.fit(["a", "b"], [("l", "a", "b", 1.0)] * 20)
    assert math.isfinite(lopsided["b"]) and lopsided["a"] > lopsided["b"]


def test_leaderboard_intervals_and_rank_ranges():
    truth = {"x": 1200, "y": 1000, "z": 990}
    rows = bt.leaderboard(truth, simulate(truth, n=900), resamples=100)
    assert rows[0]["system"] == "x"
    for r in rows:
        assert r["ci95"][0] <= r["elo"] <= r["ci95"][1]
        assert 1 <= r["rank_range"][0] <= r["rank"] <= r["rank_range"][1] <= 3
    by = {r["system"]: r for r in rows}
    assert by["x"]["rank_range"] == [1, 1]
    assert by["y"]["rank_range"][1] >= by["z"]["rank_range"][0]  # 10 Elo apart: statistically tied


def test_listener_bootstrap_is_wider_than_vote_bootstrap_for_one_heavy_listener():
    # One listener casts most votes; resampling listeners must reflect that.
    votes = [("heavy", "a", "b", 1.0)] * 60 + [(f"l{i}", "a", "b", float(i % 2)) for i in range(10)]
    rows = {r["system"]: r for r in bt.leaderboard(["a", "b"], votes, resamples=200)}
    low, high = rows["a"]["ci95"]
    assert high - low > 100


def test_needs_two_systems():
    assert bt.leaderboard(["only"], []) == []


def test_one_heavy_listener_cannot_outvote_the_crowd():
    # One account votes A over B 500 times; 20 listeners each prefer B five times.
    votes = [("spammer", "A", "B", 1.0)] * 500
    votes += [(f"l{i}", "A", "B", 0.0) for i in range(20) for _ in range(5)]
    assert bt.fit(["A", "B"], votes)["A"] > 1000  # uncapped, the spammer wins
    weighted, capped = bt.cap_listener_weight(votes, 50)
    assert capped == {"spammer"}
    assert sum(v[4] for v in weighted if v[0] == "spammer") == 50
    assert bt.fit(["A", "B"], weighted)["B"] > bt.fit(["A", "B"], weighted)["A"]


def test_weights_below_the_cap_change_nothing():
    votes = [(f"l{i % 7}", "A", "B", 1.0 if i % 3 else 0.0) for i in range(40)]
    weighted, capped = bt.cap_listener_weight(votes, 50)
    assert capped == set()
    assert bt.fit(["A", "B"], weighted) == bt.fit(["A", "B"], votes)
