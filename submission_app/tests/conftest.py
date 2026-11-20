import pytest


@pytest.fixture(autouse=True)
def manual_review(monkeypatch):
    """Most tests drive the reviewer by hand; test_autoreview switches automation on."""
    monkeypatch.setenv("NEPTTS_AUTO_REVIEW", "off")


@pytest.fixture(autouse=True)
def rani_unpinned(monkeypatch):
    """Tests start from an unpinned Rani contract, whatever the checked-in pin is.

    Tests that need Rani pin a fake checkpoint themselves. Imported here, after
    submission_app has put the evaluator toolkit on sys.path.
    """
    import submission_app.evaluation  # noqa: F401, I001 - must run first: it adds neptts_eval to sys.path
    from neptts_eval import rani

    monkeypatch.setattr(rani, "PINS", {**rani.PINS, "checkpoint_sha256": None, "packages": None})


@pytest.fixture(autouse=True)
def fresh_arena_ratings(monkeypatch):
    """Tests read the board right after voting; production refreshes it on a timer."""
    from submission_app import arena

    monkeypatch.setattr(arena, "RATINGS_REFRESH", 0)
