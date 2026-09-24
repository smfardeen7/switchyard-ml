import pytest
from switchyard.control import PolicyStore, Conflict, InvalidRelease, route_version

MODELS = [
    dict(version="digits-logreg-v1", accuracy=0.97),
    dict(version="digits-rf-v2", accuracy=0.96),
    dict(version="bad", accuracy=0.80),
]


def test_quality_gate_and_compare_swap(tmp_path):
    store = PolicyStore(tmp_path / "state.db", MODELS)
    with pytest.raises(InvalidRelease):
        store.start("bad", 0.25, 0)
    p = store.start("digits-rf-v2", 0.25, 0)
    assert p["revision"] == 1 and p["status"] == "running"
    with pytest.raises(Conflict):
        store.rollback(0, "stale")
    with pytest.raises(Conflict):
        store.start("digits-rf-v2", 0.25, 1)
    p = store.rollback(1, "operator stopped")
    assert p["champion"] == "digits-logreg-v1" and p["canary"] is None
    assert len(store.events()) == 2
    store.close()


def test_restart_rolls_back_running_candidate(tmp_path):
    path = tmp_path / "state.db"
    store = PolicyStore(path, MODELS)
    store.start("digits-rf-v2", 0.25, 0)
    store.close()
    restarted = PolicyStore(path, MODELS)
    assert restarted.get()["status"] == "rolled_back"
    assert restarted.get()["revision"] == 2
    assert restarted.events()[0]["type"] == "restart_rollback"
    restarted.close()


def test_promote_updates_champion_and_stale_rollout_cannot_rollback(tmp_path):
    store = PolicyStore(tmp_path / "state.db", MODELS)
    old = store.start("digits-rf-v2", 0.25, 0)
    promoted = store.promote(old["revision"])
    assert promoted["champion"] == "digits-rf-v2"
    store.start("digits-logreg-v1", 0.25, promoted["revision"])
    with pytest.raises(Conflict):
        store.rollback(old["revision"], "old request", automatic=True)
    assert store.get()["status"] == "running"
    store.close()


def test_sticky_routing_distribution(tmp_path):
    store = PolicyStore(tmp_path / "state.db", MODELS)
    p = store.start("digits-rf-v2", 0.25, 0)
    routed = [route_version(p, str(i)) for i in range(10000)]
    assert routed == [route_version(p, str(i)) for i in range(10000)]
    assert 0.23 < routed.count("digits-rf-v2") / len(routed) < 0.27
    store.close()


def test_restart_fails_if_registered_champion_missing(tmp_path):
    path = tmp_path / "state.db"
    store = PolicyStore(path, MODELS)
    store.close()
    with pytest.raises(InvalidRelease):
        PolicyStore(path, MODELS[1:])


def test_failed_store_initialization_closes_connection(tmp_path, monkeypatch):
    import sqlite3

    path = tmp_path / "state.db"
    store = PolicyStore(path, MODELS)
    store.close()
    opened = []
    connect = sqlite3.connect

    def tracking(*args, **kwargs):
        db = connect(*args, **kwargs)
        opened.append(db)
        return db

    monkeypatch.setattr(sqlite3, "connect", tracking)
    with pytest.raises(InvalidRelease):
        PolicyStore(path, MODELS[1:])
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        opened[0].execute("SELECT 1")
