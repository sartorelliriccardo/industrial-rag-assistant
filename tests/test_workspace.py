import time

from rag_assistant.workspace import WorkspaceStore


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make_store(active: dict[str, bool], grace: float = 15.0):
    clock = Clock()
    return WorkspaceStore(lambda sid: active[sid], grace_seconds=grace, clock=clock), clock


def test_workspaces_are_created_once_per_session():
    store, _ = make_store({"a": True})
    first = store.get_or_create("a")
    assert store.get_or_create("a") is first
    assert len(store) == 1


def test_connected_session_is_never_removed():
    store, clock = make_store({"a": True})
    store.get_or_create("a").files["manual.pdf"] = b"data"
    clock.now += 10_000
    assert store.sweep() == 0
    assert store.get("a").files == {"manual.pdf": b"data"}


def test_disconnected_session_is_removed_only_after_the_grace_period():
    active = {"a": False}
    store, clock = make_store(active, grace=15)
    store.get_or_create("a").files["manual.pdf"] = b"data"

    assert store.sweep() == 0  # first look: starts the countdown
    clock.now += 14
    assert store.sweep() == 0  # still within the grace period
    clock.now += 2
    assert store.sweep() == 1  # grace elapsed
    assert store.get("a") is None and len(store) == 0


def test_reconnecting_inside_the_grace_period_keeps_the_data():
    active = {"a": False}
    store, clock = make_store(active, grace=15)
    store.get_or_create("a").files["manual.pdf"] = b"data"
    store.sweep()
    clock.now += 10
    active["a"] = True  # e.g. a short network blip
    store.sweep()
    clock.now += 100
    active["a"] = False
    assert store.sweep() == 0  # the countdown restarted
    assert store.get("a") is not None


def test_only_the_disconnected_session_is_removed():
    active = {"a": True, "b": False}
    store, clock = make_store(active, grace=5)
    store.get_or_create("a")
    store.get_or_create("b")
    store.sweep()
    clock.now += 6
    assert store.sweep() == 1
    assert store.get("a") is not None and store.get("b") is None


def test_unknown_session_state_never_deletes_anything():
    def broken(_sid: str) -> bool:
        raise RuntimeError("runtime not available")

    clock = Clock()
    store = WorkspaceStore(broken, grace_seconds=1, clock=clock)
    store.get_or_create("a")
    store.sweep()
    clock.now += 1000
    assert store.sweep() == 0
    assert store.get("a") is not None


def test_discard_removes_a_workspace_on_demand():
    store, _ = make_store({"a": True})
    store.get_or_create("a")
    assert store.discard("a") is True
    assert store.discard("a") is False


def test_a_removed_workspace_is_emptied_in_place_for_anyone_still_holding_it():
    """The Streamlit script keeps module-level names pointing at these containers."""

    class FakePipeline:
        cleared = False

        def clear(self):
            self.cleared = True

    active = {"a": False}
    store, clock = make_store(active, grace=1)
    workspace = store.get_or_create("a")
    files, pipeline = workspace.files, FakePipeline()
    files["manual.pdf"] = b"x" * 1000
    workspace.pipeline = pipeline

    store.sweep()
    clock.now += 2
    assert store.sweep() == 1
    assert files == {}  # the same dict object the script still references
    assert pipeline.cleared is True
    assert workspace.pipeline is None


def test_discard_also_empties_the_workspace():
    store, _ = make_store({"a": True})
    workspace = store.get_or_create("a")
    workspace.files["manual.pdf"] = b"data"
    store.discard("a")
    assert workspace.files == {}


def test_background_sweeper_removes_a_disconnected_session():
    active = {"a": False}
    store = WorkspaceStore(lambda sid: active[sid], grace_seconds=0.05)
    store.get_or_create("a").files["manual.pdf"] = b"data"
    store.start_sweeper(interval_seconds=0.01)
    try:
        deadline = time.time() + 3
        while store.get("a") is not None and time.time() < deadline:
            time.sleep(0.02)
        assert store.get("a") is None
    finally:
        store.stop_sweeper()
