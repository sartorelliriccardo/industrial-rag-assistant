"""Per-session working data (uploaded PDFs and their search index) with prompt cleanup.

Why not leave this in Streamlit's session state? When a browser tab closes, Streamlit keeps
the session (state and uploaded files) in memory for `server.disconnectedSessionTTL`
seconds (120 by default) and only drops it lazily, when some other session disconnects.
On a quiet demo that can mean a closed session's documents stay in memory for a long time
(measured, see README). Here the documents live in a store of our own, and a small
background thread discards a workspace a few seconds after its session stops being active.

This module does not import Streamlit: whether a session is still connected is injected.
"""

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_GRACE_SECONDS = 15.0  # tolerates a page refresh or a short network blip
DEFAULT_SWEEP_INTERVAL = 2.0


@dataclass
class Workspace:
    """Everything one browser session owns."""

    session_id: str
    pipeline: Any = None  # RagPipeline (kept loose: this module stays dependency-free)
    config_key: tuple | None = None  # the settings the pipeline was built with
    files: dict[str, bytes] = field(default_factory=dict)  # name -> PDF bytes
    disconnected_since: float | None = None

    def release(self) -> None:
        """Empty the workspace in place.

        Dropping the reference is not enough: the Streamlit script keeps module-level names
        (and a fragment keeps its whole namespace) that still point at `files` and
        `pipeline`. Clearing them in place frees the PDF bytes and the extracted text no
        matter who else still holds the (now empty) containers.
        """
        self.files.clear()
        if self.pipeline is not None and hasattr(self.pipeline, "clear"):
            self.pipeline.clear()
        self.pipeline = None
        self.config_key = None


class WorkspaceStore:
    """Thread-safe map session id -> Workspace, with a sweeper for disconnected sessions."""

    def __init__(
        self,
        is_active: Callable[[str], bool],
        grace_seconds: float = DEFAULT_GRACE_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._is_active = is_active
        self._grace = grace_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._workspaces: dict[str, Workspace] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __len__(self) -> int:
        with self._lock:
            return len(self._workspaces)

    def get(self, session_id: str) -> Workspace | None:
        with self._lock:
            return self._workspaces.get(session_id)

    def get_or_create(self, session_id: str) -> Workspace:
        with self._lock:
            workspace = self._workspaces.get(session_id)
            if workspace is None:
                workspace = self._workspaces[session_id] = Workspace(session_id)
            return workspace

    def discard(self, session_id: str) -> bool:
        with self._lock:
            workspace = self._workspaces.pop(session_id, None)
        if workspace is None:
            return False
        workspace.release()
        return True

    def sweep(self) -> int:
        """Drop workspaces whose session has been inactive for longer than the grace period.

        If it cannot be determined whether a session is active, the workspace is kept: not
        being able to check must never delete the data of someone still using the app.
        """
        now = self._clock()
        expired: list[Workspace] = []
        with self._lock:
            for session_id, workspace in list(self._workspaces.items()):
                try:
                    active = self._is_active(session_id)
                except Exception:
                    active = True
                if active:
                    workspace.disconnected_since = None
                elif workspace.disconnected_since is None:
                    workspace.disconnected_since = now
                elif now - workspace.disconnected_since >= self._grace:
                    del self._workspaces[session_id]
                    expired.append(workspace)
        for workspace in expired:  # outside the lock: releasing can take a moment
            workspace.release()
        if expired:
            logger.info("discarded %d disconnected workspace(s)", len(expired))
        return len(expired)

    def start_sweeper(self, interval_seconds: float = DEFAULT_SWEEP_INTERVAL) -> None:
        if self._thread and self._thread.is_alive():
            return

        def loop() -> None:
            while not self._stop.wait(interval_seconds):
                try:
                    self.sweep()
                except Exception:  # a failing sweep must never kill the thread
                    logger.exception("workspace sweep failed")

        self._stop.clear()
        self._thread = threading.Thread(target=loop, name="workspace-sweeper", daemon=True)
        self._thread.start()

    def stop_sweeper(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


_store: WorkspaceStore | None = None
_store_lock = threading.Lock()


def get_store(
    is_active: Callable[[str], bool],
    grace_seconds: float = DEFAULT_GRACE_SECONDS,
    interval_seconds: float = DEFAULT_SWEEP_INTERVAL,
) -> WorkspaceStore:
    """The process-wide store (created, with its sweeper running, on first use)."""
    global _store
    with _store_lock:
        if _store is None:
            _store = WorkspaceStore(is_active, grace_seconds)
            _store.start_sweeper(interval_seconds)
        return _store
