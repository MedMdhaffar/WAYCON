"""Live-progress status callback, kept out of PersonCreationState on purpose.

`PersonCreationState` previously carried `_status_callback: Callable[..., Any]` so
nodes/process_live_stream.py could report progress back to service.py's job registry.
A bare Python callable is not JSON/pickle-serializable, which is one of the reasons a
LangGraph checkpointer (MemorySaver) was dropped from graph.py in an earlier commit --
any future checkpointing/resumability work (segment crash-recovery replay) needs state
to stay serializable end to end.

A ContextVar does the same job without touching state: `service.py` sets it once
before calling `graph.stream(...)`, and any node running synchronously in that call
(the graph has no parallel/threaded node execution today) can read it back via
`notify()`. Nothing here is part of the graph's data -- it never gets checkpointed,
never needs to serialize.
"""

from __future__ import annotations

import contextvars
from typing import Any, Callable

_status_callback_var: "contextvars.ContextVar[Callable[..., Any] | None]" = contextvars.ContextVar(
    "person_creation_status_callback", default=None
)


def set_status_callback(callback: Callable[..., Any] | None) -> contextvars.Token:
    return _status_callback_var.set(callback)


def reset_status_callback(token: contextvars.Token) -> None:
    _status_callback_var.reset(token)


def notify(status: str, snapshot_update: dict | None = None) -> None:
    callback = _status_callback_var.get()
    if callable(callback):
        callback(status, snapshot_update)
