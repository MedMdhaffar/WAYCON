from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sqlite3
import threading
import time

import numpy as np
import pytest

from forensics.global_memory import GlobalMemory
import forensics.global_memory.backup as backup_module
from forensics.global_memory.backup import BackupTimeoutError, backup_database


def _profile(values=(1.0, 0.0, 0.0)) -> dict:
    vector = np.asarray(values, dtype=np.float32)
    vector /= np.linalg.norm(vector)
    return {
        "face_embedding": vector.tolist(),
        "face_crops": [],
        "appearance": {"date": "2026-07-16"},
        "best_body_crops": [],
        "video_sources": [],
    }


def _count(path: Path, table: str) -> int:
    connection = sqlite3.connect(str(path))
    try:
        return connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
    finally:
        connection.close()


def test_backup_creates_valid_independent_database(tmp_path):
    source = tmp_path / "memory.db"
    memory = GlobalMemory(source)
    memory.register(_profile())
    memory.close()
    source_before = source.read_bytes()
    destination = tmp_path / "backups" / "memory.db"

    result = backup_database(source, destination)

    assert result.destination_path == destination.resolve()
    assert result.size_bytes == destination.stat().st_size
    assert result.sha256 == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert _count(destination, "persons") == _count(source, "persons") == 1
    connection = sqlite3.connect(str(destination))
    try:
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()
    assert source.read_bytes() == source_before


def test_wal_backed_source_is_copied_consistently_while_open(tmp_path):
    source = tmp_path / "memory.db"
    writer = GlobalMemory(source)
    try:
        writer.register(_profile((1.0, 0.0, 0.0)))
        writer.register(_profile((0.0, 1.0, 0.0)))
        assert writer._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

        destination = tmp_path / "snapshot.db"
        backup_database(source, destination)

        assert _count(destination, "persons") == 2
        assert _count(destination, "identity_match_suggestions") == 0
    finally:
        writer.close()


def test_backup_rejects_invalid_destination_safely(tmp_path):
    source = tmp_path / "memory.db"
    memory = GlobalMemory(source)
    memory.close()

    with pytest.raises(ValueError, match="must differ"):
        backup_database(source, source)

    destination = tmp_path / "existing.db"
    destination.write_bytes(b"keep")
    with pytest.raises(FileExistsError, match="already exists"):
        backup_database(source, destination)
    assert destination.read_bytes() == b"keep"


def test_backup_missing_source_does_not_create_destination(tmp_path):
    destination = tmp_path / "backup" / "memory.db"
    with pytest.raises(FileNotFoundError):
        backup_database(tmp_path / "missing.db", destination)
    assert not destination.exists()


@pytest.mark.parametrize("raced_content", ["empty", "sqlite", "bytes"])
def test_backup_atomic_reservation_preserves_raced_destination(
    monkeypatch,
    tmp_path,
    raced_content,
):
    source = tmp_path / "source.db"
    memory = GlobalMemory(source)
    memory.register(_profile())
    memory.close()
    destination = tmp_path / "backup.db"
    original_open = backup_module.os.open

    def racing_open(path, flags, mode=0o777):
        if Path(path) == destination and flags & os.O_EXCL:
            if raced_content == "empty":
                destination.write_bytes(b"")
            elif raced_content == "sqlite":
                unrelated = sqlite3.connect(str(destination))
                unrelated.execute("CREATE TABLE unrelated(value TEXT)")
                unrelated.execute("INSERT INTO unrelated VALUES ('preserve')")
                unrelated.commit()
                unrelated.close()
            else:
                destination.write_bytes(b"unrelated non-SQLite bytes")
        return original_open(path, flags, mode)

    monkeypatch.setattr(backup_module.os, "open", racing_open)
    with pytest.raises(FileExistsError):
        backup_database(source, destination)

    if raced_content == "empty":
        assert destination.read_bytes() == b""
    elif raced_content == "sqlite":
        connection = sqlite3.connect(str(destination))
        try:
            assert connection.execute("SELECT value FROM unrelated").fetchone()[0] == "preserve"
        finally:
            connection.close()
    else:
        assert destination.read_bytes() == b"unrelated non-SQLite bytes"


def _create_delete_journal_source(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=0.1)
    assert connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
    connection.execute("CREATE TABLE evidence(value INTEGER)")
    connection.execute("INSERT INTO evidence VALUES (1)")
    connection.commit()
    return connection


def test_backup_timeout_is_bounded_and_removes_owned_partial_destination(tmp_path):
    source = tmp_path / "source.db"
    destination = tmp_path / "backup.db"
    holder = _create_delete_journal_source(source)
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("UPDATE evidence SET value=2")
    started = time.monotonic()
    try:
        with pytest.raises(BackupTimeoutError, match="time limit"):
            backup_database(source, destination, timeout_seconds=0.25)
    finally:
        elapsed = time.monotonic() - started
        holder.execute("ROLLBACK")
        holder.close()

    assert elapsed < 2.0
    assert not destination.exists()
    connection = sqlite3.connect(str(source))
    try:
        assert connection.execute("SELECT value FROM evidence").fetchone()[0] == 1
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        connection.close()


def test_backup_succeeds_when_exclusive_lock_released_before_deadline(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "source.db"
    destination = tmp_path / "backup.db"
    holder = _create_delete_journal_source(source)
    holder.execute("BEGIN EXCLUSIVE")
    reserved = threading.Event()
    original_open = backup_module.os.open
    result: list[object] = []
    errors: list[BaseException] = []

    def observed_open(path, flags, mode=0o777):
        descriptor = original_open(path, flags, mode)
        if Path(path) == destination and flags & os.O_EXCL:
            reserved.set()
        return descriptor

    def create_backup():
        try:
            result.append(backup_database(source, destination, timeout_seconds=2.0))
        except BaseException as exc:
            errors.append(exc)

    monkeypatch.setattr(backup_module.os, "open", observed_open)
    worker = threading.Thread(target=create_backup)
    worker.start()
    try:
        assert reserved.wait(timeout=1.0)
        holder.execute("ROLLBACK")
        holder.close()
        worker.join(timeout=3.0)
    finally:
        if holder:
            try:
                holder.close()
            except sqlite3.Error:
                pass

    assert not worker.is_alive()
    assert errors == []
    assert len(result) == 1
    assert _count(destination, "evidence") == 1


@pytest.mark.parametrize(
    "timeout_seconds",
    [0.0, -1.0, float("nan"), float("inf"), float("-inf")],
)
def test_backup_rejects_invalid_timeout_before_creating_destination(
    tmp_path,
    timeout_seconds,
):
    source = tmp_path / "source.db"
    memory = GlobalMemory(source)
    memory.close()
    destination = tmp_path / "backup.db"

    with pytest.raises(ValueError, match="finite and greater than zero"):
        backup_database(source, destination, timeout_seconds=timeout_seconds)

    assert not destination.exists()
