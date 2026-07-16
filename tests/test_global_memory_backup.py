from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from forensics.global_memory import GlobalMemory
from forensics.global_memory.backup import backup_database


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
