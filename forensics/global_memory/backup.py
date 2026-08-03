from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import time


@dataclass(frozen=True)
class BackupResult:
    destination_path: Path
    size_bytes: int
    sha256: str


class BackupTimeoutError(TimeoutError):
    """A SQLite backup did not complete before its deadline."""


def _readonly_uri(path: Path) -> str:
    return f"{path.resolve().as_uri()}?mode=ro"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_database(
    source_db: str | Path,
    destination_db: str | Path,
    *,
    timeout_seconds: float = 10.0,
) -> BackupResult:
    """Create and verify a consistent SQLite backup without exposing rows."""
    timeout = float(timeout_seconds)
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("backup timeout_seconds must be finite and greater than zero")

    source = Path(source_db)
    destination = Path(destination_db)
    if not source.is_file():
        raise FileNotFoundError("source SQLite database does not exist")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination database paths must differ")
    destination.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    destination_owned = False
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        try:
            descriptor = os.open(
                destination,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
        except FileExistsError as exc:
            raise FileExistsError("backup destination already exists") from exc
        destination_owned = True
        os.close(descriptor)

        source_connection = sqlite3.connect(
            _readonly_uri(source),
            uri=True,
            timeout=min(timeout, 0.1),
        )
        source_connection.execute("PRAGMA query_only=ON")
        source_connection.execute(
            f"PRAGMA busy_timeout={max(1, min(100, int(timeout * 1000)))}"
        )
        destination_connection = sqlite3.connect(str(destination))

        def enforce_deadline(status: int, remaining: int, total: int) -> None:
            del status, remaining, total
            if time.monotonic() >= deadline:
                raise BackupTimeoutError("SQLite backup exceeded its time limit")

        source_connection.backup(
            destination_connection,
            pages=64,
            progress=enforce_deadline,
            sleep=0.05,
        )
        if time.monotonic() >= deadline:
            raise BackupTimeoutError("SQLite backup exceeded its time limit")
        destination_connection.commit()
        result = destination_connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or str(result[0]).lower() != "ok":
            raise sqlite3.DatabaseError("backup failed SQLite integrity_check")
    except BaseException:
        if destination_connection is not None:
            destination_connection.close()
            destination_connection = None
        if destination_owned:
            destination.unlink(missing_ok=True)
        raise
    finally:
        if destination_connection is not None:
            destination_connection.close()
        if source_connection is not None:
            source_connection.close()

    return BackupResult(
        destination_path=destination.resolve(),
        size_bytes=destination.stat().st_size,
        sha256=_sha256(destination),
    )
