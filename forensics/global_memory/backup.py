from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import sqlite3


@dataclass(frozen=True)
class BackupResult:
    destination_path: Path
    size_bytes: int
    sha256: str


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
) -> BackupResult:
    """Create and verify a consistent SQLite backup without exposing rows."""
    source = Path(source_db)
    destination = Path(destination_db)
    if not source.is_file():
        raise FileNotFoundError("source SQLite database does not exist")
    if source.resolve() == destination.resolve():
        raise ValueError("source and destination database paths must differ")
    if destination.exists():
        raise FileExistsError("backup destination already exists")

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_connection: sqlite3.Connection | None = None
    destination_connection: sqlite3.Connection | None = None
    try:
        source_connection = sqlite3.connect(_readonly_uri(source), uri=True)
        source_connection.execute("PRAGMA query_only=ON")
        source_connection.execute("PRAGMA busy_timeout=5000")
        destination_connection = sqlite3.connect(str(destination))
        source_connection.backup(destination_connection)
        destination_connection.commit()
        result = destination_connection.execute("PRAGMA integrity_check").fetchone()
        if result is None or str(result[0]).lower() != "ok":
            raise sqlite3.DatabaseError("backup failed SQLite integrity_check")
    except BaseException:
        if destination_connection is not None:
            destination_connection.close()
            destination_connection = None
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
