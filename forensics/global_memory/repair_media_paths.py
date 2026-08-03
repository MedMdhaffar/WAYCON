from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from forensics.media_paths import MediaPathError, get_media_root, normalize_media_path
from forensics.global_memory.backup import backup_database


@dataclass
class RepairReport:
    scanned: int = 0
    valid: int = 0
    rewritten: int = 0
    stale: int = 0
    ambiguous: int = 0
    gallery_removed: int = 0
    profile_fallbacks: int = 0
    body_references_removed: int = 0
    dry_run: bool = True
    backup_path: str | None = None


def _basename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _recover(
    raw: str,
    *,
    root: Path,
    person_id: str,
    crop_type: str,
) -> tuple[str | None, str]:
    try:
        normalized = normalize_media_path(
            raw,
            media_root=root,
            allow_legacy_absolute=True,
            require_exists=True,
        )
        portable = raw.replace("\\", "/")
        return normalized, "valid" if portable == normalized else "rewritten"
    except (MediaPathError, FileNotFoundError, OSError):
        pass

    filename = _basename(raw)
    if not filename:
        return None, "stale"
    crop_dir = "face_crops" if crop_type == "face" else "body_crops"
    person_root = root / person_id / crop_dir
    try:
        candidates = sorted({
            candidate.resolve()
            for candidate in person_root.rglob(filename)
            if candidate.is_file()
            and candidate.name == filename
            and candidate.resolve().is_relative_to(root)
        })
    except OSError:
        return None, "stale"
    if len(candidates) > 1:
        return None, "ambiguous"
    if not candidates:
        return None, "stale"
    return normalize_media_path(candidates[0], media_root=root), "rewritten"


def _record(report: RepairReport, outcome: str) -> None:
    report.scanned += 1
    if outcome == "valid":
        report.valid += 1
    elif outcome == "rewritten":
        report.rewritten += 1
    elif outcome == "ambiguous":
        report.ambiguous += 1
    else:
        report.stale += 1


def repair_media_paths(
    database_path: str | Path,
    *,
    media_root: str | Path | None = None,
    dry_run: bool = True,
) -> dict:
    database = Path(database_path)
    if not database.is_file():
        raise FileNotFoundError("supplied Global Memory database does not exist")
    root = get_media_root(media_root)
    report = RepairReport(dry_run=bool(dry_run))
    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        backup = database.with_name(f"{database.name}.backup-{stamp}")
        backup_database(database, backup)
        report.backup_path = backup.name

    connection = sqlite3.connect(str(database))
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        if not dry_run:
            journal_mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()
            if journal_mode is None or str(journal_mode[0]).lower() != "wal":
                raise RuntimeError("media repair requires WAL journal mode")
            connection.execute("BEGIN IMMEDIATE")

        gallery_fallbacks: dict[str, list[tuple[float, str]]] = {}
        recognition_fallbacks: dict[str, list[str]] = {}

        gallery_rows = connection.execute(
            "SELECT id, person_id, crop_type, path, sharpness FROM person_gallery"
        ).fetchall()
        for row in gallery_rows:
            repaired, outcome = _recover(
                row["path"],
                root=root,
                person_id=row["person_id"],
                crop_type=row["crop_type"],
            )
            _record(report, outcome)
            if repaired:
                if row["crop_type"] == "face":
                    gallery_fallbacks.setdefault(row["person_id"], []).append(
                        (float(row["sharpness"] or 0.0), repaired)
                    )
                if not dry_run and repaired != row["path"]:
                    try:
                        connection.execute(
                            "UPDATE person_gallery SET path=? WHERE id=?",
                            (repaired, row["id"]),
                        )
                    except sqlite3.IntegrityError:
                        connection.execute(
                            "DELETE FROM person_gallery WHERE id=?",
                            (row["id"],),
                        )
                        report.gallery_removed += 1
            elif outcome == "stale" and not dry_run:
                connection.execute("DELETE FROM person_gallery WHERE id=?", (row["id"],))
                report.gallery_removed += 1

        log_rows = connection.execute(
            "SELECT id, person_id, best_face_crop FROM recognition_log "
            "WHERE best_face_crop IS NOT NULL"
        ).fetchall()
        for row in log_rows:
            repaired, outcome = _recover(
                row["best_face_crop"],
                root=root,
                person_id=row["person_id"],
                crop_type="face",
            )
            _record(report, outcome)
            if repaired:
                recognition_fallbacks.setdefault(row["person_id"], []).append(repaired)
            if not dry_run and repaired != row["best_face_crop"]:
                connection.execute(
                    "UPDATE recognition_log SET best_face_crop=? WHERE id=?",
                    (repaired, row["id"]),
                )

        appearance_rows = connection.execute(
            "SELECT id, person_id, best_body_crops FROM appearances"
        ).fetchall()
        for row in appearance_rows:
            try:
                raw_paths = json.loads(row["best_body_crops"] or "[]")
            except (TypeError, json.JSONDecodeError):
                raw_paths = []
            raw_paths = raw_paths if isinstance(raw_paths, list) else []
            repaired_paths: list[str] = []
            for raw in raw_paths:
                repaired, outcome = _recover(
                    str(raw),
                    root=root,
                    person_id=row["person_id"],
                    crop_type="body",
                )
                _record(report, outcome)
                if repaired and repaired not in repaired_paths:
                    repaired_paths.append(repaired)
                elif not repaired:
                    report.body_references_removed += 1
            if not dry_run and repaired_paths != raw_paths:
                connection.execute(
                    "UPDATE appearances SET best_body_crops=? WHERE id=?",
                    (json.dumps(repaired_paths), row["id"]),
                )

        person_rows = connection.execute(
            "SELECT person_id, profile_image FROM persons WHERE profile_image IS NOT NULL"
        ).fetchall()
        for row in person_rows:
            repaired, outcome = _recover(
                row["profile_image"],
                root=root,
                person_id=row["person_id"],
                crop_type="face",
            )
            _record(report, outcome)
            if not repaired:
                gallery = sorted(
                    gallery_fallbacks.get(row["person_id"], []),
                    key=lambda item: (-item[0], item[1]),
                )
                repaired = (
                    gallery[0][1]
                    if gallery
                    else next(iter(recognition_fallbacks.get(row["person_id"], [])), None)
                )
                if repaired:
                    report.profile_fallbacks += 1
            if not dry_run and repaired != row["profile_image"]:
                connection.execute(
                    "UPDATE persons SET profile_image=? WHERE person_id=?",
                    (repaired, row["person_id"]),
                )

        if not dry_run:
            connection.execute("COMMIT")
    except Exception:
        if not dry_run and connection.in_transaction:
            connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()
    return asdict(report)


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair portable WAYCON media references.")
    parser.add_argument("--database", required=True, help="SQLite database copy to inspect")
    parser.add_argument("--media-root", default=None, help="Configured person_db media root")
    parser.add_argument("--apply", action="store_true", help="Apply repairs; default is dry-run")
    args = parser.parse_args()
    report = repair_media_paths(
        args.database,
        media_root=args.media_root,
        dry_run=not args.apply,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
