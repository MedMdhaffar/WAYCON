"""Safely inspect or remove WAYCON-generated runtime state.

Usage:
    python3 -m forensics.reset_runtime
    python3 -m forensics.reset_runtime --dry-run
    python3 -m forensics.reset_runtime --yes
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
from pathlib import Path
from typing import Callable, Iterable


SERVICE_PORTS = {
    5009: "Person Creation backend",
    5010: "Face Engine",
    5175: "frontend (informational)",
}
BLOCKING_PORTS = {5009, 5010}
RUNTIME_RELATIVE_PATHS = (
    "forensics/global_memory.db",
    "forensics/global_memory.db-wal",
    "forensics/global_memory.db-shm",
    "forensics/global_memory.db-x-persons-1-embedding.bin",
    "forensics/person_db",
)
CACHE_DIRECTORY_NAMES = {
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}
CACHE_FILE_SUFFIXES = {".pyc", ".pyo"}
EXPLICIT_CACHE_RELATIVE_PATHS = (
    "forensics/person_creation/frontend/node_modules/.vite",
    "forensics/person_creation/frontend/dist",
)

_STOP_INSTRUCTIONS = (
    "Press Stop in the frontend; wait for capture shutdown, preprocessing "
    "drain, rolling-analysis drain, and bounded VLM finalization; confirm the "
    "job reached done or explicit failure; then stop the Person Creation "
    "backend. Stop Face Engine before the destructive reset as well."
)


def resolve_project_root(module_file: str | Path | None = None) -> Path:
    """Resolve the repository root from this module, never from the CWD."""
    source = Path(module_file or __file__).resolve()
    return source.parent.parent


def _is_port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.25):
            return True
    except OSError:
        return False


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False


def _deduplicate_targets(paths: Iterable[Path], root: Path) -> list[Path]:
    unique: list[Path] = []
    for path in sorted(
        {item.absolute() for item in paths},
        key=lambda item: (len(item.parts), str(item).lower()),
    ):
        if not _is_within(path, root):
            raise ValueError(f"reset candidate escapes project root: {path}")
        if any(path == parent or parent in path.parents for parent in unique):
            continue
        unique.append(path)
    return sorted(unique, key=lambda item: str(item).lower())


def runtime_candidates(project_root: Path) -> list[Path]:
    root = project_root.absolute()
    candidates = [root / relative for relative in RUNTIME_RELATIVE_PATHS]
    candidates.extend(
        root / relative for relative in EXPLICIT_CACHE_RELATIVE_PATHS
    )
    protected_tree_names = {
        ".git",
        ".venv",
        "env",
        "venv",
        "node_modules",
        "huggingface",
        "person_db",
    }
    for directory, dirnames, filenames in os.walk(root, topdown=True):
        current = Path(directory)
        retained = []
        for name in dirnames:
            if name in CACHE_DIRECTORY_NAMES:
                candidates.append(current / name)
                continue
            if (
                name in protected_tree_names
                or name.startswith(".venv-")
                or name.startswith("models--")
            ):
                continue
            retained.append(name)
        dirnames[:] = retained
        candidates.extend(
            current / name
            for name in filenames
            if Path(name).suffix.lower() in CACHE_FILE_SUFFIXES
        )
    return _deduplicate_targets(candidates, root)


def _path_metrics(path: Path) -> tuple[int, int]:
    if path.is_symlink() or path.is_file():
        return 1, path.lstat().st_size
    files = 0
    byte_count = 0
    for child in path.rglob("*"):
        if child.is_file() or child.is_symlink():
            files += 1
            try:
                byte_count += child.lstat().st_size
            except OSError:
                pass
    return files, byte_count


def _delete_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    else:
        shutil.rmtree(path)


def _display_path(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def reset_runtime(
    *,
    project_root: str | Path | None = None,
    destructive: bool = False,
    port_checker: Callable[[int], bool] = _is_port_open,
    delete_path: Callable[[Path], None] = _delete_path,
    output: Callable[[str], None] = print,
) -> dict:
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else resolve_project_root()
    )
    ports = {
        str(port): {
            "service": service,
            "active": bool(port_checker(port)),
            "blocking": port in BLOCKING_PORTS,
        }
        for port, service in SERVICE_PORTS.items()
    }
    output(f"WAYCON project root: {root}")
    for port, status in ports.items():
        activity = "ACTIVE" if status["active"] else "stopped"
        output(f"Port {port} ({status['service']}): {activity}")

    candidates = runtime_candidates(root)
    output("Reset candidates:")
    for path in candidates:
        state = "present" if path.exists() or path.is_symlink() else "absent"
        output(f"  [{state}] {_display_path(path, root)}")

    summary = {
        "project_root": str(root),
        "mode": "destructive" if destructive else "dry-run",
        "ports": ports,
        "deleted_paths": [],
        "absent_paths": [],
        "skipped_paths": [],
        "failed_paths": [],
        "files_removed": 0,
        "bytes_removed": 0,
        "completed_fully": True,
    }

    active_blockers = [
        f"{status['service']} on port {port}"
        for port, status in ports.items()
        if status["active"] and status["blocking"]
    ]
    if destructive and active_blockers:
        summary["completed_fully"] = False
        for path in candidates:
            label = _display_path(path, root)
            if path.exists() or path.is_symlink():
                summary["skipped_paths"].append({
                    "path": label,
                    "reason": "active_service",
                })
            else:
                summary["absent_paths"].append(label)
        output("REFUSED: active service(s): " + ", ".join(active_blockers))
        output(_STOP_INSTRUCTIONS)
        output("RESET_SUMMARY " + json.dumps(summary, sort_keys=True))
        return summary

    for path in candidates:
        label = _display_path(path, root)
        if not path.exists() and not path.is_symlink():
            summary["absent_paths"].append(label)
            continue
        if not destructive:
            summary["skipped_paths"].append({
                "path": label,
                "reason": "dry_run",
            })
            continue

        files, byte_count = _path_metrics(path)
        try:
            delete_path(path)
        except OSError as exc:
            summary["completed_fully"] = False
            failure = {
                "path": label,
                "error": f"{type(exc).__name__}: {exc}",
            }
            summary["failed_paths"].append(failure)
            output(f"ERROR deleting {label}: {failure['error']}")
            continue
        summary["deleted_paths"].append(label)
        summary["files_removed"] += files
        summary["bytes_removed"] += byte_count

    output("RESET_SUMMARY " + json.dumps(summary, sort_keys=True))
    return summary


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely inspect or remove WAYCON-generated runtime state."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Show candidates without deleting them (the default).",
    )
    mode.add_argument(
        "--yes",
        action="store_true",
        help="Delete known runtime paths after service safety checks.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = reset_runtime(destructive=bool(args.yes))
    return 0 if summary["completed_fully"] else 1


if __name__ == "__main__":
    sys.exit(main())
