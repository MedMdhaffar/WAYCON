from __future__ import annotations

from pathlib import Path

from forensics import reset_runtime as reset


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "forensics" / "person_creation" / "frontend").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / ".git").mkdir()
    return root


def _stopped(_port: int) -> bool:
    return False


def _populate(root: Path) -> dict[str, Path]:
    paths = {
        "database": root / "forensics" / "global_memory.db",
        "wal": root / "forensics" / "global_memory.db-wal",
        "sidecar": (
            root / "forensics"
            / "global_memory.db-x-persons-1-embedding.bin"
        ),
        "person_media": root / "forensics" / "person_db" / "person_001" / "face.jpg",
        "pycache": root / "forensics" / "__pycache__" / "module.pyc",
        "pytest_cache": root / ".pytest_cache" / "state",
        "vite": (
            root / "forensics" / "person_creation" / "frontend"
            / "node_modules" / ".vite" / "cache.js"
        ),
        "dist": (
            root / "forensics" / "person_creation" / "frontend"
            / "dist" / "index.html"
        ),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"runtime")
    return paths


def test_default_and_explicit_dry_run_delete_nothing(tmp_path):
    root = _project(tmp_path)
    paths = _populate(root)

    default = reset.reset_runtime(
        project_root=root,
        port_checker=_stopped,
        output=lambda _line: None,
    )
    explicit = reset.reset_runtime(
        project_root=root,
        destructive=False,
        port_checker=_stopped,
        output=lambda _line: None,
    )

    assert default["mode"] == explicit["mode"] == "dry-run"
    assert default["deleted_paths"] == explicit["deleted_paths"] == []
    assert default["skipped_paths"]
    assert all(path.exists() for path in paths.values())


def test_yes_deletes_only_expected_runtime_and_preserves_protected_files(tmp_path):
    root = _project(tmp_path)
    runtime = _populate(root)
    protected = [
        root / ".env",
        root / ".env.example",
        root / "source.py",
        root / "tests" / "fixture.jpg",
        root / "models" / "face.safetensors",
        root / "forensics" / "person_creation" / "frontend"
        / "node_modules" / "react" / "index.js",
    ]
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")

    summary = reset.reset_runtime(
        project_root=root,
        destructive=True,
        port_checker=_stopped,
        output=lambda _line: None,
    )

    assert summary["completed_fully"] is True
    assert summary["failed_paths"] == []
    assert summary["files_removed"] == len(runtime)
    assert summary["bytes_removed"] == len(runtime) * len(b"runtime")
    assert all(not path.exists() for path in runtime.values())
    assert all(path.exists() for path in protected)
    assert (
        root / "forensics" / "person_creation" / "frontend" / "node_modules"
    ).is_dir()


def test_missing_paths_are_harmless_and_reported_absent(tmp_path):
    root = _project(tmp_path)

    summary = reset.reset_runtime(
        project_root=root,
        destructive=True,
        port_checker=_stopped,
        output=lambda _line: None,
    )

    assert summary["completed_fully"] is True
    assert set(reset.RUNTIME_RELATIVE_PATHS) <= set(summary["absent_paths"])


def test_active_backend_refuses_destructive_reset(tmp_path):
    root = _project(tmp_path)
    paths = _populate(root)
    lines = []

    summary = reset.reset_runtime(
        project_root=root,
        destructive=True,
        port_checker=lambda port: port == 5009,
        output=lines.append,
    )

    assert summary["completed_fully"] is False
    assert summary["deleted_paths"] == []
    assert all(path.exists() for path in paths.values())
    assert any("REFUSED" in line for line in lines)
    assert any("Press Stop" in line for line in lines)


def test_deletion_failure_is_reported_and_makes_reset_incomplete(tmp_path):
    root = _project(tmp_path)
    paths = _populate(root)
    failed_target = paths["database"]

    def fail_one(path: Path):
        if path == failed_target:
            raise PermissionError("locked for test")
        reset._delete_path(path)

    summary = reset.reset_runtime(
        project_root=root,
        destructive=True,
        port_checker=_stopped,
        delete_path=fail_one,
        output=lambda _line: None,
    )

    assert summary["completed_fully"] is False
    assert summary["failed_paths"] == [{
        "path": "forensics/global_memory.db",
        "error": "PermissionError: locked for test",
    }]
    assert failed_target.exists()
    assert summary["deleted_paths"]
    assert summary["absent_paths"]


def test_project_root_resolution_does_not_depend_on_current_directory(
    tmp_path,
    monkeypatch,
):
    root = _project(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    resolved = reset.resolve_project_root(
        root / "forensics" / "reset_runtime.py"
    )

    assert resolved == root.resolve()
    assert Path.cwd() == elsewhere


def test_cli_returns_nonzero_for_incomplete_destructive_reset(monkeypatch):
    monkeypatch.setattr(
        reset,
        "reset_runtime",
        lambda **_kwargs: {"completed_fully": False},
    )

    assert reset.main(["--yes"]) == 1
    assert reset.main(["--dry-run"]) == 1


def test_summary_contains_all_required_categories(tmp_path):
    root = _project(tmp_path)
    paths = _populate(root)
    dry_run = reset.reset_runtime(
        project_root=root,
        port_checker=_stopped,
        output=lambda _line: None,
    )

    assert {
        "deleted_paths",
        "absent_paths",
        "skipped_paths",
        "failed_paths",
        "files_removed",
        "bytes_removed",
        "completed_fully",
    } <= dry_run.keys()
    assert dry_run["absent_paths"]
    assert dry_run["skipped_paths"]
    assert paths["database"].exists()
