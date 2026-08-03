"""Offline representative-comparison tool: isolation, safety, and shape."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from forensics.person_creation.tools import compare_cluster_representatives as tool


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _unit(vector) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def _write_memory(path: Path, embeddings) -> None:
    connection = sqlite3.connect(str(path))
    try:
        connection.execute(
            "create table persons ("
            "person_id text primary key, embedding blob, is_active integer)"
        )
        for index, embedding in enumerate(embeddings, start=1):
            connection.execute(
                "insert into persons values (?, ?, 1)",
                (
                    f"person_{index:03d}",
                    np.asarray(embedding, dtype=np.float32).tobytes(),
                ),
            )
        connection.commit()
    finally:
        connection.close()


def _cluster(members, representative):
    return {
        "cluster_id": 0,
        "face_count": len(members),
        "representative_embedding": list(representative),
        "face_records": [{"embedding": list(member)} for member in members],
    }


def _session_file(path: Path, clusters) -> Path:
    path.write_text(
        json.dumps({"identity_clusters": clusters}),
        encoding="utf-8",
    )
    return path


# --- isolation from live runtime ------------------------------------------------

def test_tool_is_never_imported_by_live_runtime_code():
    """No live module may import the offline comparison tool."""
    live_directory = REPOSITORY_ROOT / "forensics"
    offenders = []
    for source in live_directory.rglob("*.py"):
        relative = source.relative_to(REPOSITORY_ROOT).as_posix()
        if relative.endswith("tools/compare_cluster_representatives.py"):
            continue
        if "/tools/" in relative or "__pycache__" in relative:
            continue
        text = source.read_text(encoding="utf-8", errors="replace")
        if "compare_cluster_representatives" in text.replace(
            "tools/compare_cluster_representatives.py", ""
        ):
            offenders.append(relative)
    assert offenders == [], f"live modules reference the offline tool: {offenders}"


def test_live_analysis_no_longer_exposes_the_comparison():
    from forensics.person_creation import live_analysis

    assert not hasattr(live_analysis, "representative_similarity_diagnostic")


# --- shape / unit behaviour -----------------------------------------------------

def test_two_dimensional_shape_check_is_not_calibration(tmp_path):
    """Retained 2-D fixture: exercises plumbing only, never a calibration claim."""
    memory = tmp_path / "memory.db"
    _write_memory(memory, [_unit([1.0, 0.0]), _unit([0.0, 1.0])])
    members = [_unit([1.0, 0.0]), _unit([0.98, 0.2]), _unit([1.0, 0.0])]
    session = _session_file(
        tmp_path / "session.json",
        [_cluster(members, members[0])],
    )

    report = tool.build_report(
        session_path=session,
        database_path=memory,
        minimum_similarity=0.68,
        maximum_similarity=0.78,
    )

    assert report["status"] == "compared"
    assert report["current_strategy"] == "normalized_medoid"
    assert report["previous_strategy"] == "normalized_mean"
    assert report["compared_cluster_count"] == 1
    assert report["enrolled_identity_count"] == 2
    assert report["embedding_dimension"] == 2
    assert report["maximum_similarity_drift"] >= 0.0


def test_five_hundred_twelve_dimensional_multi_face_clusters(tmp_path):
    """The real measurement path: 512-D, multi-face, several clusters."""
    generator = np.random.default_rng(20260727)
    enrolled = [_unit(generator.normal(size=512)) for _ in range(20)]
    memory = tmp_path / "memory.db"
    _write_memory(memory, enrolled)

    clusters = []
    for base in enrolled[:5]:
        members = []
        for _ in range(6):
            members.append(_unit(base + generator.normal(scale=0.05, size=512)))
        similarities = np.vstack(members) @ np.vstack(members).T
        mean_similarity = (similarities.sum(axis=1) - 1.0) / (len(members) - 1)
        medoid = members[int(np.argmax(mean_similarity))]
        clusters.append(_cluster(members, medoid))
    session = _session_file(tmp_path / "session.json", clusters)

    report = tool.build_report(
        session_path=session,
        database_path=memory,
        minimum_similarity=0.68,
        maximum_similarity=0.78,
    )

    assert report["status"] == "compared"
    assert report["embedding_dimension"] == 512
    assert report["compared_cluster_count"] == 5
    assert report["minimum_face_count"] == 6
    assert report["maximum_face_count"] == 6
    for field in (
        "maximum_similarity_drift",
        "mean_similarity_drift",
        "best_match_identity_changed_count",
        "minimum_similarity_crossing_count",
        "maximum_similarity_crossing_count",
        "decision_band_change_count",
    ):
        assert field in report
    assert report["mean_similarity_drift"] <= report["maximum_similarity_drift"]


# --- anonymity ------------------------------------------------------------------

def test_report_contains_no_roster_and_no_raw_embeddings(tmp_path):
    generator = np.random.default_rng(11)
    enrolled = [_unit(generator.normal(size=512)) for _ in range(20)]
    memory = tmp_path / "memory.db"
    _write_memory(memory, enrolled)
    members = [_unit(enrolled[0] + generator.normal(scale=0.03, size=512))
               for _ in range(5)]
    session = _session_file(
        tmp_path / "session.json",
        [_cluster(members, members[0])],
    )

    report = tool.build_report(
        session_path=session,
        database_path=memory,
        minimum_similarity=0.68,
        maximum_similarity=0.78,
    )
    serialized = json.dumps(report)

    assert "person_" not in serialized
    assert "comparisons" not in report
    assert "person_id" not in serialized
    for value in report.values():
        assert not isinstance(value, list), (
            "aggregate report must not carry per-identity rows"
        )


# --- read-only guarantees -------------------------------------------------------

def test_inputs_are_not_modified(tmp_path):
    generator = np.random.default_rng(5)
    enrolled = [_unit(generator.normal(size=512)) for _ in range(4)]
    memory = tmp_path / "memory.db"
    _write_memory(memory, enrolled)
    members = [_unit(enrolled[0] + generator.normal(scale=0.02, size=512))
               for _ in range(4)]
    session = _session_file(
        tmp_path / "session.json",
        [_cluster(members, members[0])],
    )

    def digest(path: Path) -> str | None:
        return (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.exists() else None
        )

    before = {
        name: digest(tmp_path / name)
        for name in ("memory.db", "memory.db-wal", "memory.db-shm", "session.json")
    }
    tool.build_report(
        session_path=session,
        database_path=memory,
        minimum_similarity=0.68,
        maximum_similarity=0.78,
    )
    after = {
        name: digest(tmp_path / name)
        for name in ("memory.db", "memory.db-wal", "memory.db-shm", "session.json")
    }
    assert before == after


def test_missing_database_is_reported_not_created(tmp_path):
    session = _session_file(
        tmp_path / "session.json",
        [_cluster([_unit([1.0, 0.0])], _unit([1.0, 0.0]))],
    )
    missing = tmp_path / "absent.db"
    with pytest.raises(tool.RepresentativeComparisonError):
        tool.build_report(
            session_path=session,
            database_path=missing,
            minimum_similarity=0.68,
            maximum_similarity=0.78,
        )
    assert not missing.exists()


def test_unreadable_session_is_reported(tmp_path):
    memory = tmp_path / "memory.db"
    _write_memory(memory, [_unit([1.0, 0.0])])
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(tool.RepresentativeComparisonError):
        tool.build_report(session_path=broken, database_path=memory)


# --- CLI ------------------------------------------------------------------------

def test_cli_writes_report_outside_the_repository(tmp_path):
    generator = np.random.default_rng(3)
    enrolled = [_unit(generator.normal(size=512)) for _ in range(6)]
    memory = tmp_path / "memory.db"
    _write_memory(memory, enrolled)
    members = [_unit(enrolled[0] + generator.normal(scale=0.02, size=512))
               for _ in range(4)]
    session = _session_file(
        tmp_path / "session.json",
        [_cluster(members, members[0])],
    )
    destination = tmp_path / "report.json"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forensics.person_creation.tools.compare_cluster_representatives",
            "--session", str(session),
            "--memory-db", str(memory),
            "--output", str(destination),
            "--minimum-similarity", "0.68",
            "--maximum-similarity", "0.78",
        ],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )

    assert completed.returncode == 0, completed.stderr
    assert destination.is_file()
    report = json.loads(destination.read_text(encoding="utf-8"))
    assert report["status"] == "compared"
    assert report["embedding_dimension"] == 512
    assert "person_" not in json.dumps(report)
    # The tool must not leave a report inside the working tree.
    assert not (REPOSITORY_ROOT / "representative_comparison.json").exists()


def test_cli_reports_failure_without_traceback(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "forensics.person_creation.tools.compare_cluster_representatives",
            "--session", str(tmp_path / "missing.json"),
            "--memory-db", str(tmp_path / "missing.db"),
        ],
        cwd=str(REPOSITORY_ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
