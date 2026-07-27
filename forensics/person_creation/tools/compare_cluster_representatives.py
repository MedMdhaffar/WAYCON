"""Offline comparison of cluster face-representative strategies.

This module is deliberately **not** imported by any live runtime path.  The
comparison needs one Global Memory read per enrolled person per cluster, so
running it inside ``LiveRollingAnalysisSession._analyze`` cost
``O(active identities x enrolled persons)`` queries on every rolling pass and
published the enrolled-person roster through ``/api/person/status``.

It answers one question: would switching the cluster representative between the
current ``normalized_medoid`` (``cluster_identities._robust_representative``)
and the previous ``normalized_mean`` change which enrolled identity a live
cluster matches, or move a top similarity across an identity-policy threshold?

The report is aggregate and anonymised: no enrolled-person identifiers, no raw
embeddings, and no per-person similarity rows ever leave this tool.  Global
Memory is opened read-only and neither the session evidence nor the database,
WAL, or SHM is modified.

Usage::

    python -m forensics.person_creation.tools.compare_cluster_representatives \\
        --session forensics/person_db/<session>/canonical_live_session.json \\
        --memory-db /tmp/global_memory_copy.db \\
        --output /tmp/representative_comparison.json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CURRENT_STRATEGY = "normalized_medoid"
PREVIOUS_STRATEGY = "normalized_mean"


class RepresentativeComparisonError(RuntimeError):
    """The comparison inputs are unusable."""


def _unit(vector: Any, *, expected_size: int | None = None) -> np.ndarray | None:
    """Return an L2-normalised copy, or None when the vector is unusable."""
    candidate = np.asarray(vector, dtype=np.float64)
    if candidate.ndim != 1 or candidate.size == 0:
        return None
    if expected_size is not None and candidate.size != expected_size:
        return None
    if not np.isfinite(candidate).all():
        return None
    norm = float(np.linalg.norm(candidate))
    if norm <= 0:
        return None
    return candidate / norm


def cluster_representatives(
    cluster: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Return (medoid, normalized_mean, face_count) for one cluster.

    The medoid is read from the persisted ``representative_embedding`` so the
    comparison reflects what the pipeline actually produced, not a
    reconstruction.  The mean is recomputed from the same member vectors.
    """
    members = []
    for record in cluster.get("face_records") or []:
        unit = _unit(record.get("embedding"))
        if unit is not None:
            members.append(unit)
    if not members:
        return None
    medoid = _unit(cluster.get("representative_embedding"), expected_size=members[0].size)
    if medoid is None:
        return None
    normalized_mean = _unit(np.mean(np.vstack(members), axis=0))
    if normalized_mean is None:
        return None
    return medoid, normalized_mean, len(members)


def load_session_clusters(session_path: str | Path) -> list[dict]:
    """Read multi-face clusters from a canonical live-session report."""
    path = Path(session_path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RepresentativeComparisonError(
            f"Session evidence could not be read: {type(exc).__name__}"
        ) from exc
    if isinstance(payload, dict):
        clusters = payload.get("identity_clusters")
    elif isinstance(payload, list):
        clusters = payload
    else:
        clusters = None
    if not isinstance(clusters, list):
        raise RepresentativeComparisonError(
            "Session evidence contains no identity_clusters array."
        )
    return [cluster for cluster in clusters if isinstance(cluster, dict)]


def load_enrolled_gallery(database_path: str | Path) -> list[np.ndarray]:
    """Read enrolled face embeddings read-only, discarding identifiers.

    Person identifiers are dropped here on purpose: the aggregate report must
    not be able to name anyone even by accident.
    """
    path = Path(database_path)
    if not path.is_file():
        raise RepresentativeComparisonError(
            "Global Memory database path is not a regular file."
        )
    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        raise RepresentativeComparisonError(
            f"Global Memory could not be opened read-only: {type(exc).__name__}"
        ) from exc
    try:
        rows = list(
            connection.execute("select embedding from persons where is_active = 1")
        )
    except sqlite3.Error as exc:
        raise RepresentativeComparisonError(
            f"Global Memory could not be queried: {type(exc).__name__}"
        ) from exc
    finally:
        connection.close()

    gallery: list[np.ndarray] = []
    for (blob,) in rows:
        if blob is None:
            continue
        unit = _unit(np.frombuffer(blob, dtype=np.float32).astype(np.float64))
        if unit is not None:
            gallery.append(unit)
    return gallery


def _band(similarity: float, minimum: float, maximum: float) -> str:
    if similarity < minimum:
        return "below_minimum"
    if similarity >= maximum:
        return "at_or_above_maximum"
    return "between_thresholds"


def compare_representatives(
    clusters: Sequence[Mapping[str, Any]],
    gallery: Sequence[np.ndarray],
    *,
    minimum_similarity: float,
    maximum_similarity: float,
) -> dict:
    """Return anonymised aggregate drift metrics for the supplied clusters."""
    if not gallery:
        return {
            "status": "no_enrolled_identities",
            "compared_cluster_count": 0,
            "enrolled_identity_count": 0,
        }
    stack = np.vstack(gallery)

    drifts: list[float] = []
    face_counts: list[int] = []
    best_match_changed = 0
    minimum_crossings = 0
    maximum_crossings = 0
    decision_band_changes = 0
    skipped = 0

    for cluster in clusters:
        resolved = cluster_representatives(cluster)
        if resolved is None:
            skipped += 1
            continue
        medoid, normalized_mean, face_count = resolved
        if medoid.size != stack.shape[1]:
            skipped += 1
            continue
        face_counts.append(face_count)

        medoid_scores = stack @ medoid
        mean_scores = stack @ normalized_mean
        drifts.append(float(np.max(np.abs(medoid_scores - mean_scores))))

        medoid_best = int(np.argmax(medoid_scores))
        mean_best = int(np.argmax(mean_scores))
        if medoid_best != mean_best:
            best_match_changed += 1

        medoid_top = float(medoid_scores[medoid_best])
        mean_top = float(mean_scores[mean_best])
        low, high = min(medoid_top, mean_top), max(medoid_top, mean_top)
        if low < minimum_similarity <= high:
            minimum_crossings += 1
        if low < maximum_similarity <= high:
            maximum_crossings += 1
        if _band(medoid_top, minimum_similarity, maximum_similarity) != _band(
            mean_top, minimum_similarity, maximum_similarity
        ):
            decision_band_changes += 1

    if not drifts:
        return {
            "status": "no_comparable_clusters",
            "compared_cluster_count": 0,
            "enrolled_identity_count": len(gallery),
            "skipped_cluster_count": skipped,
        }

    return {
        "status": "compared",
        "current_strategy": CURRENT_STRATEGY,
        "previous_strategy": PREVIOUS_STRATEGY,
        "enrolled_identity_count": len(gallery),
        "embedding_dimension": int(stack.shape[1]),
        "compared_cluster_count": len(drifts),
        "skipped_cluster_count": skipped,
        "minimum_face_count": min(face_counts),
        "maximum_face_count": max(face_counts),
        "maximum_similarity_drift": round(max(drifts), 6),
        "mean_similarity_drift": round(sum(drifts) / len(drifts), 6),
        "best_match_identity_changed_count": best_match_changed,
        "minimum_similarity_crossing_count": minimum_crossings,
        "maximum_similarity_crossing_count": maximum_crossings,
        "decision_band_change_count": decision_band_changes,
        "minimum_similarity": minimum_similarity,
        "maximum_similarity": maximum_similarity,
    }


def _default_output() -> Path:
    return Path(tempfile.gettempdir()) / "representative_comparison.json"


def _policy_thresholds() -> tuple[float, float]:
    from forensics.global_memory.identity_policy import IdentityPolicyConfig

    policy = IdentityPolicyConfig.from_environment()
    return float(policy.minimum_similarity), float(policy.maximum_similarity)


def build_report(
    *,
    session_path: str | Path,
    database_path: str | Path,
    minimum_similarity: float | None = None,
    maximum_similarity: float | None = None,
) -> dict:
    """Load both inputs read-only and return the anonymised aggregate report."""
    if minimum_similarity is None or maximum_similarity is None:
        resolved_minimum, resolved_maximum = _policy_thresholds()
        minimum_similarity = (
            resolved_minimum if minimum_similarity is None else minimum_similarity
        )
        maximum_similarity = (
            resolved_maximum if maximum_similarity is None else maximum_similarity
        )
    clusters = load_session_clusters(session_path)
    gallery = load_enrolled_gallery(database_path)
    report = compare_representatives(
        clusters,
        gallery,
        minimum_similarity=float(minimum_similarity),
        maximum_similarity=float(maximum_similarity),
    )
    report["session_cluster_count"] = len(clusters)
    return report


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compare_cluster_representatives",
        description=(
            "Offline medoid-versus-mean representative comparison. Read-only; "
            "emits aggregate metrics only."
        ),
    )
    parser.add_argument(
        "--session",
        required=True,
        help="Path to a canonical_live_session.json (or an identity_clusters array).",
    )
    parser.add_argument(
        "--memory-db",
        required=True,
        help="Path to a Global Memory database. Use a copy; opened read-only.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Report destination. Defaults to the system temp directory.",
    )
    parser.add_argument("--minimum-similarity", type=float, default=None)
    parser.add_argument("--maximum-similarity", type=float, default=None)
    arguments = parser.parse_args(list(argv) if argv is not None else None)

    try:
        report = build_report(
            session_path=arguments.session,
            database_path=arguments.memory_db,
            minimum_similarity=arguments.minimum_similarity,
            maximum_similarity=arguments.maximum_similarity,
        )
    except RepresentativeComparisonError as exc:
        print(f"[compare_cluster_representatives] {exc}", file=sys.stderr)
        return 2

    destination = Path(arguments.output) if arguments.output else _default_output()
    if not destination.is_absolute():
        destination = Path.cwd() / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"[compare_cluster_representatives] report written to {destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
