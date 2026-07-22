from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path, PureWindowsPath
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import numpy as np

from forensics.identity_evidence import identity_evidence_key
from forensics.media_paths import (
    MediaPathError,
    get_media_root,
    normalize_media_path,
    resolve_media_path,
)

from . import config
from .identity_policy import (
    IdentityCandidate,
    IdentityDecision,
    IdentityDecisionType,
    IdentityEvidenceAppendResult,
    IdentityPolicyConfig,
    IdentityPolicyInputError,
    IdentityRegistrationResult,
    evaluate_identity_decision,
)
from .merge import (
    AlreadyMergedConflictError,
    IdentityLineage,
    IdentityLineageMember,
    InactiveSourceError,
    InactiveTargetError,
    InvalidMergeEmbeddingError,
    InvalidMergeMetadataError,
    InvalidMergeRequestError,
    MergeAuditIntegrityError,
    PersonMergeError,
    PersonMergeResult,
    PersonNotFoundError,
    RedirectChainError,
    SelfMergeError,
)
from .review import (
    IdentityReviewDecision,
    IdentityReviewDecisionResult,
    IdentityReviewDetail,
    IdentityReviewSummary,
    InvalidReviewDecisionError,
    InvalidReviewRequestError,
    ReviewSuggestionConflictError,
    ReviewSuggestionIntegrityError,
    ReviewSuggestionNotFoundError,
    ReviewSuggestionStaleError,
)


SQLITE_MAX_INTEGER = 2**63 - 1
# A drive letter or any URI scheme marks a non-canonical absolute reference.
_ABSOLUTE_MEDIA_PREFIX = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
MERGE_WEIGHTED_NORM_RELATIVE_MINIMUM = 1e-8
IDENTITY_REVIEW_DEFAULT_LIMIT = 50
IDENTITY_REVIEW_MAX_LIMIT = 100
IDENTITY_REVIEW_MAX_OFFSET = SQLITE_MAX_INTEGER
IDENTITY_REVIEW_GALLERY_LIMIT_PER_TYPE = 6
IDENTITY_REVIEW_APPEARANCE_LIMIT = 10
IDENTITY_REVIEW_RECOGNITION_LIMIT = 10


class ReadOnlyGlobalMemoryError(RuntimeError):
    """A mutating operation was attempted through a read-only memory handle."""


class GlobalMemory:
    def __init__(
        self,
        db_path: str | None = None,
        *,
        read_only: bool = False,
        media_root: str | Path | None = None,
    ):
        self.db_path = Path(db_path or os.getenv("FORENSICS_MEMORY_DB", config.DB_PATH))
        self.media_root = get_media_root(media_root)
        self.read_only = bool(read_only)
        self._lock = threading.RLock()
        self._conn = self._open_connection()
        try:
            self._conn.row_factory = sqlite3.Row
            self._configure_connection()
            if not self.read_only:
                self._initialize_schema()
        except BaseException:
            self._conn.close()
            raise

    def _open_connection(self) -> sqlite3.Connection:
        if self.read_only:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            return sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
                isolation_level=None,
            )
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level=None,
        )

    def _configure_connection(self) -> None:
        if self.read_only:
            self._conn.execute("PRAGMA query_only=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            return
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA busy_timeout=5000")
        journal_mode = self._conn.execute("PRAGMA journal_mode=WAL").fetchone()
        if journal_mode is None or str(journal_mode[0]).lower() != "wal":
            actual = "unknown" if journal_mode is None else str(journal_mode[0])
            raise RuntimeError(
                f"Global Memory requires WAL journal mode; SQLite returned {actual!r}."
            )

    def _initialize_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema.sql")
        try:
            self._conn.executescript(
                "BEGIN IMMEDIATE;\n" + schema_path.read_text(encoding="utf-8")
            )
            self._ensure_schema_columns()
            self._conn.execute("COMMIT")
        except BaseException:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
            raise

    def register(self, profile: dict) -> str:
        self._require_writable("register")
        new_vec = self._normalize_embedding(profile["face_embedding"])
        new_count = len(profile.get("face_crops") or []) or 1
        appearance = profile.get("appearance") or {}
        appearance_date = str(appearance.get("date") or date.today().isoformat())
        best_face_crop = self._best_face_crop(profile)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                existing = self._find_existing_person(new_vec)
                if existing is not None:
                    person_id = existing["person_id"]
                    count_before = int(existing["embedding_count"])
                    count_after = self._update_embedding(
                        person_id=person_id,
                        embedding=new_vec,
                        new_count=new_count,
                        updated_at=appearance_date,
                        profile=profile,
                    )
                    self._upsert_appearance(person_id, appearance_date, profile)
                    self.update_gallery(person_id, profile)
                    self._log_event(
                        person_id=person_id,
                        event_type="recognized",
                        similarity=existing["similarity"],
                        embedding_count_before=count_before,
                        embedding_count_after=count_after,
                        video_sources=profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                    )
                else:
                    person_id, name = self._next_person_id()
                    self._insert_person(
                        person_id=person_id,
                        name=name,
                        embedding=new_vec,
                        embedding_count=new_count,
                        enrolled_at=appearance_date,
                        cameras=profile.get("cameras") or [],
                        profile=profile,
                    )
                    self._upsert_appearance(person_id, appearance_date, profile)
                    self.update_gallery(person_id, profile)
                    self._log_event(
                        person_id=person_id,
                        event_type="new_enrollment",
                        similarity=None,
                        embedding_count_before=None,
                        embedding_count_after=new_count,
                        video_sources=profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                    )
                self._conn.execute("COMMIT")
                return person_id
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def query_by_face(self, embedding, top_k: int = 5, threshold: float | None = None) -> list[dict]:
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else float(threshold)
        query_vec = self._normalize_embedding(embedding)

        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, name, embedding FROM persons WHERE is_active=1"
            ).fetchall()
            if not rows:
                return []

            matrix = np.vstack([
                np.frombuffer(row["embedding"], dtype=np.float32) for row in rows
            ])
            sims = matrix @ query_vec
            order = np.argsort(-sims)

            results: list[dict] = []
            for idx in order:
                similarity = float(sims[int(idx)])
                if similarity < threshold:
                    break
                row = rows[int(idx)]
                results.append({
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "similarity": similarity,
                    "appearance": self._latest_appearance(row["person_id"]),
                })
                if len(results) >= top_k:
                    break
            return results

    def query_by_date(self, date: str) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT p.person_id, p.name, a.*
                 FROM appearances a
                  JOIN persons p ON p.person_id = a.person_id
                 WHERE a.date = ? AND p.is_active = 1
                 ORDER BY p.name
                """,
                (date,),
            ).fetchall()
            return [
                {
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "appearance": self._appearance_from_row(row, include_stale=False),
                }
                for row in rows
            ]

    def query_by_camera(self, camera_id: str) -> list[dict]:
        camera_id = str(camera_id)
        with self._lock:
            rows = self._conn.execute(
                "SELECT person_id, name, enrolled_at, cameras "
                "FROM persons WHERE is_active=1"
            ).fetchall()
            results = []
            for row in rows:
                cameras = self._json_list(row["cameras"])
                if camera_id in {str(item) for item in cameras}:
                    results.append({
                        "person_id": row["person_id"],
                        "name": row["name"],
                        "enrolled_at": row["enrolled_at"],
                        "cameras": cameras,
                    })
            return results

    def get_person(self, person_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM persons WHERE person_id=?",
                (person_id,),
            ).fetchone()
            if row is None:
                return None
            embedding = np.frombuffer(row["embedding"], dtype=np.float32).tolist()
            return {
                "person_id": row["person_id"],
                "name": row["name"],
                "embedding": embedding,
                "embedding_count": row["embedding_count"],
                "enrolled_at": row["enrolled_at"],
                "updated_at": row["updated_at"],
                "cameras": self._json_list(row["cameras"]),
                "profile_image": self._public_media_path(row["profile_image"]),
                "profile_image_source": row["profile_image_source"],
                "is_active": bool(row["is_active"]),
                "merged_into_person_id": row["merged_into_person_id"],
                "image_candidates": self._image_candidates(row["person_id"]),
                "latest_appearance": self._latest_appearance(row["person_id"]),
            }

    def list_all(self, include_inactive: bool = False) -> list[dict]:
        with self._lock:
            where = "" if include_inactive else "WHERE is_active=1"
            rows = self._conn.execute(
                f"""
                SELECT person_id, name, enrolled_at, updated_at, cameras,
                       profile_image, profile_image_source,
                       is_active, merged_into_person_id
                  FROM persons
                 {where}
                 ORDER BY person_id
                """
            ).fetchall()
            return [
                {
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "enrolled_at": row["enrolled_at"],
                    "updated_at": row["updated_at"],
                    "cameras": self._json_list(row["cameras"]),
                    "profile_image": self._public_media_path(row["profile_image"]),
                    "profile_image_source": row["profile_image_source"],
                    "is_active": bool(row["is_active"]),
                    "merged_into_person_id": row["merged_into_person_id"],
                    "image_candidates": self._image_candidates(row["person_id"]),
                    "latest_appearance": self._latest_appearance(row["person_id"]),
                }
                for row in rows
            ]

    def get_recognition_history(self, person_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._lock:
            if person_id:
                rows = self._conn.execute(
                    """
                    SELECT * FROM recognition_log
                     WHERE person_id = ?
                     ORDER BY id DESC
                     LIMIT ?
                    """,
                    (person_id, int(limit)),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM recognition_log
                     ORDER BY id DESC
                     LIMIT ?
                    """,
                    (int(limit),),
                ).fetchall()

            return [
                {
                    "id": row["id"],
                    "person_id": row["person_id"],
                    "event_type": row["event_type"],
                    "similarity": row["similarity"],
                    "embedding_count_before": row["embedding_count_before"],
                    "embedding_count_after": row["embedding_count_after"],
                    "video_sources": self._json_list(row["video_sources"]),
                    "best_face_crop": self._public_media_path(row["best_face_crop"]),
                    "ts": row["ts"],
                }
                for row in rows
            ]

    def rename_person(self, person_id: str, new_name: str) -> None:
        self._require_writable("rename_person")
        with self._lock:
            self._conn.execute(
                "UPDATE persons SET name=? WHERE person_id=?",
                (new_name, person_id),
            )

    def update_crop_paths(self, person_id: str, profile: dict) -> None:
        self._require_writable("update_crop_paths")
        appearance = profile.get("appearance") or {}
        appearance_date = str(appearance.get("date") or date.today().isoformat())
        profile_image = self._best_face_crop(profile)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                person = self._conn.execute(
                    "SELECT profile_image_source FROM persons WHERE person_id=?",
                    (person_id,),
                ).fetchone()
                image_source = person["profile_image_source"] if person else "auto"
                if profile_image:
                    if image_source != "manual":
                        self._conn.execute(
                            """
                            UPDATE persons
                               SET profile_image=?, profile_image_source='auto'
                             WHERE person_id=?
                            """,
                            (profile_image, person_id),
                        )
                    latest = self._conn.execute(
                        """
                        SELECT id FROM recognition_log
                         WHERE person_id=?
                         ORDER BY id DESC
                         LIMIT 1
                        """,
                        (person_id,),
                    ).fetchone()
                    if latest is not None:
                        self._conn.execute(
                            "UPDATE recognition_log SET best_face_crop=? WHERE id=?",
                            (profile_image, latest["id"]),
                        )
                self._upsert_appearance(person_id, appearance_date, profile)
                self._remove_missing_gallery_paths(person_id)
                self.update_gallery(person_id, profile)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def register_with_identity_policy(
        self,
        profile: dict,
        *,
        observation_count: int,
        low_confidence: bool = False,
        configuration: IdentityPolicyConfig | None = None,
        prepare_profile_for_person: Callable[[str, dict], dict] | None = None,
        evidence_keys=None,
    ) -> IdentityRegistrationResult:
        """Apply the Phase 3E policy and persist one atomic identity outcome."""
        self._require_writable("register_with_identity_policy")
        if configuration is None:
            policy_config = IdentityPolicyConfig.from_environment()
        elif isinstance(configuration, IdentityPolicyConfig):
            policy_config = configuration
        else:
            raise TypeError("configuration must be an IdentityPolicyConfig")

        new_vec = self._validated_identity_embedding(profile["face_embedding"])
        new_count = len(profile.get("face_crops") or []) or 1

        def prepared_profile(person_id: str) -> tuple[dict, str, str | None]:
            prepared = (
                profile
                if prepare_profile_for_person is None
                else prepare_profile_for_person(person_id, profile)
            )
            if not isinstance(prepared, dict):
                raise IdentityPolicyInputError(
                    "prepared identity profile must be a dictionary"
                )
            appearance = prepared.get("appearance") or {}
            appearance_date = str(
                appearance.get("date") or date.today().isoformat()
            )
            return prepared, appearance_date, self._best_face_crop(prepared)

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                candidates = self._rank_active_identity_candidates(new_vec)
                top_candidate = candidates[0] if candidates else None
                second_candidate = candidates[1] if len(candidates) > 1 else None
                identity_decision = evaluate_identity_decision(
                    top_candidate=top_candidate,
                    second_candidate=second_candidate,
                    observation_count=observation_count,
                    low_confidence=low_confidence,
                    configuration=policy_config,
                )

                suggestion_id: int | None = None
                if identity_decision.decision is IdentityDecisionType.ATTACH_EXISTING:
                    person_id = identity_decision.top_candidate_person_id
                    if person_id is None:
                        raise IdentityPolicyInputError(
                            "attach_existing requires a top candidate"
                        )
                    stored_profile, appearance_date, best_face_crop = (
                        prepared_profile(person_id)
                    )
                    count_before = self._get_embedding_count(person_id)
                    count_after = self._update_embedding(
                        person_id=person_id,
                        embedding=new_vec,
                        new_count=new_count,
                        updated_at=appearance_date,
                        profile=stored_profile,
                    )
                    self._upsert_appearance(
                        person_id,
                        appearance_date,
                        stored_profile,
                    )
                    self.update_gallery(person_id, stored_profile)
                    self._log_event(
                        person_id=person_id,
                        event_type="recognized",
                        similarity=identity_decision.top_similarity,
                        embedding_count_before=count_before,
                        embedding_count_after=count_after,
                        video_sources=stored_profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                        strict=True,
                    )
                    if evidence_keys is not None:
                        evidence = self._identity_evidence_items(
                            person_id,
                            stored_profile,
                            evidence_keys=evidence_keys,
                        )
                        self._insert_initial_identity_evidence(
                            person_id,
                            evidence,
                            created_at=appearance_date,
                        )
                else:
                    person_id, name = self._next_person_id()
                    stored_profile, appearance_date, best_face_crop = (
                        prepared_profile(person_id)
                    )
                    self._insert_person(
                        person_id=person_id,
                        name=name,
                        embedding=new_vec,
                        embedding_count=new_count,
                        enrolled_at=appearance_date,
                        cameras=stored_profile.get("cameras") or [],
                        profile=stored_profile,
                    )
                    self._upsert_appearance(
                        person_id,
                        appearance_date,
                        stored_profile,
                    )
                    self.update_gallery(person_id, stored_profile)
                    self._log_event(
                        person_id=person_id,
                        event_type="new_enrollment",
                        similarity=None,
                        embedding_count_before=None,
                        embedding_count_after=new_count,
                        video_sources=stored_profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                        strict=True,
                    )
                    if evidence_keys is not None:
                        evidence = self._identity_evidence_items(
                            person_id,
                            stored_profile,
                            evidence_keys=evidence_keys,
                        )
                        self._insert_initial_identity_evidence(
                            person_id,
                            evidence,
                            created_at=appearance_date,
                        )
                    if (
                        identity_decision.decision
                        is IdentityDecisionType.REVIEW_REQUIRED
                    ):
                        suggestion_id = self._insert_identity_suggestion(
                            source_person_id=person_id,
                            decision=identity_decision,
                        )

                result = IdentityRegistrationResult.from_decision(
                    person_id=person_id,
                    suggestion_id=suggestion_id,
                    decision=identity_decision,
                    configuration=policy_config,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

        self._log_identity_decision(result)
        return result

    def rank_identity_candidates(self, embedding) -> tuple[IdentityCandidate, ...]:
        """Read-only Phase 3E candidate ranking used for provisional decisions."""
        vector = self._validated_identity_embedding(embedding)
        with self._lock:
            return self._rank_active_identity_candidates(vector)

    def identity_evidence_keys(self, person_id: str) -> frozenset[str]:
        """Return durable evidence keys owned by the active canonical person."""
        person_id = self._validated_merge_text(person_id, "person_id")
        with self._lock:
            canonical = self._resolve_canonical_person_id_locked(person_id)
            rows = self._conn.execute(
                "SELECT evidence_key FROM identity_evidence WHERE person_id=?",
                (canonical,),
            ).fetchall()
            return frozenset(str(row["evidence_key"]) for row in rows)

    def append_identity_evidence(
        self,
        person_id: str,
        embedding=None,
        observation_count: int = 0,
        face_crops=None,
        body_crops=None,
        appearance=None,
        evidence_keys=None,
    ) -> IdentityEvidenceAppendResult:
        """Atomically append evidence not already present in the durable ledger."""
        self._require_writable("append_identity_evidence")
        person_id = self._validated_merge_text(person_id, "person_id")
        appearance = dict(appearance or {})
        appearance_date = str(appearance.get("date") or date.today().isoformat())

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                canonical = self._resolve_canonical_person_id_locked(person_id)
                count_before = self._get_embedding_count(canonical)
                candidate_profile = {
                    "face_crops": list(face_crops or []),
                    "body_crops": list(body_crops or []),
                }
                evidence = self._identity_evidence_items(
                    canonical,
                    candidate_profile,
                    evidence_keys=evidence_keys,
                )
                existing = {
                    str(row["evidence_key"])
                    for row in self._conn.execute(
                        "SELECT evidence_key FROM identity_evidence WHERE person_id=?",
                        (canonical,),
                    ).fetchall()
                }
                unseen = [item for item in evidence if item[0] not in existing]
                if not unseen:
                    self._conn.execute("COMMIT")
                    return IdentityEvidenceAppendResult(
                        person_id=person_id,
                        canonical_person_id=canonical,
                        appended=False,
                        idempotent_replay=True,
                        appended_evidence_keys=(),
                        embedding_count_before=count_before,
                        embedding_count_after=count_before,
                        gallery_rows_added=0,
                    )

                unseen_faces = [item for item in unseen if item[1] == "face"]
                unseen_bodies = [item for item in unseen if item[1] == "body"]
                requested = int(observation_count or 0)
                if requested < 0:
                    raise IdentityPolicyInputError(
                        "observation_count must not be negative"
                    )
                if unseen_faces and requested not in (0, len(unseen_faces)):
                    raise IdentityPolicyInputError(
                        "observation_count must equal the unseen face evidence count"
                    )
                new_count = len(unseen_faces)
                if count_before > SQLITE_MAX_INTEGER - new_count:
                    raise IdentityPolicyInputError(
                        "appended observation count would overflow"
                    )

                for key, crop_type, canonical_path in unseen:
                    self._conn.execute(
                        """
                        INSERT INTO identity_evidence (
                            person_id, evidence_key, crop_type, canonical_path,
                            embedding_applied, observation_weight, created_at
                        ) VALUES (?, ?, ?, ?, 0, ?, ?)
                        """,
                        (
                            canonical,
                            key,
                            crop_type,
                            canonical_path,
                            1 if crop_type == "face" else 0,
                            appearance_date,
                        ),
                    )

                profile = {
                    "face_crops": [item[2] for item in unseen_faces],
                    "body_crops": [item[2] for item in unseen_bodies],
                    "best_body_crops": [item[2] for item in unseen_bodies],
                    "appearance": appearance,
                    "face_crop_sharpness": (
                        appearance.get("face_crop_sharpness") or {}
                    ),
                    "body_crop_sharpness": (
                        appearance.get("body_crop_sharpness") or {}
                    ),
                    "video_sources": appearance.get("video_sources") or [],
                }
                gallery_before = self._gallery_row_count(canonical)
                count_after = count_before
                if unseen_faces:
                    new_vec = self._validated_identity_embedding(embedding)
                    count_after = self._update_embedding(
                        person_id=canonical,
                        embedding=new_vec,
                        new_count=new_count,
                        updated_at=appearance_date,
                        profile=profile,
                    )
                    face_keys = [item[0] for item in unseen_faces]
                    placeholders = ",".join("?" for _ in face_keys)
                    self._conn.execute(
                        "UPDATE identity_evidence SET embedding_applied=1 "
                        f"WHERE person_id=? AND evidence_key IN ({placeholders})",
                        (canonical, *face_keys),
                    )
                self._upsert_appearance(canonical, appearance_date, profile)
                self.update_gallery(canonical, profile)
                gallery_added = max(
                    0,
                    self._gallery_row_count(canonical) - gallery_before,
                )
                self._log_event(
                    person_id=canonical,
                    event_type="evidence_appended",
                    similarity=None,
                    embedding_count_before=count_before,
                    embedding_count_after=count_after,
                    video_sources=profile["video_sources"],
                    best_face_crop=self._best_face_crop(profile),
                    strict=True,
                )
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

        return IdentityEvidenceAppendResult(
            person_id=person_id,
            canonical_person_id=canonical,
            appended=True,
            idempotent_replay=False,
            appended_evidence_keys=tuple(item[0] for item in unseen),
            embedding_count_before=count_before,
            embedding_count_after=count_after,
            gallery_rows_added=gallery_added,
        )

    def _gallery_row_count(self, person_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM person_gallery WHERE person_id=?",
            (person_id,),
        ).fetchone()
        return int(row[0]) if row else 0

    def _validated_canonical_media(
        self,
        values,
        label: str,
        canonical_person_id: str,
        crop_type: str,
    ) -> tuple[str, ...]:
        """Accept existing ``person_NNN/<type>_crops`` references only."""
        resolved: list[str] = []
        for raw in values or []:
            text = str(raw or "").strip().replace("\\", "/")
            if not text:
                continue
            if text.startswith("/") or _ABSOLUTE_MEDIA_PREFIX.match(text):
                raise MediaPathError(
                    f"{label} must contain canonical relative media paths"
                )
            stored = normalize_media_path(
                text,
                media_root=self.media_root,
                allow_legacy_absolute=False,
                require_exists=True,
            )
            parts = stored.split("/")
            if (
                len(parts) < 3
                or parts[0] != canonical_person_id
                or parts[1] != f"{crop_type}_crops"
                or any(
                    part == "_staging"
                    or part == "session"
                    or re.fullmatch(r"cluster_[0-9]+", part)
                    for part in parts
                )
            ):
                raise MediaPathError(
                    f"{label} must contain canonical {canonical_person_id}/"
                    f"{crop_type}_crops paths"
                )
            if stored not in resolved:
                resolved.append(stored)
        return tuple(resolved)

    def _identity_evidence_items(
        self,
        canonical_person_id: str,
        profile: dict,
        *,
        evidence_keys=None,
    ) -> tuple[tuple[str, str, str], ...]:
        faces = self._validated_canonical_media(
            profile.get("face_crops"),
            "face_crops",
            canonical_person_id,
            "face",
        )
        bodies = self._validated_canonical_media(
            profile.get("body_crops"),
            "body_crops",
            canonical_person_id,
            "body",
        )
        items: list[tuple[str, str, str]] = []
        seen: set[str] = set()
        for crop_type, paths in (("face", faces), ("body", bodies)):
            for path in paths:
                key = identity_evidence_key(
                    path,
                    crop_type,
                    media_root=self.media_root,
                    allow_legacy_absolute=False,
                )
                if key in seen:
                    continue
                seen.add(key)
                items.append((key, crop_type, path))

        supplied = {
            str(key).strip()
            for key in (evidence_keys or [])
            if str(key).strip()
        }
        if evidence_keys is not None and supplied != seen:
            raise IdentityPolicyInputError(
                "evidence_keys do not match the supplied evidence contents"
            )
        return tuple(items)

    def _insert_initial_identity_evidence(
        self,
        person_id: str,
        evidence: tuple[tuple[str, str, str], ...],
        *,
        created_at: str,
    ) -> None:
        for key, crop_type, canonical_path in evidence:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO identity_evidence (
                    person_id, evidence_key, crop_type, canonical_path,
                    embedding_applied, observation_weight, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    key,
                    crop_type,
                    canonical_path,
                    1 if crop_type == "face" else 0,
                    1 if crop_type == "face" else 0,
                    created_at,
                ),
            )

    def resolve_canonical_person_id(self, person_id: str) -> str:
        """Resolve one active identity or one direct inactive redirect."""
        person_id = self._validated_merge_text(person_id, "person_id")
        with self._lock:
            return self._resolve_canonical_person_id_locked(person_id)

    def get_identity_lineage(self, person_id: str) -> IdentityLineage:
        """Return an active canonical person and its direct merged sources."""
        person_id = self._validated_merge_text(person_id, "person_id")
        with self._lock:
            return self._get_identity_lineage_locked(person_id)

    def get_lineage_appearances(self, person_id: str) -> list[dict[str, Any]]:
        """Return every appearance in a direct lineage without collapsing dates."""
        person_id = self._validated_merge_text(person_id, "person_id")
        with self._lock:
            lineage = self._get_identity_lineage_locked(person_id)
            placeholders = ",".join("?" for _ in lineage.members)
            rows = self._conn.execute(
                f"SELECT * FROM appearances WHERE person_id IN ({placeholders})",
                lineage.member_person_ids,
            ).fetchall()
            return self._lineage_evidence_rows(lineage, rows)

    def get_lineage_gallery(self, person_id: str) -> list[dict[str, Any]]:
        """Return every gallery row in a direct lineage with original ownership."""
        person_id = self._validated_merge_text(person_id, "person_id")
        with self._lock:
            lineage = self._get_identity_lineage_locked(person_id)
            placeholders = ",".join("?" for _ in lineage.members)
            rows = self._conn.execute(
                f"SELECT * FROM person_gallery WHERE person_id IN ({placeholders})",
                lineage.member_person_ids,
            ).fetchall()
            return self._lineage_evidence_rows(lineage, rows)

    def get_lineage_recognition_history(
        self,
        person_id: str,
        *,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return recognition history across a direct lineage with provenance."""
        person_id = self._validated_merge_text(person_id, "person_id")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 1
        ):
            raise InvalidMergeRequestError("limit must be a positive integer or None")
        with self._lock:
            lineage = self._get_identity_lineage_locked(person_id)
            placeholders = ",".join("?" for _ in lineage.members)
            rows = self._conn.execute(
                f"SELECT * FROM recognition_log WHERE person_id IN ({placeholders})",
                lineage.member_person_ids,
            ).fetchall()
            evidence = self._lineage_evidence_rows(lineage, rows)
            return evidence if limit is None else evidence[:limit]

    def list_pending_identity_reviews(
        self,
        *,
        limit: int = IDENTITY_REVIEW_DEFAULT_LIMIT,
        offset: int = 0,
    ) -> list[IdentityReviewSummary]:
        """Return the oldest pending supervisor reviews without biometric data."""
        page_limit, page_offset = self._validated_review_pagination(limit, offset)
        with self._lock:
            try:
                rows = self._conn.execute(
                    """
                    SELECT s.*,
                           source.name AS source_name,
                           source.profile_image AS source_profile_image,
                           candidate.name AS candidate_name,
                           candidate.profile_image AS candidate_profile_image
                      FROM identity_match_suggestions AS s
                      JOIN persons AS source
                        ON source.person_id = s.source_person_id
                      JOIN persons AS candidate
                        ON candidate.person_id = s.candidate_person_id
                     WHERE s.status = 'pending'
                     ORDER BY s.created_at ASC, s.id ASC
                     LIMIT ? OFFSET ?
                    """,
                    (page_limit, page_offset),
                ).fetchall()
            except (OverflowError, sqlite3.DataError) as exc:
                raise InvalidReviewRequestError(
                    "identity review pagination is outside SQLite's supported range"
                ) from exc
            return [self._identity_review_summary_from_row(row) for row in rows]

    def count_pending_identity_reviews(self) -> int:
        """Return the current pending supervisor-review count."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM identity_match_suggestions "
                "WHERE status='pending'"
            ).fetchone()
            return int(row[0])

    def get_identity_review(self, suggestion_id: str | int) -> IdentityReviewDetail:
        """Return a bounded, embedding-free source/candidate comparison."""
        review_id = self._validated_review_suggestion_id(suggestion_id)
        with self._lock:
            row = self._load_identity_review_summary_row(review_id)
            source = self._load_review_person(str(row["source_person_id"]), "source")
            candidate = self._load_review_person(
                str(row["candidate_person_id"]),
                "candidate",
            )
            self._validate_review_relationship(row, source, candidate)
            summary = self._identity_review_summary_from_row(row)

            source_ids = (str(source["person_id"]),)
            candidate_lineage = self._get_identity_lineage_locked(
                str(candidate["person_id"])
            )
            candidate_ids = candidate_lineage.member_person_ids
            source_lineage = self._review_lineage_summary(source)

            return IdentityReviewDetail(
                suggestion=summary,
                source_profile=self._review_profile_summary(source),
                candidate_profile=self._review_profile_summary(candidate),
                source_lineage=source_lineage,
                candidate_lineage=tuple(
                    {
                        "person_id": member.person_id,
                        "is_active": member.is_active,
                        "merged_into_person_id": member.merged_into_person_id,
                        "canonical_person_id": candidate_lineage.canonical_person_id,
                    }
                    for member in candidate_lineage.members
                ),
                source_gallery=self._review_gallery(source_ids),
                candidate_gallery=self._review_gallery(candidate_ids),
                source_appearances=self._review_appearances(source_ids),
                candidate_appearances=self._review_appearances(candidate_ids),
                source_recognition_events=self._review_recognition_events(source_ids),
                candidate_recognition_events=self._review_recognition_events(
                    candidate_ids
                ),
            )

    def resolve_identity_review(
        self,
        suggestion_id: str | int,
        decision: IdentityReviewDecision | str,
        *,
        reason: str | None = None,
        decision_source: str | None = None,
    ) -> IdentityReviewDecisionResult:
        """Atomically accept or reject one persisted identity suggestion."""
        self._require_writable("resolve_identity_review")
        review_id = self._validated_review_suggestion_id(suggestion_id)
        review_decision = self._validated_review_decision(decision)
        review_reason = self._validated_optional_review_text(reason, "reason")
        reviewer = self._validated_optional_review_text(
            decision_source,
            "decision_source",
        )
        if review_decision is IdentityReviewDecision.REJECT and review_reason is not None:
            raise InvalidReviewRequestError(
                "reason is unsupported for rejection because the schema has no review-note field"
            )

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                suggestion = self._load_identity_review_row(review_id)
                source_id = str(suggestion["source_person_id"])
                candidate_id = str(suggestion["candidate_person_id"])
                source = self._load_review_person(source_id, "source")
                candidate = self._load_review_person(candidate_id, "candidate")
                if source_id == candidate_id:
                    raise ReviewSuggestionIntegrityError(
                        "review suggestion source and candidate must differ"
                    )

                status = str(suggestion["status"])
                if status == "stale":
                    raise ReviewSuggestionStaleError(
                        f"identity review suggestion {review_id} is stale"
                    )
                if status == "accepted":
                    if review_decision is IdentityReviewDecision.REJECT:
                        raise ReviewSuggestionConflictError(
                            f"identity review suggestion {review_id} was already accepted"
                        )
                    result = self._replay_accepted_identity_review(
                        review_id,
                        source,
                        candidate,
                    )
                    self._conn.execute("COMMIT")
                    return result
                if status == "rejected":
                    if review_decision is IdentityReviewDecision.ACCEPT:
                        raise ReviewSuggestionConflictError(
                            f"identity review suggestion {review_id} was already rejected"
                        )
                    self._validate_rejected_review_state(
                        review_id,
                        source,
                        candidate,
                    )
                    result = self._rejected_identity_review_result(
                        review_id,
                        source_id,
                        candidate_id,
                        idempotent_replay=True,
                    )
                    self._conn.execute("COMMIT")
                    return result
                if status != "pending":
                    raise ReviewSuggestionIntegrityError(
                        f"identity review suggestion {review_id} has invalid status {status!r}"
                    )

                self._validate_pending_review_people(source, candidate)
                reviewed_at = datetime.now().isoformat(timespec="seconds")
                if review_decision is IdentityReviewDecision.ACCEPT:
                    merge_result = self._merge_persons_in_transaction(
                        source_id,
                        candidate_id,
                        reason=(
                            review_reason
                            or f"supervisor accepted identity review {review_id}"
                        ),
                        decision_source=self._review_audit_source(review_id),
                        preserve_suggestion_id=review_id,
                    )
                    self._update_identity_review_status(
                        review_id,
                        status="accepted",
                        reviewed_at=reviewed_at,
                        reviewed_by=reviewer,
                    )
                    result = IdentityReviewDecisionResult(
                        suggestion_id=str(review_id),
                        decision=review_decision,
                        status="accepted",
                        source_person_id=source_id,
                        target_person_id=candidate_id,
                        audit_id=merge_result.audit_id,
                        idempotent_replay=False,
                        staled_suggestion_count=merge_result.staled_suggestion_count,
                        source_embedding_count=merge_result.source_embedding_count,
                        target_embedding_count_before=(
                            merge_result.target_embedding_count_before
                        ),
                        target_embedding_count_after=(
                            merge_result.target_embedding_count_after
                        ),
                    )
                else:
                    self._update_identity_review_status(
                        review_id,
                        status="rejected",
                        reviewed_at=reviewed_at,
                        reviewed_by=reviewer,
                    )
                    result = self._rejected_identity_review_result(
                        review_id,
                        source_id,
                        candidate_id,
                        idempotent_replay=False,
                    )

                self._assert_merge_foreign_keys()
                self._before_review_commit()
                self._conn.execute("COMMIT")
                return result
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

    def merge_persons(
        self,
        source_person_id: str,
        target_person_id: str,
        *,
        reason: str,
        decision_source: str | None = None,
    ) -> PersonMergeResult:
        """Logically merge an active source into an active canonical target."""
        self._require_writable("merge_persons")
        source_id = self._validated_merge_text(source_person_id, "source_person_id")
        target_id = self._validated_merge_text(target_person_id, "target_person_id")
        merge_reason = self._validated_merge_text(reason, "reason")
        if source_id == target_id:
            raise SelfMergeError("source_person_id and target_person_id must differ")
        if decision_source is None:
            audit_source = "phase_3f_logical_merge"
        else:
            audit_source = self._validated_merge_text(
                decision_source,
                "decision_source",
            )

        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                result = self._merge_persons_in_transaction(
                    source_id,
                    target_id,
                    reason=merge_reason,
                    decision_source=audit_source,
                )
                self._before_merge_commit()
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                raise

        return result

    def _merge_persons_in_transaction(
        self,
        source_id: str,
        target_id: str,
        *,
        reason: str,
        decision_source: str,
        preserve_suggestion_id: int | None = None,
    ) -> PersonMergeResult:
        """Execute Phase 3F merge logic inside the caller-owned transaction."""
        if not self._conn.in_transaction:
            raise PersonMergeError("person merge primitive requires an active transaction")

        source = self._load_merge_person(source_id, "source")
        target = self._load_merge_person(target_id, "target")
        self._validate_merge_target(target)

        if not bool(source["is_active"]):
            redirect = source["merged_into_person_id"]
            if redirect is None:
                raise InactiveSourceError(
                    f"source person {source_id!r} is inactive without a redirect"
                )
            if redirect != target_id:
                raise AlreadyMergedConflictError(
                    f"source person {source_id!r} already redirects to {redirect!r}"
                )
            return self._replay_person_merge(source, target)

        if source["merged_into_person_id"] is not None:
            raise RedirectChainError(
                f"active source person {source_id!r} has an invalid redirect"
            )
        self._reject_redirect_children(source_id)
        self._require_no_source_merge_audit(source_id)

        source_embedding = self._validated_stored_merge_embedding(source, "source")
        target_embedding = self._validated_stored_merge_embedding(target, "target")
        if source_embedding.shape != target_embedding.shape:
            raise InvalidMergeEmbeddingError(
                "source and target embedding dimensions differ"
            )
        source_count = self._validated_stored_merge_count(source, "source")
        target_count_before = self._validated_stored_merge_count(target, "target")
        target_count_after = self._validated_merge_count_sum(
            target_count_before,
            source_count,
        )
        target_cameras = self._validated_merge_cameras(target["cameras"], "target")
        source_cameras = self._validated_merge_cameras(source["cameras"], "source")
        combined = self._combined_merge_embedding(
            target_embedding=target_embedding,
            target_count=target_count_before,
            source_embedding=source_embedding,
            source_count=source_count,
            weight_scale=target_count_after,
        )

        created_at = datetime.now().isoformat(timespec="seconds")
        audit_id = self._insert_person_merge_audit(
            source=source,
            target=target,
            reason=reason,
            decision_source=decision_source,
            source_embedding_count=source_count,
            created_at=created_at,
        )
        self._update_merge_target_embedding(
            target_id,
            combined,
            target_count_after,
            created_at,
        )
        self._update_merge_target_metadata(
            target_id,
            self._merge_lists(target_cameras, source_cameras),
        )
        if preserve_suggestion_id is None:
            staled_count = self._stale_merge_suggestions(source_id)
        else:
            staled_count = self._stale_merge_suggestions(
                source_id,
                preserve_suggestion_id=preserve_suggestion_id,
            )
        self._deactivate_merge_source(source_id)
        self._redirect_merge_source(source_id, target_id)
        self._assert_merge_foreign_keys()
        lineage_count = self._merge_lineage_member_count(target_id)
        return PersonMergeResult(
            source_person_id=source_id,
            target_person_id=target_id,
            audit_id=audit_id,
            idempotent_replay=False,
            source_embedding_count=source_count,
            target_embedding_count_before=target_count_before,
            target_embedding_count_after=target_count_after,
            staled_suggestion_count=staled_count,
            lineage_member_count_after=lineage_count,
        )

    def set_profile_image(self, person_id: str, image_path: str, source: str = "auto") -> None:
        self._require_writable("set_profile_image")
        source = source if source in {"auto", "manual"} else "auto"
        with self._lock:
            self._conn.execute(
                """
                UPDATE persons
                   SET profile_image=?, profile_image_source=?
                 WHERE person_id=?
                """,
                (self._stored_media_path(image_path, require_exists=True), source, person_id),
            )

    def get_best_face_crop(self, person_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT path, sharpness FROM person_gallery
                 WHERE person_id=? AND crop_type='face'
                 ORDER BY sharpness DESC
                 LIMIT 1
                """,
                (person_id,),
            ).fetchone()
            if row is not None:
                path = self._public_media_path(row["path"])
                if path:
                    return {"path": path, "sharpness": row["sharpness"]}

            row = self._conn.execute(
                """
                SELECT best_face_crop FROM recognition_log
                 WHERE person_id=? AND best_face_crop IS NOT NULL
                 ORDER BY ts DESC
                 LIMIT 1
                """,
                (person_id,),
            ).fetchone()
            if row is not None and row["best_face_crop"]:
                path = self._public_media_path(row["best_face_crop"])
                if path:
                    return {"path": path, "sharpness": 0.0}
            return None

    def update_gallery(self, person_id: str, profile: dict) -> None:
        self._require_writable("update_gallery")
        session_date = str((profile.get("appearance") or {}).get("date") or date.today().isoformat())
        video_sources = profile.get("video_sources") or []
        video_source = video_sources[0] if video_sources else None
        face_sharpness = profile.get("face_crop_sharpness") or {}
        body_sharpness = profile.get("body_crop_sharpness") or {}

        with self._lock:
            for path in profile.get("face_crops") or []:
                self._insert_gallery_crop(
                    person_id=person_id,
                    crop_type="face",
                    path=path,
                    sharpness=self._sharpness_for_path(face_sharpness, path),
                    session_date=session_date,
                    video_source=video_source,
                )

            for path in profile.get("best_body_crops") or []:
                self._insert_gallery_crop(
                    person_id=person_id,
                    crop_type="body",
                    path=path,
                    sharpness=self._sharpness_for_path(body_sharpness, path),
                    session_date=session_date,
                    video_source=video_source,
                )

            for crop_type in ("face", "body"):
                self._prune_gallery(person_id, crop_type, limit=10)

    def get_gallery(self, person_id: str, crop_type: str | None = None, limit: int = 10) -> list[dict]:
        limit = max(1, int(limit))
        with self._lock:
            if crop_type in {"face", "body"}:
                rows = self._conn.execute(
                    """
                    SELECT * FROM person_gallery
                     WHERE person_id=? AND crop_type=?
                     ORDER BY sharpness DESC
                     LIMIT ?
                    """,
                    (person_id, crop_type, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM person_gallery
                     WHERE person_id=?
                     ORDER BY crop_type, sharpness DESC
                     LIMIT ?
                    """,
                    (person_id, limit),
                ).fetchall()

            return [
                {
                    "id": row["id"],
                    "person_id": row["person_id"],
                    "crop_type": row["crop_type"],
                    "path": self._public_media_path(row["path"]),
                    "sharpness": row["sharpness"],
                    "session_date": row["session_date"],
                    "video_source": row["video_source"],
                    "width": row["width"],
                    "height": row["height"],
                }
                for row in rows
                if self._public_media_path(row["path"])
            ]

    def referenced_media_paths(self, person_id: str) -> set[str]:
        """Return every persistent media reference for conservative pruning."""
        with self._lock:
            values: list[str] = []
            person = self._conn.execute(
                "SELECT profile_image FROM persons WHERE person_id=?",
                (person_id,),
            ).fetchone()
            if person is not None and person["profile_image"]:
                values.append(person["profile_image"])
            values.extend(
                row["path"]
                for row in self._conn.execute(
                    "SELECT path FROM person_gallery WHERE person_id=?",
                    (person_id,),
                ).fetchall()
                if row["path"]
            )
            for row in self._conn.execute(
                "SELECT best_body_crops FROM appearances WHERE person_id=?",
                (person_id,),
            ).fetchall():
                values.extend(self._json_list(row["best_body_crops"]))
            values.extend(
                row["best_face_crop"]
                for row in self._conn.execute(
                    "SELECT best_face_crop FROM recognition_log WHERE person_id=?",
                    (person_id,),
                ).fetchall()
                if row["best_face_crop"]
            )

        references: set[str] = set()
        for value in values:
            try:
                references.add(normalize_media_path(
                    value,
                    media_root=self.media_root,
                    allow_legacy_absolute=True,
                    require_exists=False,
                ))
            except MediaPathError:
                continue
        return references

    def all_referenced_media_paths(self) -> set[str]:
        with self._lock:
            person_ids = [
                row["person_id"]
                for row in self._conn.execute("SELECT person_id FROM persons").fetchall()
            ]
        references: set[str] = set()
        for person_id in person_ids:
            references.update(self.referenced_media_paths(person_id))
        return references

    def _image_candidates(self, person_id: str) -> list[str]:
        values: list[str | None] = []
        person = self._conn.execute(
            "SELECT profile_image FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        values.append(person["profile_image"] if person is not None else None)
        values.extend(
            row["path"]
            for row in self._conn.execute(
                "SELECT path FROM person_gallery "
                "WHERE person_id=? AND crop_type='face' "
                "ORDER BY sharpness DESC, id DESC",
                (person_id,),
            ).fetchall()
        )
        values.extend(
            row["best_face_crop"]
            for row in self._conn.execute(
                "SELECT best_face_crop FROM recognition_log "
                "WHERE person_id=? AND best_face_crop IS NOT NULL "
                "ORDER BY id DESC",
                (person_id,),
            ).fetchall()
        )
        candidates: list[str] = []
        for value in values:
            path = self._public_media_path(value)
            if path and path not in candidates:
                candidates.append(path)
        return candidates

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _require_writable(self, operation: str) -> None:
        if self.read_only:
            raise ReadOnlyGlobalMemoryError(
                f"Global Memory operation '{operation}' is unavailable in read-only mode."
            )

    def __enter__(self) -> GlobalMemory:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_schema_columns(self) -> None:
        person_columns = self._table_columns("persons")
        if "profile_image" not in person_columns:
            self._conn.execute("ALTER TABLE persons ADD COLUMN profile_image TEXT DEFAULT NULL")
        if "profile_image_source" not in person_columns:
            self._conn.execute("ALTER TABLE persons ADD COLUMN profile_image_source TEXT NOT NULL DEFAULT 'auto'")

        log_columns = self._table_columns("recognition_log")
        if "best_face_crop" not in log_columns:
            self._conn.execute("ALTER TABLE recognition_log ADD COLUMN best_face_crop TEXT DEFAULT NULL")

        appearance_columns = self._table_columns("appearances")
        if "clothing_status" not in appearance_columns:
            self._conn.execute(
                "ALTER TABLE appearances ADD COLUMN clothing_status "
                "TEXT NOT NULL DEFAULT 'not_attempted'"
            )

        person_columns = self._table_columns("persons")
        if "is_active" not in person_columns:
            self._conn.execute(
                "ALTER TABLE persons ADD COLUMN is_active INTEGER NOT NULL "
                "DEFAULT 1 CHECK (is_active IN (0, 1))"
            )
        if "merged_into_person_id" not in person_columns:
            self._conn.execute(
                "ALTER TABLE persons ADD COLUMN merged_into_person_id TEXT DEFAULT NULL"
            )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_persons_active ON persons(is_active)"
        )
        self._ensure_identity_evidence_schema()
        self._ensure_identity_merge_audit_schema()

    def _ensure_identity_evidence_schema(self) -> None:
        """Create the Phase 4 ledger when opening a pre-Phase 4 database."""
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_evidence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL REFERENCES persons(person_id),
                evidence_key TEXT NOT NULL,
                crop_type TEXT NOT NULL CHECK (crop_type IN ('face', 'body')),
                canonical_path TEXT NOT NULL,
                embedding_applied INTEGER NOT NULL DEFAULT 0
                    CHECK (embedding_applied IN (0, 1)),
                observation_weight INTEGER NOT NULL DEFAULT 0
                    CHECK (observation_weight >= 0),
                created_at TEXT NOT NULL,
                UNIQUE(person_id, evidence_key)
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_identity_evidence_person "
            "ON identity_evidence(person_id)"
        )

    def _ensure_identity_merge_audit_schema(self) -> None:
        columns = {
            row["name"]: row
            for row in self._conn.execute(
                "PRAGMA table_info(identity_merge_audit)"
            ).fetchall()
        }
        schema_row = self._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='identity_merge_audit'"
        ).fetchone()
        schema_sql = "" if schema_row is None else str(schema_row["sql"] or "")
        compact_schema = "".join(schema_sql.lower().split())
        required_not_null = {
            "source_person_id",
            "target_person_id",
            "decision_source",
            "source_embedding",
            "source_embedding_count",
            "created_at",
        }
        hardened = (
            required_not_null <= columns.keys()
            and all(int(columns[name]["notnull"]) == 1 for name in required_not_null)
            and "check(source_embedding_count>0)" in compact_schema
            and "check(source_person_id<>target_person_id)" in compact_schema
        )
        if hardened:
            return

        invalid = self._conn.execute(
            "SELECT merge_id FROM identity_merge_audit WHERE "
            "source_person_id IS NULL OR target_person_id IS NULL OR "
            "decision_source IS NULL OR source_embedding IS NULL OR "
            "source_embedding_count IS NULL OR source_embedding_count <= 0 OR "
            "created_at IS NULL OR source_person_id = target_person_id LIMIT 1"
        ).fetchone()
        if invalid is not None:
            raise sqlite3.IntegrityError(
                "identity_merge_audit contains rows that violate required constraints"
            )

        self._conn.execute("DROP TRIGGER IF EXISTS trg_identity_merge_audit_no_update")
        self._conn.execute("DROP TRIGGER IF EXISTS trg_identity_merge_audit_no_delete")
        self._conn.execute("DROP INDEX IF EXISTS idx_merge_audit_source")
        self._conn.execute("DROP INDEX IF EXISTS idx_merge_audit_target")
        self._conn.execute(
            "ALTER TABLE identity_merge_audit RENAME TO identity_merge_audit_legacy"
        )
        self._conn.execute(
            """
            CREATE TABLE identity_merge_audit (
                merge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_person_id TEXT NOT NULL,
                target_person_id TEXT NOT NULL,
                reason TEXT,
                decision_source TEXT NOT NULL,
                similarity REAL,
                source_embedding BLOB NOT NULL,
                source_embedding_count INTEGER NOT NULL
                    CHECK (source_embedding_count > 0),
                source_name TEXT,
                target_name_before TEXT,
                created_at TEXT NOT NULL,
                CHECK (source_person_id <> target_person_id)
            )
            """
        )
        self._conn.execute(
            """
            INSERT INTO identity_merge_audit(
                merge_id, source_person_id, target_person_id, reason,
                decision_source, similarity, source_embedding,
                source_embedding_count, source_name, target_name_before,
                created_at
            )
            SELECT
                merge_id, source_person_id, target_person_id, reason,
                decision_source, similarity, source_embedding,
                source_embedding_count, source_name, target_name_before,
                created_at
            FROM identity_merge_audit_legacy
            ORDER BY merge_id
            """
        )
        self._conn.execute("DROP TABLE identity_merge_audit_legacy")
        self._conn.execute(
            "CREATE INDEX idx_merge_audit_source "
            "ON identity_merge_audit(source_person_id)"
        )
        self._conn.execute(
            "CREATE INDEX idx_merge_audit_target "
            "ON identity_merge_audit(target_person_id)"
        )
        self._conn.execute(
            """
            CREATE TRIGGER trg_identity_merge_audit_no_update
            BEFORE UPDATE ON identity_merge_audit
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'identity_merge_audit is append-only: UPDATE is not allowed'
                );
            END
            """
        )
        self._conn.execute(
            """
            CREATE TRIGGER trg_identity_merge_audit_no_delete
            BEFORE DELETE ON identity_merge_audit
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'identity_merge_audit is append-only: DELETE is not allowed'
                );
            END
            """
        )

    def _table_columns(self, table: str) -> set[str]:
        rows = self._conn.execute(f"PRAGMA table_info({table})").fetchall()
        return {row["name"] for row in rows}

    def _next_person_id(self) -> tuple[str, str]:
        self._conn.execute(
            "UPDATE counters SET value = value + 1 WHERE key = 'person_count'"
        )
        row = self._conn.execute(
            "SELECT value FROM counters WHERE key = 'person_count'"
        ).fetchone()
        n = int(row["value"])
        person_id = f"person_{n:03d}"
        name = f"Person {n:03d}"

        # Defensive guard for pre-migration databases: never collide with an
        # existing ID even if the counter was not seeded yet.
        while self._conn.execute("SELECT 1 FROM persons WHERE person_id=?", (person_id,)).fetchone():
            self._conn.execute(
                "UPDATE counters SET value = value + 1 WHERE key = 'person_count'"
            )
            row = self._conn.execute(
                "SELECT value FROM counters WHERE key = 'person_count'"
            ).fetchone()
            n = int(row["value"])
            person_id = f"person_{n:03d}"
            name = f"Person {n:03d}"
        return person_id, name

    def _active_identity_similarities(
        self,
        embedding: np.ndarray,
    ) -> list[tuple[sqlite3.Row, float]]:
        rows = self._conn.execute(
            "SELECT person_id, name, embedding, embedding_count "
            "FROM persons WHERE is_active=1"
        ).fetchall()
        if not rows:
            return []

        matrix = np.vstack([
            np.frombuffer(row["embedding"], dtype=np.float32).copy() for row in rows
        ])
        sims = matrix @ embedding
        return [(row, float(sims[index])) for index, row in enumerate(rows)]

    def _find_existing_person(self, embedding: np.ndarray, threshold: float | None = None) -> dict | None:
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else float(threshold)
        similarities = self._active_identity_similarities(embedding)
        if not similarities:
            return None

        best_idx = int(np.argmax([similarity for _row, similarity in similarities]))
        row, best_sim = similarities[best_idx]
        if best_sim < threshold:
            return None

        return {
            "person_id": row["person_id"],
            "name": row["name"],
            "similarity": best_sim,
            "embedding_count": row["embedding_count"],
        }

    def _rank_active_identity_candidates(
        self,
        embedding: np.ndarray,
    ) -> tuple[IdentityCandidate, ...]:
        candidates: list[IdentityCandidate] = []
        for row, similarity in self._active_identity_similarities(embedding):
            stored = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            if stored.size == 0:
                raise IdentityPolicyInputError(
                    f"active person {row['person_id']} has an empty embedding"
                )
            if stored.shape != embedding.shape:
                raise IdentityPolicyInputError(
                    f"active person {row['person_id']} has an incompatible embedding shape"
                )
            if not np.isfinite(stored).all():
                raise IdentityPolicyInputError(
                    f"active person {row['person_id']} has a non-finite embedding"
                )
            if float(np.linalg.norm(stored)) <= 0:
                raise IdentityPolicyInputError(
                    f"active person {row['person_id']} has a zero-norm embedding"
                )
            if not np.isfinite(similarity):
                raise IdentityPolicyInputError("candidate similarity must be finite")
            candidates.append(IdentityCandidate(row["person_id"], similarity))
        candidates.sort(key=lambda item: (-item.similarity, item.person_id))
        return tuple(candidates[:2])

    @staticmethod
    def _validated_merge_text(value: object, label: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise InvalidMergeRequestError(f"{label} must be a non-empty string")
        return value.strip()

    def _resolve_canonical_person_id_locked(self, person_id: str) -> str:
        row = self._conn.execute(
            "SELECT person_id, is_active, merged_into_person_id "
            "FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        if row is None:
            raise PersonNotFoundError(f"person {person_id!r} does not exist")

        redirect = row["merged_into_person_id"]
        if bool(row["is_active"]):
            if redirect is not None:
                raise RedirectChainError(
                    f"active person {person_id!r} has an invalid redirect"
                )
            return person_id

        if redirect is None:
            raise InactiveSourceError(
                f"inactive person {person_id!r} has no canonical redirect"
            )
        if redirect == person_id:
            raise RedirectChainError(f"person {person_id!r} redirects to itself")
        child = self._conn.execute(
            "SELECT person_id FROM persons "
            "WHERE merged_into_person_id=? AND person_id<>? LIMIT 1",
            (person_id, person_id),
        ).fetchone()
        if child is not None:
            raise RedirectChainError(
                f"redirect through {person_id!r} would form a chain"
            )

        target = self._conn.execute(
            "SELECT person_id, is_active, merged_into_person_id "
            "FROM persons WHERE person_id=?",
            (redirect,),
        ).fetchone()
        if target is None:
            raise RedirectChainError(
                f"person {person_id!r} redirects to missing person {redirect!r}"
            )
        if not bool(target["is_active"]):
            raise RedirectChainError(
                f"person {person_id!r} redirects to inactive person {redirect!r}"
            )
        if target["merged_into_person_id"] is not None:
            raise RedirectChainError(
                f"person {person_id!r} redirects through a non-canonical target"
            )
        return str(target["person_id"])

    def _get_identity_lineage_locked(self, person_id: str) -> IdentityLineage:
        canonical_id = self._resolve_canonical_person_id_locked(person_id)
        rows = self._conn.execute(
            """
            SELECT person_id, is_active, merged_into_person_id
              FROM persons
             WHERE person_id=? OR merged_into_person_id=?
            """,
            (canonical_id, canonical_id),
        ).fetchall()
        by_id = {str(row["person_id"]): row for row in rows}
        canonical = by_id.get(canonical_id)
        if canonical is None or not bool(canonical["is_active"]):
            raise InactiveTargetError(
                f"canonical person {canonical_id!r} is not active"
            )
        if canonical["merged_into_person_id"] is not None:
            raise RedirectChainError(
                f"canonical person {canonical_id!r} redirects elsewhere"
            )

        source_ids = sorted(person for person in by_id if person != canonical_id)
        members = [IdentityLineageMember(canonical_id, True, None)]
        for source_id in source_ids:
            source = by_id[source_id]
            if bool(source["is_active"]):
                raise RedirectChainError(
                    f"active person {source_id!r} redirects to {canonical_id!r}"
                )
            if source["merged_into_person_id"] != canonical_id:
                raise RedirectChainError(
                    f"person {source_id!r} has an invalid lineage redirect"
                )
            child = self._conn.execute(
                "SELECT person_id FROM persons "
                "WHERE merged_into_person_id=? AND person_id<>? LIMIT 1",
                (source_id, source_id),
            ).fetchone()
            if child is not None:
                raise RedirectChainError(
                    f"lineage contains a redirect chain through {source_id!r}"
                )
            members.append(
                IdentityLineageMember(source_id, False, canonical_id)
            )
        return IdentityLineage(canonical_id, tuple(members))

    @staticmethod
    def _lineage_evidence_rows(
        lineage: IdentityLineage,
        rows: list[sqlite3.Row],
    ) -> list[dict[str, Any]]:
        member_order = {
            person_id: index
            for index, person_id in enumerate(lineage.member_person_ids)
        }
        ordered = sorted(
            rows,
            key=lambda row: (member_order[str(row["person_id"])], int(row["id"])),
        )
        evidence: list[dict[str, Any]] = []
        for row in ordered:
            item = dict(row)
            item["original_person_id"] = str(row["person_id"])
            item["canonical_person_id"] = lineage.canonical_person_id
            evidence.append(item)
        return evidence

    @staticmethod
    def _validated_review_pagination(limit: int, offset: int) -> tuple[int, int]:
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise InvalidReviewRequestError("limit must be an integer")
        if limit < 1 or limit > IDENTITY_REVIEW_MAX_LIMIT:
            raise InvalidReviewRequestError(
                f"limit must be between 1 and {IDENTITY_REVIEW_MAX_LIMIT}"
            )
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or offset > IDENTITY_REVIEW_MAX_OFFSET
        ):
            raise InvalidReviewRequestError(
                f"offset must be between 0 and {IDENTITY_REVIEW_MAX_OFFSET}"
            )
        return limit, offset

    @staticmethod
    def _validated_review_suggestion_id(suggestion_id: str | int) -> int:
        if isinstance(suggestion_id, bool):
            raise InvalidReviewRequestError("suggestion_id must be a positive integer")
        if isinstance(suggestion_id, int):
            review_id = suggestion_id
        elif isinstance(suggestion_id, str):
            value = suggestion_id
            if not re.fullmatch(r"[0-9]+", value, flags=re.ASCII):
                raise InvalidReviewRequestError(
                    "suggestion_id must be a positive integer"
                )
            if len(value) > len(str(SQLITE_MAX_INTEGER)):
                raise InvalidReviewRequestError(
                    "suggestion_id must be a positive SQLite integer"
                )
            try:
                review_id = int(value)
            except ValueError as exc:
                raise InvalidReviewRequestError(
                    "suggestion_id must be a positive integer"
                ) from exc
        else:
            raise InvalidReviewRequestError("suggestion_id must be a positive integer")
        if review_id < 1 or review_id > SQLITE_MAX_INTEGER:
            raise InvalidReviewRequestError("suggestion_id must be a positive integer")
        return review_id

    @staticmethod
    def _validated_review_decision(
        decision: IdentityReviewDecision | str,
    ) -> IdentityReviewDecision:
        if isinstance(decision, IdentityReviewDecision):
            return decision
        if isinstance(decision, str):
            try:
                return IdentityReviewDecision(decision.strip())
            except ValueError as exc:
                raise InvalidReviewDecisionError(
                    "decision must be 'accept' or 'reject'"
                ) from exc
        raise InvalidReviewDecisionError("decision must be 'accept' or 'reject'")

    @staticmethod
    def _validated_optional_review_text(
        value: str | None,
        label: str,
    ) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise InvalidReviewRequestError(f"{label} must be a string or null")
        normalized = value.strip()
        if not normalized:
            return None
        if len(normalized) > 500:
            raise InvalidReviewRequestError(f"{label} is too long (max 500 characters)")
        return normalized

    def _load_identity_review_row(self, review_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM identity_match_suggestions WHERE id=?",
            (review_id,),
        ).fetchone()
        if row is None:
            raise ReviewSuggestionNotFoundError(
                f"identity review suggestion {review_id} does not exist"
            )
        return row

    def _load_identity_review_summary_row(self, review_id: int) -> sqlite3.Row:
        row = self._conn.execute(
            """
            SELECT s.*,
                   source.name AS source_name,
                   source.profile_image AS source_profile_image,
                   candidate.name AS candidate_name,
                   candidate.profile_image AS candidate_profile_image
              FROM identity_match_suggestions AS s
              JOIN persons AS source ON source.person_id = s.source_person_id
              JOIN persons AS candidate ON candidate.person_id = s.candidate_person_id
             WHERE s.id=?
            """,
            (review_id,),
        ).fetchone()
        if row is None:
            suggestion = self._conn.execute(
                "SELECT id FROM identity_match_suggestions WHERE id=?",
                (review_id,),
            ).fetchone()
            if suggestion is None:
                raise ReviewSuggestionNotFoundError(
                    f"identity review suggestion {review_id} does not exist"
                )
            raise ReviewSuggestionIntegrityError(
                f"identity review suggestion {review_id} references a missing person"
            )
        return row

    def _identity_review_summary_from_row(
        self,
        row: sqlite3.Row,
    ) -> IdentityReviewSummary:
        source_id = str(row["source_person_id"])
        candidate_id = str(row["candidate_person_id"])
        source_candidates = self._image_candidates(source_id)
        candidate_candidates = self._image_candidates(candidate_id)
        return IdentityReviewSummary(
            suggestion_id=str(row["id"]),
            status=str(row["status"]),
            source_person_id=source_id,
            candidate_person_id=candidate_id,
            similarity=float(row["similarity"]),
            second_similarity=(
                None
                if row["second_similarity"] is None
                else float(row["second_similarity"])
            ),
            margin=None if row["margin"] is None else float(row["margin"]),
            reason=None if row["reason"] is None else str(row["reason"]),
            created_at=str(row["created_at"]),
            reviewed_at=(
                None if row["reviewed_at"] is None else str(row["reviewed_at"])
            ),
            reviewed_by=(
                None if row["reviewed_by"] is None else str(row["reviewed_by"])
            ),
            source_name=str(row["source_name"]),
            candidate_name=str(row["candidate_name"]),
            source_preview=source_candidates[0] if source_candidates else None,
            candidate_preview=(
                candidate_candidates[0] if candidate_candidates else None
            ),
        )

    def _load_review_person(self, person_id: str, role: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        if row is None:
            raise ReviewSuggestionIntegrityError(
                f"identity review {role} person {person_id!r} does not exist"
            )
        return row

    def _validate_review_relationship(
        self,
        suggestion: sqlite3.Row,
        source: sqlite3.Row,
        candidate: sqlite3.Row,
    ) -> None:
        review_id = int(suggestion["id"])
        if source["person_id"] == candidate["person_id"]:
            raise ReviewSuggestionIntegrityError(
                "identity review source and candidate must differ"
            )
        status = str(suggestion["status"])
        if status == "pending":
            self._validate_pending_review_people(source, candidate)
        elif status == "accepted":
            self._validate_accepted_review_state(review_id, source, candidate)
        elif status == "rejected":
            self._validate_rejected_review_state(review_id, source, candidate)
        elif status != "stale":
            raise ReviewSuggestionIntegrityError(
                f"identity review suggestion {review_id} has invalid status {status!r}"
            )

    def _validate_pending_review_people(
        self,
        source: sqlite3.Row,
        candidate: sqlite3.Row,
    ) -> None:
        source_id = str(source["person_id"])
        candidate_id = str(candidate["person_id"])
        if not bool(source["is_active"]) or source["merged_into_person_id"] is not None:
            redirect = source["merged_into_person_id"]
            detail = "inactive" if redirect is None else f"merged into {redirect!r}"
            raise ReviewSuggestionConflictError(
                f"pending review source {source_id!r} is already {detail}"
            )
        if (
            not bool(candidate["is_active"])
            or candidate["merged_into_person_id"] is not None
        ):
            raise ReviewSuggestionConflictError(
                f"pending review candidate {candidate_id!r} is not canonical and active"
            )

    @staticmethod
    def _review_audit_source(review_id: int) -> str:
        return f"identity_review:{review_id}"

    def _validate_accepted_review_state(
        self,
        review_id: int,
        source: sqlite3.Row,
        candidate: sqlite3.Row,
    ) -> None:
        source_id = str(source["person_id"])
        candidate_id = str(candidate["person_id"])
        if bool(source["is_active"]) or source["merged_into_person_id"] != candidate_id:
            raise ReviewSuggestionIntegrityError(
                f"accepted review {review_id} has no matching source redirect"
            )
        if not bool(candidate["is_active"]) or candidate["merged_into_person_id"] is not None:
            raise ReviewSuggestionIntegrityError(
                f"accepted review {review_id} candidate is not canonical and active"
            )
        audits = self._conn.execute(
            "SELECT * FROM identity_merge_audit WHERE source_person_id=?",
            (source_id,),
        ).fetchall()
        if len(audits) != 1:
            raise ReviewSuggestionIntegrityError(
                f"accepted review {review_id} does not have exactly one merge audit"
            )
        audit = audits[0]
        if (
            audit["target_person_id"] != candidate_id
            or audit["decision_source"] != self._review_audit_source(review_id)
            or bytes(audit["source_embedding"]) != bytes(source["embedding"])
            or audit["source_embedding_count"] != source["embedding_count"]
        ):
            raise ReviewSuggestionIntegrityError(
                f"accepted review {review_id} redirect and merge audit disagree"
            )

    def _require_no_review_merge_audit(self, review_id: int) -> None:
        audit = self._conn.execute(
            "SELECT merge_id FROM identity_merge_audit WHERE decision_source=? LIMIT 1",
            (self._review_audit_source(review_id),),
        ).fetchone()
        if audit is not None:
            raise ReviewSuggestionIntegrityError(
                f"rejected review {review_id} has a review merge audit"
            )

    def _validate_rejected_review_state(
        self,
        review_id: int,
        source: sqlite3.Row,
        candidate: sqlite3.Row,
    ) -> None:
        source_id = str(source["person_id"])
        candidate_id = str(candidate["person_id"])
        if source_id == candidate_id:
            raise ReviewSuggestionIntegrityError(
                "identity review source and candidate must differ"
            )
        if not bool(source["is_active"]) or source["merged_into_person_id"] is not None:
            raise ReviewSuggestionIntegrityError(
                f"rejected review {review_id} source is not active and unredirected"
            )
        if (
            not bool(candidate["is_active"])
            or candidate["merged_into_person_id"] is not None
        ):
            raise ReviewSuggestionIntegrityError(
                f"rejected review {review_id} candidate is not a separate canonical profile"
            )
        self._require_no_review_merge_audit(review_id)
        try:
            self._require_no_source_merge_audit(source_id)
        except MergeAuditIntegrityError as exc:
            raise ReviewSuggestionIntegrityError(
                f"rejected review {review_id} source has a merge audit"
            ) from exc

    def _replay_accepted_identity_review(
        self,
        review_id: int,
        source: sqlite3.Row,
        candidate: sqlite3.Row,
    ) -> IdentityReviewDecisionResult:
        self._validate_accepted_review_state(review_id, source, candidate)
        try:
            merge = self._replay_person_merge(source, candidate)
        except PersonMergeError as exc:
            raise ReviewSuggestionIntegrityError(
                f"accepted review {review_id} merge state is inconsistent"
            ) from exc
        return IdentityReviewDecisionResult(
            suggestion_id=str(review_id),
            decision=IdentityReviewDecision.ACCEPT,
            status="accepted",
            source_person_id=merge.source_person_id,
            target_person_id=merge.target_person_id,
            audit_id=merge.audit_id,
            idempotent_replay=True,
            staled_suggestion_count=0,
            source_embedding_count=merge.source_embedding_count,
            target_embedding_count_before=merge.target_embedding_count_before,
            target_embedding_count_after=merge.target_embedding_count_after,
        )

    @staticmethod
    def _rejected_identity_review_result(
        review_id: int,
        source_id: str,
        candidate_id: str,
        *,
        idempotent_replay: bool,
    ) -> IdentityReviewDecisionResult:
        return IdentityReviewDecisionResult(
            suggestion_id=str(review_id),
            decision=IdentityReviewDecision.REJECT,
            status="rejected",
            source_person_id=source_id,
            target_person_id=candidate_id,
            audit_id=None,
            idempotent_replay=idempotent_replay,
            staled_suggestion_count=0,
            source_embedding_count=None,
            target_embedding_count_before=None,
            target_embedding_count_after=None,
        )

    def _update_identity_review_status(
        self,
        review_id: int,
        *,
        status: str,
        reviewed_at: str,
        reviewed_by: str | None,
    ) -> None:
        cursor = self._conn.execute(
            "UPDATE identity_match_suggestions "
            "SET status=?, reviewed_at=?, reviewed_by=? "
            "WHERE id=? AND status='pending'",
            (status, reviewed_at, reviewed_by, review_id),
        )
        if cursor.rowcount != 1:
            raise ReviewSuggestionConflictError(
                f"identity review suggestion {review_id} changed concurrently"
            )

    def _before_review_commit(self) -> None:
        """Failure-injection seam; intentionally performs no work."""

    def _review_profile_summary(self, person: sqlite3.Row) -> dict[str, Any]:
        person_id = str(person["person_id"])
        redirect = person["merged_into_person_id"]
        return {
            "person_id": person_id,
            "name": str(person["name"]),
            "embedding_count": int(person["embedding_count"]),
            "enrolled_at": str(person["enrolled_at"]),
            "updated_at": str(person["updated_at"]),
            "cameras": self._safe_review_sources(self._json_list(person["cameras"])),
            "profile_image": self._public_media_path(person["profile_image"]),
            "profile_image_source": str(person["profile_image_source"]),
            "is_active": bool(person["is_active"]),
            "merged_into_person_id": None if redirect is None else str(redirect),
            "canonical_person_id": person_id if redirect is None else str(redirect),
        }

    @staticmethod
    def _review_lineage_summary(person: sqlite3.Row) -> tuple[dict[str, Any], ...]:
        person_id = str(person["person_id"])
        redirect = person["merged_into_person_id"]
        return ({
            "person_id": person_id,
            "is_active": bool(person["is_active"]),
            "merged_into_person_id": None if redirect is None else str(redirect),
            "canonical_person_id": person_id if redirect is None else str(redirect),
        },)

    def _review_gallery(self, person_ids: tuple[str, ...]) -> tuple[dict[str, Any], ...]:
        placeholders = ",".join("?" for _ in person_ids)
        rows = self._conn.execute(
            f"SELECT * FROM person_gallery WHERE person_id IN ({placeholders}) "
            "ORDER BY crop_type ASC, sharpness DESC, id DESC",
            person_ids,
        ).fetchall()
        counts = {"face": 0, "body": 0}
        evidence: list[dict[str, Any]] = []
        for row in rows:
            crop_type = str(row["crop_type"])
            if crop_type not in counts or counts[crop_type] >= IDENTITY_REVIEW_GALLERY_LIMIT_PER_TYPE:
                continue
            path = self._public_media_path(row["path"])
            if path is None:
                continue
            counts[crop_type] += 1
            evidence.append({
                "id": str(row["id"]),
                "original_person_id": str(row["person_id"]),
                "crop_type": crop_type,
                "path": path,
                "sharpness": float(row["sharpness"]),
                "session_date": str(row["session_date"]),
                "video_source": self._safe_review_source(row["video_source"]),
                "width": row["width"],
                "height": row["height"],
            })
        return tuple(evidence)

    def _review_appearances(
        self,
        person_ids: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        placeholders = ",".join("?" for _ in person_ids)
        rows = self._conn.execute(
            f"SELECT * FROM appearances WHERE person_id IN ({placeholders}) "
            "ORDER BY date DESC, id DESC LIMIT ?",
            (*person_ids, IDENTITY_REVIEW_APPEARANCE_LIMIT),
        ).fetchall()
        results = []
        for row in rows:
            item = self._appearance_from_row(row)
            item.update({
                "id": str(row["id"]),
                "original_person_id": str(row["person_id"]),
                "video_sources": self._safe_review_sources(
                    self._json_list(row["video_sources"])
                ),
            })
            results.append(item)
        return tuple(results)

    def _review_recognition_events(
        self,
        person_ids: tuple[str, ...],
    ) -> tuple[dict[str, Any], ...]:
        placeholders = ",".join("?" for _ in person_ids)
        rows = self._conn.execute(
            f"SELECT * FROM recognition_log WHERE person_id IN ({placeholders}) "
            "ORDER BY ts DESC, id DESC LIMIT ?",
            (*person_ids, IDENTITY_REVIEW_RECOGNITION_LIMIT),
        ).fetchall()
        return tuple({
            "id": str(row["id"]),
            "original_person_id": str(row["person_id"]),
            "event_type": str(row["event_type"]),
            "similarity": row["similarity"],
            "embedding_count_before": row["embedding_count_before"],
            "embedding_count_after": row["embedding_count_after"],
            "video_sources": self._safe_review_sources(
                self._json_list(row["video_sources"])
            ),
            "best_face_crop": self._public_media_path(row["best_face_crop"]),
            "ts": str(row["ts"]),
        } for row in rows)

    @staticmethod
    def _safe_review_source(value: object) -> str | None:
        if value is None:
            return None
        display = str(value).strip()
        if not display:
            return None
        parsed = urlsplit(display)
        if parsed.scheme and parsed.netloc:
            host = parsed.hostname or ""
            if parsed.port is not None:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, ""))
        if Path(display).is_absolute() or PureWindowsPath(display).is_absolute():
            return PureWindowsPath(display).name or Path(display).name
        return display

    @classmethod
    def _safe_review_sources(cls, values: list) -> list[str]:
        return [safe for value in values if (safe := cls._safe_review_source(value))]

    def _load_merge_person(self, person_id: str, role: str) -> sqlite3.Row:
        row = self._conn.execute(
            "SELECT * FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        if row is None:
            raise PersonNotFoundError(f"{role} person {person_id!r} does not exist")
        return row

    @staticmethod
    def _validate_merge_target(target: sqlite3.Row) -> None:
        target_id = str(target["person_id"])
        if not bool(target["is_active"]):
            raise InactiveTargetError(f"target person {target_id!r} is inactive")
        if target["merged_into_person_id"] is not None:
            raise RedirectChainError(
                f"target person {target_id!r} redirects elsewhere"
            )

    def _reject_redirect_children(self, source_id: str) -> None:
        child = self._conn.execute(
            "SELECT person_id FROM persons "
            "WHERE merged_into_person_id=? AND person_id<>? LIMIT 1",
            (source_id, source_id),
        ).fetchone()
        if child is not None:
            raise RedirectChainError(
                f"merging {source_id!r} would create a redirect chain"
            )

    def _require_no_source_merge_audit(self, source_id: str) -> None:
        row = self._conn.execute(
            "SELECT merge_id FROM identity_merge_audit "
            "WHERE source_person_id=? LIMIT 1",
            (source_id,),
        ).fetchone()
        if row is not None:
            raise MergeAuditIntegrityError(
                f"active source person {source_id!r} already has a merge audit"
            )

    @staticmethod
    def _validated_stored_merge_embedding(
        person: sqlite3.Row,
        role: str,
    ) -> np.ndarray:
        raw = person["embedding"]
        if raw is None:
            raise InvalidMergeEmbeddingError(f"{role} embedding is missing")
        try:
            blob = bytes(raw)
        except (TypeError, ValueError) as exc:
            raise InvalidMergeEmbeddingError(
                f"{role} embedding is malformed"
            ) from exc
        item_size = np.dtype(np.float32).itemsize
        if not blob or len(blob) % item_size:
            raise InvalidMergeEmbeddingError(f"{role} embedding is malformed")
        embedding = np.frombuffer(blob, dtype=np.float32).copy()
        if embedding.ndim != 1 or embedding.size == 0:
            raise InvalidMergeEmbeddingError(f"{role} embedding is malformed")
        if not np.isfinite(embedding).all():
            raise InvalidMergeEmbeddingError(f"{role} embedding must be finite")
        norm = float(np.linalg.norm(embedding))
        if not np.isfinite(norm) or norm <= 0:
            raise InvalidMergeEmbeddingError(f"{role} embedding must be non-zero")
        return embedding

    @staticmethod
    def _validated_stored_merge_count(person: sqlite3.Row, role: str) -> int:
        count = person["embedding_count"]
        if (
            isinstance(count, bool)
            or type(count) is not int
            or count <= 0
            or count > SQLITE_MAX_INTEGER
        ):
            raise InvalidMergeEmbeddingError(
                f"{role} embedding_count must be a positive integer "
                "within the SQLite signed range"
            )
        return count

    @staticmethod
    def _validated_merge_count_sum(target_count: int, source_count: int) -> int:
        if target_count > SQLITE_MAX_INTEGER - source_count:
            raise InvalidMergeEmbeddingError(
                "combined embedding_count exceeds SQLite signed INTEGER range"
            )
        return target_count + source_count

    @staticmethod
    def _validated_merge_cameras(value: object, role: str) -> list[str]:
        if not isinstance(value, str):
            raise InvalidMergeMetadataError(
                f"{role} cameras must be a JSON array of strings"
            )
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise InvalidMergeMetadataError(
                f"{role} cameras contain malformed JSON"
            ) from exc
        if not isinstance(decoded, list) or any(
            not isinstance(camera, str) for camera in decoded
        ):
            raise InvalidMergeMetadataError(
                f"{role} cameras must be a JSON array of strings"
            )
        return decoded

    @staticmethod
    def _combined_merge_embedding(
        *,
        target_embedding: np.ndarray,
        target_count: int,
        source_embedding: np.ndarray,
        source_count: int,
        weight_scale: int,
    ) -> np.ndarray:
        target_safe = target_embedding.astype(np.float64, copy=False)
        source_safe = source_embedding.astype(np.float64, copy=False)
        weighted = target_safe * target_count + source_safe * source_count
        if not np.isfinite(weighted).all():
            raise InvalidMergeEmbeddingError(
                "combined target embedding contains non-finite values"
            )
        weighted_norm = float(np.linalg.norm(weighted))
        minimum_safe_norm = max(
            float(np.finfo(np.float64).eps),
            float(weight_scale) * MERGE_WEIGHTED_NORM_RELATIVE_MINIMUM,
        )
        if not np.isfinite(weighted_norm) or weighted_norm <= minimum_safe_norm:
            raise InvalidMergeEmbeddingError(
                "combined target embedding is cancelled or numerically unstable"
            )
        combined = weighted / weighted_norm
        if (
            combined.shape != target_embedding.shape
            or not np.isfinite(combined).all()
        ):
            raise InvalidMergeEmbeddingError(
                "combined target embedding is invalid"
            )
        return combined

    def _insert_person_merge_audit(
        self,
        *,
        source: sqlite3.Row,
        target: sqlite3.Row,
        reason: str,
        decision_source: str,
        source_embedding_count: int,
        created_at: str,
    ) -> int:
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO identity_merge_audit(
                    source_person_id, target_person_id, reason,
                    decision_source, similarity, source_embedding,
                    source_embedding_count, source_name,
                    target_name_before, created_at
                ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?)
                """,
                (
                    source["person_id"],
                    target["person_id"],
                    reason,
                    decision_source,
                    sqlite3.Binary(bytes(source["embedding"])),
                    source_embedding_count,
                    source["name"],
                    target["name"],
                    created_at,
                ),
            )
        except sqlite3.DatabaseError as exc:
            raise MergeAuditIntegrityError("merge audit insertion failed") from exc
        return int(cursor.lastrowid)

    def _update_merge_target_embedding(
        self,
        target_id: str,
        embedding: np.ndarray,
        embedding_count: int,
        updated_at: str,
    ) -> None:
        cursor = self._conn.execute(
            "UPDATE persons SET embedding=?, embedding_count=?, updated_at=? "
            "WHERE person_id=?",
            (
                sqlite3.Binary(embedding.astype(np.float32).tobytes()),
                embedding_count,
                updated_at,
                target_id,
            ),
        )
        if cursor.rowcount != 1:
            raise PersonMergeError("target embedding update did not affect one person")

    def _update_merge_target_metadata(
        self,
        target_id: str,
        cameras: list,
    ) -> None:
        cursor = self._conn.execute(
            "UPDATE persons SET cameras=? WHERE person_id=?",
            (json.dumps(cameras), target_id),
        )
        if cursor.rowcount != 1:
            raise PersonMergeError("target metadata update did not affect one person")

    def _stale_merge_suggestions(
        self,
        source_id: str,
        *,
        preserve_suggestion_id: int | None = None,
    ) -> int:
        sql = (
            "UPDATE identity_match_suggestions SET status='stale' "
            "WHERE status='pending' "
            "AND (source_person_id=? OR candidate_person_id=?)"
        )
        parameters: tuple[object, ...] = (source_id, source_id)
        if preserve_suggestion_id is not None:
            sql += " AND id<>?"
            parameters = (*parameters, preserve_suggestion_id)
        cursor = self._conn.execute(sql, parameters)
        return int(cursor.rowcount)

    def _deactivate_merge_source(self, source_id: str) -> None:
        cursor = self._conn.execute(
            "UPDATE persons SET is_active=0 WHERE person_id=? AND is_active=1",
            (source_id,),
        )
        if cursor.rowcount != 1:
            raise PersonMergeError("source deactivation did not affect one person")

    def _redirect_merge_source(self, source_id: str, target_id: str) -> None:
        cursor = self._conn.execute(
            "UPDATE persons SET merged_into_person_id=? "
            "WHERE person_id=? AND is_active=0 AND merged_into_person_id IS NULL",
            (target_id, source_id),
        )
        if cursor.rowcount != 1:
            raise PersonMergeError("source redirect did not affect one person")

    def _assert_merge_foreign_keys(self) -> None:
        violations = self._conn.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise PersonMergeError("foreign-key violations prevent person merge")

    def _merge_lineage_member_count(self, target_id: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM persons "
            "WHERE person_id=? OR (is_active=0 AND merged_into_person_id=?)",
            (target_id, target_id),
        ).fetchone()
        return int(row[0])

    def _replay_person_merge(
        self,
        source: sqlite3.Row,
        target: sqlite3.Row,
    ) -> PersonMergeResult:
        source_id = str(source["person_id"])
        target_id = str(target["person_id"])
        self._reject_redirect_children(source_id)
        source_embedding = self._validated_stored_merge_embedding(source, "source")
        target_embedding = self._validated_stored_merge_embedding(target, "target")
        if source_embedding.shape != target_embedding.shape:
            raise InvalidMergeEmbeddingError(
                "source and target embedding dimensions differ"
            )
        source_count = self._validated_stored_merge_count(source, "source")
        target_count_after = self._validated_stored_merge_count(target, "target")
        audits = self._conn.execute(
            "SELECT * FROM identity_merge_audit WHERE source_person_id=?",
            (source_id,),
        ).fetchall()
        if len(audits) != 1:
            raise MergeAuditIntegrityError(
                f"source person {source_id!r} does not have exactly one merge audit"
            )
        audit = audits[0]
        if (
            audit["target_person_id"] != target_id
            or bytes(audit["source_embedding"]) != bytes(source["embedding"])
            or audit["source_embedding_count"] != source_count
        ):
            raise MergeAuditIntegrityError(
                f"source person {source_id!r} redirect and audit disagree"
            )
        if target_count_after <= source_count:
            raise MergeAuditIntegrityError(
                f"source person {source_id!r} audit has inconsistent evidence counts"
            )
        return PersonMergeResult(
            source_person_id=source_id,
            target_person_id=target_id,
            audit_id=int(audit["merge_id"]),
            idempotent_replay=True,
            source_embedding_count=source_count,
            target_embedding_count_before=target_count_after,
            target_embedding_count_after=target_count_after,
            staled_suggestion_count=0,
            lineage_member_count_after=self._merge_lineage_member_count(target_id),
        )

    def _before_merge_commit(self) -> None:
        """Failure-injection seam; intentionally performs no work."""

    def _insert_identity_suggestion(
        self,
        *,
        source_person_id: str,
        decision: IdentityDecision,
    ) -> int:
        candidate_person_id = decision.top_candidate_person_id
        similarity = decision.top_similarity
        if candidate_person_id is None or similarity is None:
            raise IdentityPolicyInputError(
                "review_required requires a top candidate"
            )
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO identity_match_suggestions(
                    source_person_id, candidate_person_id, similarity,
                    second_similarity, margin, reason, status, created_at,
                    reviewed_at, reviewed_by
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, NULL, NULL)
                """,
                (
                    source_person_id,
                    candidate_person_id,
                    similarity,
                    decision.second_similarity,
                    decision.margin,
                    decision.reason.value,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
            return int(cursor.lastrowid)
        except sqlite3.IntegrityError:
            existing = self._conn.execute(
                """
                SELECT id FROM identity_match_suggestions
                 WHERE source_person_id=? AND candidate_person_id=?
                   AND status='pending'
                """,
                (source_person_id, candidate_person_id),
            ).fetchone()
            if existing is None:
                raise
            return int(existing["id"])

    @staticmethod
    def _log_identity_decision(result: IdentityRegistrationResult) -> None:
        print(
            "[identity_policy] "
            + json.dumps(
                {
                    "decision": result.decision.value,
                    "reason": result.reason.value,
                    "person_id": result.person_id,
                    "suggestion_id": result.suggestion_id,
                    "top_candidate_person_id": result.top_candidate_person_id,
                    "top_similarity": result.top_similarity,
                    "second_candidate_person_id": result.second_candidate_person_id,
                    "second_similarity": result.second_similarity,
                    "margin": result.margin,
                    "observation_count": result.observation_count,
                    "configuration": result.configuration.as_dict(),
                },
                sort_keys=True,
            )
        )

    @classmethod
    def _validated_identity_embedding(cls, embedding: Any) -> np.ndarray:
        try:
            raw = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise IdentityPolicyInputError(
                "face embedding must be a numeric one-dimensional vector"
            ) from exc
        if raw.ndim != 1 or raw.size == 0:
            raise IdentityPolicyInputError(
                "face embedding must be a non-empty one-dimensional vector"
            )
        if not np.isfinite(raw).all():
            raise IdentityPolicyInputError("face embedding must be finite")
        try:
            return cls._normalize_embedding(raw)
        except ValueError as exc:
            raise IdentityPolicyInputError(str(exc)) from exc

    def _insert_person(
        self,
        person_id: str,
        name: str,
        embedding: np.ndarray,
        embedding_count: int,
        enrolled_at: str,
        cameras: list,
        profile: dict,
    ) -> None:
        best_face_crop = self._best_face_crop(profile)
        self._conn.execute(
            """
            INSERT INTO persons (
                person_id, name, embedding, embedding_count,
                enrolled_at, updated_at, cameras, profile_image, profile_image_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'auto')
            """,
            (
                person_id,
                name,
                embedding.astype(np.float32).tobytes(),
                int(embedding_count),
                enrolled_at,
                enrolled_at,
                json.dumps(cameras or []),
                best_face_crop,
            ),
        )

    def _update_embedding(
        self,
        person_id: str,
        embedding: np.ndarray,
        new_count: int,
        updated_at: str,
        profile: dict,
    ) -> int:
        row = self._conn.execute(
            """
            SELECT embedding, embedding_count, profile_image, profile_image_source
              FROM persons
             WHERE person_id=?
            """,
            (person_id,),
        ).fetchone()
        stored_vec = np.frombuffer(row["embedding"], dtype=np.float32).copy()
        stored_count = int(row["embedding_count"])
        merged = (stored_vec * stored_count + embedding * new_count) / (stored_count + new_count)
        merged = self._normalize_embedding(merged)
        total_count = stored_count + int(new_count)

        new_crop = self._best_face_crop(profile)
        old_crop = row["profile_image"] if row else None
        is_manual = bool(row and row["profile_image_source"] == "manual")
        sharpness = profile.get("face_crop_sharpness") or {}
        new_sharp = self._sharpness_for_path(sharpness, new_crop)
        old_sharp = self._sharpness_for_path(sharpness, old_crop)
        update_image = not is_manual and new_crop is not None and (old_crop is None or new_sharp >= old_sharp)

        if update_image:
            self._conn.execute(
                """
                UPDATE persons
                   SET embedding=?, embedding_count=?, updated_at=?, profile_image=?
                 WHERE person_id=?
                """,
                (
                    merged.astype(np.float32).tobytes(),
                    total_count,
                    updated_at,
                    new_crop,
                    person_id,
                ),
            )
        else:
            self._conn.execute(
                """
                UPDATE persons
                   SET embedding=?, embedding_count=?, updated_at=?
                 WHERE person_id=?
                """,
                (
                    merged.astype(np.float32).tobytes(),
                    total_count,
                    updated_at,
                    person_id,
                ),
            )
        return total_count

    def _insert_gallery_crop(
        self,
        person_id: str,
        crop_type: str,
        path: str,
        sharpness: float,
        session_date: str,
        video_source: str | None,
    ) -> None:
        if not path:
            return
        try:
            stored_path = self._stored_media_path(path, require_exists=True)
            resolved = resolve_media_path(
                stored_path,
                media_root=self.media_root,
                require_exists=True,
            )
        except (MediaPathError, FileNotFoundError, OSError):
            return

        width = None
        height = None
        try:
            import cv2

            img = cv2.imread(str(resolved))
            if img is not None:
                height, width = img.shape[:2]
        except Exception:
            pass

        self._conn.execute(
            """
            INSERT OR IGNORE INTO person_gallery (
                person_id, crop_type, path, sharpness,
                session_date, video_source, width, height
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                crop_type,
                stored_path,
                float(sharpness or 0.0),
                session_date,
                video_source,
                width,
                height,
            ),
        )

    def _prune_gallery(self, person_id: str, crop_type: str, limit: int = 10) -> None:
        rows = self._conn.execute(
            """
            SELECT id FROM person_gallery
             WHERE person_id=? AND crop_type=?
             ORDER BY sharpness DESC, id DESC
            """,
            (person_id, crop_type),
        ).fetchall()
        prune_ids = [row["id"] for row in rows[int(limit):]]
        if not prune_ids:
            return
        placeholders = ",".join("?" for _ in prune_ids)
        self._conn.execute(
            f"DELETE FROM person_gallery WHERE id IN ({placeholders})",
            prune_ids,
        )

    def _remove_missing_gallery_paths(self, person_id: str) -> None:
        rows = self._conn.execute(
            "SELECT id, path FROM person_gallery WHERE person_id=?",
            (person_id,),
        ).fetchall()
        missing = [
            row["id"]
            for row in rows
            if self._public_media_path(row["path"]) is None
        ]
        if not missing:
            return
        placeholders = ",".join("?" for _ in missing)
        self._conn.execute(
            f"DELETE FROM person_gallery WHERE id IN ({placeholders})",
            missing,
        )

    def _upsert_appearance(self, person_id: str, appearance_date: str, profile: dict) -> None:
        appearance = profile.get("appearance") or {}
        color = ((profile.get("appearance_signals") or {}).get("color") or {})
        raw_clothing = [
            appearance.get("top"),
            appearance.get("bottom"),
            appearance.get("shoes"),
            appearance.get("full"),
        ]
        default_status = "ok" if any(raw_clothing) else "not_attempted"
        status = str(appearance.get("clothing_status") or default_status)
        if status not in {"ok", "failed", "not_attempted"}:
            status = "failed"

        def useful(value):
            if value is None:
                return None
            text = str(value).strip()
            if not text or text.lower() in {
                "unknown",
                "unavailable",
                "clothing description unavailable.",
            }:
                return None
            return text

        incoming_clothing = {
            "top": useful(appearance.get("top")),
            "bottom": useful(appearance.get("bottom")),
            "shoes": useful(appearance.get("shoes")),
            "full_description": useful(appearance.get("full")),
        }
        existing = self._conn.execute(
            "SELECT * FROM appearances WHERE person_id=? AND date=?",
            (person_id, appearance_date),
        ).fetchone()
        if status != "ok":
            incoming_clothing = {key: None for key in incoming_clothing}

        def preserved(column: str, incoming):
            if incoming is not None:
                return incoming
            return existing[column] if existing is not None else None

        body_paths = [
            stored
            for raw in (profile.get("best_body_crops") or [])
            if (stored := self._try_stored_media_path(raw)) is not None
        ]
        old_body = self._json_list(existing["best_body_crops"]) if existing is not None else []
        old_videos = self._json_list(existing["video_sources"]) if existing is not None else []
        self._conn.execute(
            """
            INSERT INTO appearances (
                person_id, date, top, bottom, shoes, full_description,
                top_color, bottom_color, clothing_status,
                best_body_crops, video_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, date) DO UPDATE SET
                top=excluded.top,
                bottom=excluded.bottom,
                shoes=excluded.shoes,
                full_description=excluded.full_description,
                top_color=excluded.top_color,
                bottom_color=excluded.bottom_color,
                clothing_status=excluded.clothing_status,
                best_body_crops=excluded.best_body_crops,
                video_sources=excluded.video_sources
            """,
            (
                person_id,
                appearance_date,
                preserved("top", incoming_clothing["top"]),
                preserved("bottom", incoming_clothing["bottom"]),
                preserved("shoes", incoming_clothing["shoes"]),
                preserved("full_description", incoming_clothing["full_description"]),
                preserved("top_color", color.get("top")),
                preserved("bottom_color", color.get("bottom")),
                status,
                json.dumps(self._merge_lists(old_body, body_paths)),
                json.dumps(self._merge_lists(old_videos, profile.get("video_sources") or [])),
            ),
        )

    def _log_event(
        self,
        person_id: str,
        event_type: str,
        similarity: float | None,
        embedding_count_before: int | None,
        embedding_count_after: int,
        video_sources: list[str],
        best_face_crop: str | None = None,
        *,
        strict: bool = False,
    ) -> None:
        try:
            self._conn.execute(
                """
                INSERT INTO recognition_log (
                    person_id, event_type, similarity,
                    embedding_count_before, embedding_count_after,
                    video_sources, best_face_crop, ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person_id,
                    event_type,
                    similarity,
                    embedding_count_before,
                    embedding_count_after,
                    json.dumps(video_sources or []),
                    best_face_crop,
                    datetime.now().isoformat(timespec="seconds"),
                ),
            )
        except Exception as exc:
            if strict:
                raise
            print(f"[WARNING] recognition_log write failed: {exc}")

    def _get_embedding_count(self, person_id: str) -> int:
        row = self._conn.execute(
            "SELECT embedding_count FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        return int(row["embedding_count"]) if row else 1

    def _latest_appearance(self, person_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM appearances WHERE person_id=? ORDER BY date DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return self._appearance_from_row(row, include_stale=True)

    def _appearance_from_row(self, row: sqlite3.Row, include_stale: bool = True) -> dict:
        item = {
            "date": row["date"],
            "top": row["top"],
            "bottom": row["bottom"],
            "shoes": row["shoes"],
            "full_description": row["full_description"],
            "top_color": row["top_color"],
            "bottom_color": row["bottom_color"],
            "clothing_status": (
                row["clothing_status"]
                if "clothing_status" in row.keys()
                else "not_attempted"
            ),
            "best_body_crops": [
                path
                for raw in self._json_list(row["best_body_crops"])
                if (path := self._public_media_path(raw)) is not None
            ],
            "video_sources": self._json_list(row["video_sources"]),
        }
        if include_stale:
            item["is_stale"] = row["date"] != date.today().isoformat()
        return item

    @staticmethod
    def _normalize_embedding(embedding: Any) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise ValueError("face embedding must be non-zero")
        return vec / norm

    def _best_face_crop(self, profile: dict) -> str | None:
        crops = profile.get("face_crops") or []
        sharpness = profile.get("face_crop_sharpness") or {}

        existing: list[str] = []
        for raw in crops:
            if not raw:
                continue
            path = self._try_stored_media_path(raw)
            if path is not None:
                existing.append(path)

        if not existing:
            return None
        if sharpness:
            return max(existing, key=lambda p: self._sharpness_for_path(sharpness, p))
        return existing[0]

    def _try_stored_media_path(self, path: str | Path | None) -> str | None:
        if not path:
            return None
        try:
            return self._stored_media_path(path, require_exists=True)
        except (MediaPathError, FileNotFoundError, OSError):
            return None

    def _stored_media_path(
        self,
        path: str | Path,
        *,
        require_exists: bool,
    ) -> str:
        return normalize_media_path(
            path,
            media_root=self.media_root,
            allow_legacy_absolute=True,
            require_exists=require_exists,
        )

    def _public_media_path(self, path: str | Path | None) -> str | None:
        return self._try_stored_media_path(path)

    def _sharpness_for_path(self, sharpness: dict, path: str | None) -> float:
        if not path:
            return 0.0
        for key, value in sharpness.items():
            try:
                matches = self._stored_media_path(key, require_exists=False) == path
            except MediaPathError:
                matches = str(key) == str(path)
            if matches:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return 0.0
        return 0.0

    @staticmethod
    def _json_list(value: str | None) -> list:
        if not value:
            return []
        try:
            decoded = json.loads(value)
            return decoded if isinstance(decoded, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _merge_lists(left: list, right: list) -> list:
        merged = []
        seen = set()
        for item in [*left, *right]:
            key = str(item)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged
