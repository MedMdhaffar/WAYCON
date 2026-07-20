from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
    IdentityPolicyConfig,
    IdentityPolicyInputError,
    IdentityRegistrationResult,
    evaluate_identity_decision,
)


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
        appearance = profile.get("appearance") or {}
        appearance_date = str(appearance.get("date") or date.today().isoformat())
        best_face_crop = self._best_face_crop(profile)

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
                    count_before = self._get_embedding_count(person_id)
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
                        similarity=identity_decision.top_similarity,
                        embedding_count_before=count_before,
                        embedding_count_after=count_after,
                        video_sources=profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                        strict=True,
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
                        strict=True,
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
        self._ensure_identity_merge_audit_schema()

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
