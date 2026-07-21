from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from psycopg.rows import dict_row   #this is a format of shwing the result of a query as a dict 
from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from . import config

# Single advisory-lock key serializing register()'s find-or-create critical section
# (similarity search + insert/update) across every connection/process talking to
# this database -- the Postgres equivalent of the old single-SQLite-file +
# threading.RLock() + BEGIN IMMEDIATE guarantee, except it also holds across
# multiple worker processes, not just threads in one process. An advisory *xact*
# lock releases automatically at COMMIT/ROLLBACK, matching BEGIN IMMEDIATE's shape.
_REGISTER_LOCK_KEY = 0x57415943  # arbitrary constant ('WAYC'), any int64 works


def _configure_connection(conn: psycopg.Connection) -> None:
    conn.row_factory = dict_row
    conn.autocommit = True
    register_vector(conn)


class GlobalMemory:
    def __init__(
        self,
        dsn: str | None = None,
        *,
        min_size: int | None = None,
        max_size: int | None = None,
        connect_timeout: float = 10.0,
    ) -> None:
        self.dsn = dsn or config.connection_string()
        self._pool = ConnectionPool(
            self.dsn,
            min_size=min_size if min_size is not None else config.POOL_MIN_SIZE,
            max_size=max_size if max_size is not None else config.POOL_MAX_SIZE,
            configure=_configure_connection,
            open=True,
        )
        self._pool.wait(timeout=connect_timeout)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        schema_path = Path(__file__).with_name("schema_postgres.sql")
        with self._pool.connection() as conn:
            conn.execute(schema_path.read_text(encoding="utf-8"))

    # ─── identity ─────────────────────────────────────────────────────────────────

    def register(self, profile: dict) -> str:
        new_vec = self._normalize_embedding(profile["face_embedding"])
        new_count = len(profile.get("face_crops") or []) or 1
        appearance = profile.get("appearance") or {}
        appearance_date = str(appearance.get("date") or date.today().isoformat())
        best_face_crop = self._best_face_crop(profile)

        with self._pool.connection() as conn:
            with conn.transaction():
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (_REGISTER_LOCK_KEY,))
                existing = self._find_existing_person(conn, new_vec)
                if existing is not None:
                    person_id = existing["person_id"]
                    count_before = int(existing["embedding_count"])
                    count_after = self._update_embedding(
                        conn,
                        person_id=person_id,
                        embedding=new_vec,
                        new_count=new_count,
                        updated_at=appearance_date,
                        profile=profile,
                    )
                    self._upsert_appearance(conn, person_id, appearance_date, profile)
                    self._update_gallery(conn, person_id, profile)
                    self._log_event(
                        conn,
                        person_id=person_id,
                        event_type="recognized",
                        similarity=existing["similarity"],
                        embedding_count_before=count_before,
                        embedding_count_after=count_after,
                        video_sources=profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                    )
                else:
                    person_id, name = self._next_person_id(conn)
                    self._insert_person(
                        conn,
                        person_id=person_id,
                        name=name,
                        embedding=new_vec,
                        embedding_count=new_count,
                        enrolled_at=appearance_date,
                        cameras=profile.get("cameras") or [],
                        profile=profile,
                    )
                    self._upsert_appearance(conn, person_id, appearance_date, profile)
                    self._update_gallery(conn, person_id, profile)
                    self._log_event(
                        conn,
                        person_id=person_id,
                        event_type="new_enrollment",
                        similarity=None,
                        embedding_count_before=None,
                        embedding_count_after=new_count,
                        video_sources=profile.get("video_sources") or [],
                        best_face_crop=best_face_crop,
                    )
                return person_id

    def query_by_face(self, embedding, top_k: int = 5, threshold: float | None = None) -> list[dict]:
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else float(threshold)
        query_vec = self._normalize_embedding(embedding)

        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT person_id, name, 1 - (embedding <=> %s) AS similarity
                  FROM persons
                 ORDER BY embedding <=> %s
                 LIMIT %s
                """,
                (query_vec, query_vec, int(top_k)),
            ).fetchall()

            results: list[dict] = []
            for row in rows:
                similarity = float(row["similarity"])
                if similarity < threshold:
                    break
                results.append({
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "similarity": similarity,
                    "appearance": self._latest_appearance(conn, row["person_id"]),
                })
            return results

    def query_by_date(self, date: str) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT p.person_id, p.name, a.*
                  FROM appearances a
                  JOIN persons p ON p.person_id = a.person_id
                 WHERE a.date = %s
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
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT person_id, name, enrolled_at, cameras FROM persons"
            ).fetchall()
            results = []
            for row in rows:
                cameras = row["cameras"] or []
                if camera_id in {str(item) for item in cameras}:
                    results.append({
                        "person_id": row["person_id"],
                        "name": row["name"],
                        "enrolled_at": row["enrolled_at"],
                        "cameras": cameras,
                    })
            return results

    def get_person(self, person_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM persons WHERE person_id=%s",
                (person_id,),
            ).fetchone()
            if row is None:
                return None
            embedding = self._to_numpy(row["embedding"]).tolist()
            return {
                "person_id": row["person_id"],
                "name": row["name"],
                "embedding": embedding,
                "embedding_count": row["embedding_count"],
                "enrolled_at": row["enrolled_at"],
                "updated_at": row["updated_at"],
                "cameras": row["cameras"] or [],
                "profile_image": row["profile_image"],
                "profile_image_source": row["profile_image_source"],
                "latest_appearance": self._latest_appearance(conn, row["person_id"]),
            }

    def list_all(self) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT person_id, name, enrolled_at, updated_at, cameras,
                       profile_image, profile_image_source
                  FROM persons
                 ORDER BY person_id
                """
            ).fetchall()
            return [
                {
                    "person_id": row["person_id"],
                    "name": row["name"],
                    "enrolled_at": row["enrolled_at"],
                    "updated_at": row["updated_at"],
                    "cameras": row["cameras"] or [],
                    "profile_image": row["profile_image"],
                    "profile_image_source": row["profile_image_source"],
                    "latest_appearance": self._latest_appearance(conn, row["person_id"]),
                }
                for row in rows
            ]

    def get_recognition_history(self, person_id: str | None = None, limit: int = 50) -> list[dict]:
        with self._pool.connection() as conn:
            if person_id:
                rows = conn.execute(
                    "SELECT * FROM recognition_log WHERE person_id = %s ORDER BY id DESC LIMIT %s",
                    (person_id, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM recognition_log ORDER BY id DESC LIMIT %s",
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
                    "video_sources": row["video_sources"] or [],
                    "best_face_crop": row["best_face_crop"],
                    "ts": row["ts"],
                }
                for row in rows
            ]

    def rename_person(self, person_id: str, new_name: str) -> None:
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE persons SET name=%s WHERE person_id=%s",
                (new_name, person_id),
            )

    def update_crop_paths(self, person_id: str, profile: dict) -> None:
        appearance = profile.get("appearance") or {}
        appearance_date = str(appearance.get("date") or date.today().isoformat())
        profile_image = self._best_face_crop(profile)

        with self._pool.connection() as conn:
            with conn.transaction():
                person = conn.execute(
                    "SELECT profile_image_source FROM persons WHERE person_id=%s",
                    (person_id,),
                ).fetchone()
                image_source = person["profile_image_source"] if person else "auto"
                if profile_image:
                    if image_source != "manual":
                        conn.execute(
                            "UPDATE persons SET profile_image=%s, profile_image_source='auto' WHERE person_id=%s",
                            (profile_image, person_id),
                        )
                    latest = conn.execute(
                        "SELECT id FROM recognition_log WHERE person_id=%s ORDER BY id DESC LIMIT 1",
                        (person_id,),
                    ).fetchone()
                    if latest is not None:
                        conn.execute(
                            "UPDATE recognition_log SET best_face_crop=%s WHERE id=%s",
                            (profile_image, latest["id"]),
                        )
                self._upsert_appearance(conn, person_id, appearance_date, profile)
                self._remove_missing_gallery_paths(conn, person_id)
                self._update_gallery(conn, person_id, profile)

    def set_profile_image(self, person_id: str, image_path: str, source: str = "auto") -> None:
        source = source if source in {"auto", "manual"} else "auto"
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE persons SET profile_image=%s, profile_image_source=%s WHERE person_id=%s",
                (str(image_path), source, person_id),
            )

    def get_best_face_crop(self, person_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                """
                SELECT path, sharpness FROM person_gallery
                 WHERE person_id=%s AND crop_type='face'
                 ORDER BY sharpness DESC
                 LIMIT 1
                """,
                (person_id,),
            ).fetchone()
            if row is not None and Path(row["path"]).exists():
                return {"path": row["path"], "sharpness": row["sharpness"]}

            row = conn.execute(
                """
                SELECT best_face_crop FROM recognition_log
                 WHERE person_id=%s AND best_face_crop IS NOT NULL
                 ORDER BY ts DESC
                 LIMIT 1
                """,
                (person_id,),
            ).fetchone()
            if row is not None and row["best_face_crop"]:
                return {"path": row["best_face_crop"], "sharpness": 0.0}
            return None

    def update_gallery(self, person_id: str, profile: dict) -> None:
        with self._pool.connection() as conn:
            self._update_gallery(conn, person_id, profile)

    def get_gallery(self, person_id: str, crop_type: str | None = None, limit: int = 10) -> list[dict]:
        limit = max(1, int(limit))
        with self._pool.connection() as conn:
            if crop_type in {"face", "body"}:
                rows = conn.execute(
                    """
                    SELECT * FROM person_gallery
                     WHERE person_id=%s AND crop_type=%s
                     ORDER BY sharpness DESC
                     LIMIT %s
                    """,
                    (person_id, crop_type, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM person_gallery
                     WHERE person_id=%s
                     ORDER BY crop_type, sharpness DESC
                     LIMIT %s
                    """,
                    (person_id, limit),
                ).fetchall()

            return [
                {
                    "id": row["id"],
                    "person_id": row["person_id"],
                    "crop_type": row["crop_type"],
                    "path": row["path"],
                    "sharpness": row["sharpness"],
                    "session_date": row["session_date"],
                    "video_source": row["video_source"],
                    "width": row["width"],
                    "height": row["height"],
                }
                for row in rows
            ]

    # ─── clothing_jobs (async VLM worker queue) ──────────────────────────────────

    def insert_clothing_job(
        self,
        *,
        person_id: str,
        segment_id: str,
        crop_path: str,
        pipeline_version: str | None = None,
    ) -> str | None:
        """Enqueue an async clothing-description job. Fast, non-blocking insert.

        Idempotent on (person_id, segment_id): a duplicate call for the same
        segment returns None instead of raising or enqueueing a second job.
        """
        job_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO clothing_jobs
                        (job_id, person_id, segment_id, crop_path, pipeline_version, status, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, 'pending', %s, %s)
                    """,
                    (
                        job_id,
                        person_id,
                        segment_id,
                        crop_path,
                        pipeline_version or config.CLOTHING_PIPELINE_VERSION,
                        now,
                        now,
                    ),
                )
            except psycopg.errors.UniqueViolation:
                return None
        return job_id

    def claim_next_clothing_job(self) -> dict | None:
        """Atomically pick the oldest eligible job and mark it 'processing'.

        Eligible = status='pending', or status='failed_retryable' with
        next_attempt_at in the past. `FOR UPDATE SKIP LOCKED` lets multiple worker
        processes poll the same table concurrently without blocking on or
        double-claiming a row another worker already grabbed.
        """
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    """
                    SELECT job_id, person_id, segment_id, crop_path, pipeline_version, attempts
                    FROM clothing_jobs
                    WHERE status = %s
                       OR (status = %s AND (next_attempt_at IS NULL OR next_attempt_at <= %s))
                    ORDER BY created_at ASC
                    LIMIT 1
                    FOR UPDATE SKIP LOCKED
                    """,
                    (config.CLOTHING_JOB_STATUS_PENDING, config.CLOTHING_JOB_STATUS_FAILED_RETRYABLE, now),
                ).fetchone()
                if row is None:
                    return None
                conn.execute(
                    "UPDATE clothing_jobs SET status = %s, updated_at = %s WHERE job_id = %s",
                    (config.CLOTHING_JOB_STATUS_PROCESSING, now, row["job_id"]),
                )
                return dict(row)

    def mark_clothing_job_done(self, job_id: str) -> None:
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            conn.execute(
                "UPDATE clothing_jobs SET status = %s, error = NULL, updated_at = %s WHERE job_id = %s",
                (config.CLOTHING_JOB_STATUS_DONE, now, job_id),
            )

    def mark_clothing_job_failed(
        self,
        job_id: str,
        error: str,
        *,
        max_attempts: int | None = None,
        backoff_seconds: tuple[float, ...] | None = None,
    ) -> str:
        """Mark a job failed; retries (with backoff) until max_attempts, then final.

        Returns the status the job was set to (failed_retryable or failed_final).
        """
        max_attempts = config.CLOTHING_JOB_MAX_ATTEMPTS if max_attempts is None else max_attempts
        backoff_seconds = backoff_seconds or config.CLOTHING_JOB_RETRY_BACKOFF_SECONDS
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            with conn.transaction():
                row = conn.execute(
                    "SELECT attempts FROM clothing_jobs WHERE job_id = %s", (job_id,)
                ).fetchone()
                attempts = int(row["attempts"]) + 1 if row is not None else 1

                if attempts >= max_attempts:
                    status = config.CLOTHING_JOB_STATUS_FAILED_FINAL
                    next_attempt_at = None
                else:
                    status = config.CLOTHING_JOB_STATUS_FAILED_RETRYABLE
                    delay = backoff_seconds[min(attempts - 1, len(backoff_seconds) - 1)]
                    next_attempt_at = now + timedelta(seconds=delay)

                conn.execute(
                    """
                    UPDATE clothing_jobs
                    SET status = %s, attempts = %s, error = %s, next_attempt_at = %s, updated_at = %s
                    WHERE job_id = %s
                    """,
                    (status, attempts, str(error)[:2000], next_attempt_at, now, job_id),
                )
                return status

    def get_clothing_job(self, job_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM clothing_jobs WHERE job_id = %s", (job_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def list_clothing_jobs(self, status: str | None = None, limit: int = 100) -> list[dict]:
        with self._pool.connection() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM clothing_jobs WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM clothing_jobs ORDER BY created_at DESC LIMIT %s", (int(limit),)
                ).fetchall()
            return [dict(r) for r in rows]

    def list_clothing_jobs_for_segment(self, segment_id: str) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM clothing_jobs WHERE segment_id = %s ORDER BY created_at DESC",
                (segment_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def latest_clothing_job_for_person(self, person_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM clothing_jobs WHERE person_id = %s ORDER BY created_at DESC LIMIT 1",
                (person_id,),
            ).fetchone()
            return dict(row) if row is not None else None

    def upsert_appearance_for_segment(
        self,
        *,
        person_id: str,
        segment_id: str,
        date: str,
        top: str | None,
        bottom: str | None,
        shoes: str | None,
        full_description: str | None,
        top_color: str | None = None,
        bottom_color: str | None = None,
        best_body_crops: list[str] | None = None,
        video_sources: list[str] | None = None,
    ) -> None:
        """Written by the async VLM worker once a clothing_jobs row completes.

        `appearances` carries two unique constraints: the per-day
        UNIQUE(person_id, date) from GlobalMemory.register()'s synchronous
        placeholder insert, and idx_appearances_person_segment
        UNIQUE(person_id, segment_id) for this method's idempotency. A single
        INSERT ... ON CONFLICT can only target one of them, so this resolves both
        explicitly, same as the SQLite version:
        1. a row already tagged with this segment_id -> update it in place;
        2. else an untagged placeholder row for the same day -> claim it;
        3. else insert, falling back to ON CONFLICT (person_id, date) DO UPDATE if
           a second segment lands on the same day after the first already claimed
           the placeholder (that row keeps only the most recent segment's
           description for that day).
        """
        best_body_crops_json = json.dumps(best_body_crops or [])
        video_sources_json = json.dumps(video_sources or [])
        fields = (top, bottom, shoes, full_description, top_color, bottom_color,
                  best_body_crops_json, video_sources_json)

        with self._pool.connection() as conn:
            with conn.transaction():
                updated = conn.execute(
                    """
                    UPDATE appearances SET date=%s, top=%s, bottom=%s, shoes=%s, full_description=%s,
                        top_color=%s, bottom_color=%s, best_body_crops=%s, video_sources=%s
                    WHERE person_id = %s AND segment_id = %s
                    """,
                    (date, *fields, person_id, segment_id),
                )
                if updated.rowcount > 0:
                    return

                claimed = conn.execute(
                    """
                    UPDATE appearances SET segment_id=%s, top=%s, bottom=%s, shoes=%s, full_description=%s,
                        top_color=%s, bottom_color=%s, best_body_crops=%s, video_sources=%s
                    WHERE person_id = %s AND date = %s AND segment_id IS NULL
                    """,
                    (segment_id, *fields, person_id, date),
                )
                if claimed.rowcount > 0:
                    return

                conn.execute(
                    """
                    INSERT INTO appearances (
                        person_id, date, segment_id, top, bottom, shoes, full_description,
                        top_color, bottom_color, best_body_crops, video_sources
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (person_id, date) DO UPDATE SET
                        segment_id = EXCLUDED.segment_id,
                        top = EXCLUDED.top,
                        bottom = EXCLUDED.bottom,
                        shoes = EXCLUDED.shoes,
                        full_description = EXCLUDED.full_description,
                        top_color = EXCLUDED.top_color,
                        bottom_color = EXCLUDED.bottom_color,
                        best_body_crops = EXCLUDED.best_body_crops,
                        video_sources = EXCLUDED.video_sources
                    """,
                    (person_id, date, segment_id, *fields),
                )

    # ─── segments (reliability state machine) ────────────────────────────────────

    def upsert_segment(
        self,
        *,
        segment_id: str,
        seq_num: int = 0,
        codec: str | None = None,
        pipeline_version: str | None = None,
        segment_start_ts: str | None = None,
        segment_end_ts: str | None = None,
        status: str | None = None,
        segment_incomplete: bool = False,
    ) -> str:
        """Insert or update a segment row. Returns the segment_id actually stored,
        which may differ from the `segment_id` argument: idempotency is keyed on
        (segment_start_ts, pipeline_version) -- no camera_id needed at single-camera
        scope -- so a crash-recovery replay of the same capture window (a new UUID,
        same start time + pipeline version) collapses onto the existing row instead
        of creating a duplicate logical segment. `segment_id` still needs to be a
        real value up front since it's the surrogate PK clothing_jobs/camera_events
        reference, but callers should use the *returned* id afterwards.
        """
        status = status or config.SEGMENT_STATUS_CAPTURING
        pipeline_version = pipeline_version or config.PIPELINE_VERSION
        now = datetime.now(timezone.utc)
        start_ts = segment_start_ts or now

        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT segment_id FROM segments WHERE segment_start_ts = %s AND pipeline_version = %s",
                (start_ts, pipeline_version),
            ).fetchone()
            effective_id = row["segment_id"] if row is not None else segment_id

            upsert_sql = """
                INSERT INTO segments (
                    segment_id, seq_num, codec, pipeline_version, segment_start_ts, segment_end_ts,
                    status, segment_incomplete, created_at, updated_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (segment_id) DO UPDATE SET
                    seq_num = EXCLUDED.seq_num,
                    codec = COALESCE(EXCLUDED.codec, segments.codec),
                    segment_end_ts = COALESCE(EXCLUDED.segment_end_ts, segments.segment_end_ts),
                    status = EXCLUDED.status,
                    segment_incomplete = EXCLUDED.segment_incomplete,
                    updated_at = EXCLUDED.updated_at
            """
            params = (
                effective_id, seq_num, codec, pipeline_version, start_ts, segment_end_ts,
                status, bool(segment_incomplete), now, now,
            )
            try:
                with conn.transaction():
                    conn.execute(upsert_sql, params)
            except psycopg.errors.UniqueViolation:
                # Lost a race against a concurrent insert for the same
                # (segment_start_ts, pipeline_version) -- reuse the winner's id.
                row = conn.execute(
                    "SELECT segment_id FROM segments WHERE segment_start_ts = %s AND pipeline_version = %s",
                    (start_ts, pipeline_version),
                ).fetchone()
                effective_id = row["segment_id"]
        return effective_id

    def set_segment_status(
        self,
        segment_id: str,
        status: str,
        *,
        error: str | None = None,
        increment_retry: bool = False,
    ) -> None:
        now = datetime.now(timezone.utc)
        with self._pool.connection() as conn:
            if increment_retry:
                conn.execute(
                    """
                    UPDATE segments
                    SET status = %s, error = %s, retry_count = retry_count + 1, updated_at = %s
                    WHERE segment_id = %s
                    """,
                    (status, error, now, segment_id),
                )
            else:
                conn.execute(
                    "UPDATE segments SET status = %s, error = %s, updated_at = %s WHERE segment_id = %s",
                    (status, error, now, segment_id),
                )

    def get_segment(self, segment_id: str) -> dict | None:
        with self._pool.connection() as conn:
            row = conn.execute(
                "SELECT * FROM segments WHERE segment_id = %s", (segment_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def list_segments(self, status: str | None = None, limit: int = 50) -> list[dict]:
        with self._pool.connection() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM segments WHERE status = %s ORDER BY segment_start_ts DESC LIMIT %s",
                    (status, int(limit)),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM segments ORDER BY segment_start_ts DESC LIMIT %s", (int(limit),)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_person_pipeline_status(self, person_id: str) -> dict:
        """Combined monitoring view for one person -- core detection and clothing
        enrichment reported as two separate fields, never conflated (SUCCEEDED on a
        segment means core detection only; clothing enrichment may still be
        pending independently).
        """
        clothing_job = self.latest_clothing_job_for_person(person_id)
        segment = self.get_segment(clothing_job["segment_id"]) if clothing_job else None
        return {
            "person_id": person_id,
            "core_detection": {
                "segment_id": segment["segment_id"] if segment else None,
                "status": segment["status"] if segment else None,
            },
            "clothing_enrichment": {
                "job_id": clothing_job["job_id"] if clothing_job else None,
                "segment_id": clothing_job["segment_id"] if clothing_job else None,
                "status": clothing_job["status"] if clothing_job else None,
                "attempts": clothing_job["attempts"] if clothing_job else None,
                "error": clothing_job["error"] if clothing_job else None,
            },
        }

    # ─── camera_events ────────────────────────────────────────────────────────────

    def log_camera_event(self, event_type: str, reason: str | None = None) -> None:
        """connected / disconnected / reconnect_attempt / reconnected_at, per the
        outage log described in the realtime architecture. Not yet called by any
        producer (gst_stream.py's reconnect callbacks don't reach GlobalMemory) --
        exposed here so that wiring is a one-line addition when it happens.
        """
        with self._pool.connection() as conn:
            conn.execute(
                "INSERT INTO camera_events (event_type, ts, reason) VALUES (%s, %s, %s)",
                (event_type, datetime.now(timezone.utc), reason),
            )

    def list_camera_events(self, limit: int = 100) -> list[dict]:
        with self._pool.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM camera_events ORDER BY ts DESC LIMIT %s", (int(limit),)
            ).fetchall()
            return [dict(r) for r in rows]

    # ─── lifecycle ────────────────────────────────────────────────────────────────

    def close(self) -> None:
        self._pool.close(timeout=10.0)

    def __enter__(self) -> GlobalMemory:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ─── internals (all take an explicit `conn` -- no shared self._conn with a
    # connection pool, unlike the old single-SQLite-file version) ─────────────────

    def _next_person_id(self, conn: psycopg.Connection) -> tuple[str, str]:
        row = conn.execute(
            "UPDATE counters SET value = value + 1 WHERE key = 'person_count' RETURNING value"
        ).fetchone()
        n = int(row["value"])
        person_id = f"person_{n:03d}"
        name = f"Person {n:03d}"

        # Defensive guard for pre-migration/backfilled databases: never collide
        # with an existing ID even if the counter was not seeded correctly.
        while conn.execute("SELECT 1 FROM persons WHERE person_id=%s", (person_id,)).fetchone():
            row = conn.execute(
                "UPDATE counters SET value = value + 1 WHERE key = 'person_count' RETURNING value"
            ).fetchone()
            n = int(row["value"])
            person_id = f"person_{n:03d}"
            name = f"Person {n:03d}"
        return person_id, name

    def _find_existing_person(
        self, conn: psycopg.Connection, embedding: np.ndarray, threshold: float | None = None
    ) -> dict | None:
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else float(threshold)
        row = conn.execute(
            """
            SELECT person_id, name, embedding_count, 1 - (embedding <=> %s) AS similarity
              FROM persons
             ORDER BY embedding <=> %s
             LIMIT 1
            """,
            (embedding, embedding),
        ).fetchone()
        if row is None:
            return None
        similarity = float(row["similarity"])
        if similarity < threshold:
            return None
        return {
            "person_id": row["person_id"],
            "name": row["name"],
            "similarity": similarity,
            "embedding_count": row["embedding_count"],
        }

    def _insert_person(
        self,
        conn: psycopg.Connection,
        person_id: str,
        name: str,
        embedding: np.ndarray,
        embedding_count: int,
        enrolled_at: str,
        cameras: list,
        profile: dict,
    ) -> None:
        best_face_crop = self._best_face_crop(profile)
        conn.execute(
            """
            INSERT INTO persons (
                person_id, name, embedding, embedding_count,
                enrolled_at, updated_at, cameras, profile_image, profile_image_source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'auto')
            """,
            (
                person_id,
                name,
                embedding.astype(np.float32),
                int(embedding_count),
                enrolled_at,
                enrolled_at,
                json.dumps(cameras or []),
                best_face_crop,
            ),
        )

    def _update_embedding(
        self,
        conn: psycopg.Connection,
        person_id: str,
        embedding: np.ndarray,
        new_count: int,
        updated_at: str,
        profile: dict,
    ) -> int:
        row = conn.execute(
            "SELECT embedding, embedding_count, profile_image, profile_image_source FROM persons WHERE person_id=%s",
            (person_id,),
        ).fetchone()
        stored_vec = self._to_numpy(row["embedding"])
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
            conn.execute(
                "UPDATE persons SET embedding=%s, embedding_count=%s, updated_at=%s, profile_image=%s WHERE person_id=%s",
                (merged.astype(np.float32), total_count, updated_at, new_crop, person_id),
            )
        else:
            conn.execute(
                "UPDATE persons SET embedding=%s, embedding_count=%s, updated_at=%s WHERE person_id=%s",
                (merged.astype(np.float32), total_count, updated_at, person_id),
            )
        return total_count

    def _update_gallery(self, conn: psycopg.Connection, person_id: str, profile: dict) -> None:
        session_date = str((profile.get("appearance") or {}).get("date") or date.today().isoformat())
        video_sources = profile.get("video_sources") or []
        video_source = video_sources[0] if video_sources else None
        face_sharpness = profile.get("face_crop_sharpness") or {}
        body_sharpness = profile.get("body_crop_sharpness") or {}

        for path in profile.get("face_crops") or []:
            self._insert_gallery_crop(
                conn,
                person_id=person_id,
                crop_type="face",
                path=path,
                sharpness=self._sharpness_for_path(face_sharpness, path),
                session_date=session_date,
                video_source=video_source,
            )

        for path in profile.get("best_body_crops") or []:
            self._insert_gallery_crop(
                conn,
                person_id=person_id,
                crop_type="body",
                path=path,
                sharpness=self._sharpness_for_path(body_sharpness, path),
                session_date=session_date,
                video_source=video_source,
            )

        for crop_type in ("face", "body"):
            self._prune_gallery(conn, person_id, crop_type, limit=10)

    def _insert_gallery_crop(
        self,
        conn: psycopg.Connection,
        person_id: str,
        crop_type: str,
        path: str,
        sharpness: float,
        session_date: str,
        video_source: str | None,
    ) -> None:
        if not path:
            return
        resolved = Path(str(path))
        if not resolved.exists():
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

        conn.execute(
            """
            INSERT INTO person_gallery (
                person_id, crop_type, path, sharpness,
                session_date, video_source, width, height
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (person_id, crop_type, path) DO NOTHING
            """,
            (
                person_id,
                crop_type,
                str(resolved.resolve()),
                float(sharpness or 0.0),
                session_date,
                video_source,
                width,
                height,
            ),
        )

    def _prune_gallery(self, conn: psycopg.Connection, person_id: str, crop_type: str, limit: int = 10) -> None:
        rows = conn.execute(
            """
            SELECT id FROM person_gallery
             WHERE person_id=%s AND crop_type=%s
             ORDER BY sharpness DESC, id DESC
            """,
            (person_id, crop_type),
        ).fetchall()
        prune_ids = [row["id"] for row in rows[int(limit):]]
        if not prune_ids:
            return
        conn.execute("DELETE FROM person_gallery WHERE id = ANY(%s)", (prune_ids,))

    def _remove_missing_gallery_paths(self, conn: psycopg.Connection, person_id: str) -> None:
        rows = conn.execute(
            "SELECT id, path FROM person_gallery WHERE person_id=%s",
            (person_id,),
        ).fetchall()
        missing = [row["id"] for row in rows if not Path(row["path"]).exists()]
        if not missing:
            return
        conn.execute("DELETE FROM person_gallery WHERE id = ANY(%s)", (missing,))

    def _upsert_appearance(self, conn: psycopg.Connection, person_id: str, appearance_date: str, profile: dict) -> None:
        appearance = profile.get("appearance") or {}
        clothing_fields = [appearance.get("top"), appearance.get("bottom"), appearance.get("shoes")]
        if not any(clothing_fields):
            return

        color = ((profile.get("appearance_signals") or {}).get("color") or {})
        conn.execute(
            """
            INSERT INTO appearances (
                person_id, date, top, bottom, shoes, full_description,
                top_color, bottom_color, best_body_crops, video_sources
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (person_id, date) DO UPDATE SET
                top = EXCLUDED.top,
                bottom = EXCLUDED.bottom,
                shoes = EXCLUDED.shoes,
                full_description = EXCLUDED.full_description,
                top_color = EXCLUDED.top_color,
                bottom_color = EXCLUDED.bottom_color,
                best_body_crops = EXCLUDED.best_body_crops,
                video_sources = EXCLUDED.video_sources
            """,
            (
                person_id,
                appearance_date,
                appearance.get("top"),
                appearance.get("bottom"),
                appearance.get("shoes"),
                appearance.get("full"),
                color.get("top"),
                color.get("bottom"),
                json.dumps(profile.get("best_body_crops") or []),
                json.dumps(profile.get("video_sources") or []),
            ),
        )

    def _log_event(
        self,
        conn: psycopg.Connection,
        person_id: str,
        event_type: str,
        similarity: float | None,
        embedding_count_before: int | None,
        embedding_count_after: int,
        video_sources: list[str],
        best_face_crop: str | None = None,
    ) -> None:
        try:
            conn.execute(
                """
                INSERT INTO recognition_log (
                    person_id, event_type, similarity,
                    embedding_count_before, embedding_count_after,
                    video_sources, best_face_crop, ts
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    person_id,
                    event_type,
                    similarity,
                    embedding_count_before,
                    embedding_count_after,
                    json.dumps(video_sources or []),
                    best_face_crop,
                    datetime.now(timezone.utc),
                ),
            )
        except Exception as exc:
            print(f"[WARNING] recognition_log write failed: {exc}")

    def _latest_appearance(self, conn: psycopg.Connection, person_id: str) -> dict | None:
        row = conn.execute(
            "SELECT * FROM appearances WHERE person_id=%s ORDER BY date DESC LIMIT 1",
            (person_id,),
        ).fetchone()
        if row is None:
            return None
        return self._appearance_from_row(row, include_stale=True)

    def _appearance_from_row(self, row: dict, include_stale: bool = True) -> dict:
        item = {
            "date": row["date"],
            "top": row["top"],
            "bottom": row["bottom"],
            "shoes": row["shoes"],
            "full_description": row["full_description"],
            "top_color": row["top_color"],
            "bottom_color": row["bottom_color"],
            "best_body_crops": row["best_body_crops"] or [],
            "video_sources": row["video_sources"] or [],
        }
        if include_stale:
            row_date = row["date"]
            row_date_iso = row_date.isoformat() if hasattr(row_date, "isoformat") else str(row_date)
            item["is_stale"] = row_date_iso != date.today().isoformat()
        return item

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        """pgvector.psycopg's register_vector() returns `vector` columns as a
        pgvector.Vector wrapper, not a bare numpy array/list -- unwrap it here so
        callers can treat an embedding read back from the DB the same way as one
        that never left Python.
        """
        if hasattr(value, "to_numpy"):
            return value.to_numpy().astype(np.float32)
        return np.asarray(value, dtype=np.float32)

    @staticmethod
    def _normalize_embedding(embedding: Any) -> np.ndarray:
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            raise ValueError("face embedding must be non-zero")
        return vec / norm

    @staticmethod
    def _best_face_crop(profile: dict) -> str | None:
        crops = profile.get("face_crops") or []
        sharpness = profile.get("face_crop_sharpness") or {}

        existing: list[str] = []
        for raw in crops:
            if not raw:
                continue
            path = Path(str(raw))
            if path.exists():
                existing.append(str(path.resolve()))

        if not existing:
            return None
        if sharpness:
            return max(existing, key=lambda p: GlobalMemory._sharpness_for_path(sharpness, p))
        return existing[0]

    @staticmethod
    def _sharpness_for_path(sharpness: dict, path: str | None) -> float:
        if not path:
            return 0.0
        variants = [
            path,
            str(Path(path)),
            str(Path(path).resolve()) if Path(path).exists() else path,
        ]
        for key in variants:
            if key in sharpness:
                try:
                    return float(sharpness[key])
                except (TypeError, ValueError):
                    return 0.0
        return 0.0
