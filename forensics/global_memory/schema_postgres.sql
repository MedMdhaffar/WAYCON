-- Postgres schema for GlobalMemory (replaces schema.sql / SQLite for the realtime
-- deployment). schema.sql is kept around only as the read source for
-- migrate_sqlite_to_postgres.py's one-time backfill -- GlobalMemory itself no longer
-- targets SQLite.
--
-- Design notes (see the migration write-up for the full rationale):
--   * TEXT timestamps -> TIMESTAMPTZ (idiomatic, and a prerequisite for the
--     time-range partitioning called out as future work on recognition_log /
--     camera_events -- not applied yet, "once volume justifies it").
--   * JSON-as-TEXT columns (cameras, best_body_crops, video_sources, color samples)
--     -> JSONB.
--   * persons.embedding: BLOB -> vector(512) (pgvector), HNSW index with
--     vector_cosine_ops. Embeddings are L2-normalized before storage, so cosine
--     distance (<=>) is equivalent to the old dot-product similarity score;
--     similarity = 1 - (embedding <=> query).
--   * segments gets a `pipeline_version` column (didn't exist in the SQLite
--     version) and a UNIQUE(segment_start_ts, pipeline_version) idempotency key,
--     in addition to the surrogate `segment_id` PK used by clothing_jobs' FK.
--   * New camera_events table (event_type, ts, reason) -- not wired to any
--     producer yet (gst_stream.py's reconnect callbacks don't call GlobalMemory);
--     that wiring is separate follow-up scope.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS persons (
    person_id             TEXT PRIMARY KEY,
    name                  TEXT NOT NULL,
    embedding             vector(512) NOT NULL,
    embedding_count       INTEGER NOT NULL DEFAULT 1,
    enrolled_at           TIMESTAMPTZ NOT NULL,
    updated_at            TIMESTAMPTZ NOT NULL,
    cameras               JSONB NOT NULL DEFAULT '[]'::jsonb,
    profile_image         TEXT DEFAULT NULL,
    profile_image_source  TEXT NOT NULL DEFAULT 'auto'
);

-- HNSW over cosine distance. No `lists`-style tuning needed (unlike IVFFlat) and no
-- retraining as the table grows -- appropriate for a table that starts empty and
-- grows continuously rather than being bulk-loaded once at a known size.
CREATE INDEX IF NOT EXISTS idx_persons_embedding_hnsw
    ON persons USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS appearances (
    id                INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    person_id         TEXT NOT NULL REFERENCES persons(person_id),
    date              DATE NOT NULL,
    segment_id        TEXT DEFAULT NULL,
    top               TEXT,
    bottom            TEXT,
    shoes             TEXT,
    full_description  TEXT,
    top_color         TEXT,
    bottom_color      TEXT,
    best_body_crops   JSONB NOT NULL DEFAULT '[]'::jsonb,
    video_sources     JSONB NOT NULL DEFAULT '[]'::jsonb,
    UNIQUE(person_id, date)
);

-- Per-segment idempotency, separate from the legacy per-day uniqueness above (see
-- store.py::upsert_appearance_for_segment for how both are reconciled). NULLS are
-- distinct in a unique index, so legacy/placeholder rows with segment_id IS NULL
-- never collide with each other or with tagged rows.
CREATE UNIQUE INDEX IF NOT EXISTS idx_appearances_person_segment
    ON appearances(person_id, segment_id);

CREATE INDEX IF NOT EXISTS idx_appearances_date   ON appearances(date);
CREATE INDEX IF NOT EXISTS idx_appearances_person ON appearances(person_id);

CREATE TABLE IF NOT EXISTS recognition_log (
    id                      INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    person_id               TEXT NOT NULL,
    event_type              TEXT NOT NULL,
    similarity              REAL,
    embedding_count_before  INTEGER,
    embedding_count_after   INTEGER NOT NULL,
    video_sources           JSONB NOT NULL DEFAULT '[]'::jsonb,
    best_face_crop          TEXT DEFAULT NULL,
    ts                      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_log_person ON recognition_log(person_id);
CREATE INDEX IF NOT EXISTS idx_log_ts     ON recognition_log(ts);
CREATE INDEX IF NOT EXISTS idx_log_event  ON recognition_log(event_type);

CREATE TABLE IF NOT EXISTS person_gallery (
    id            INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    person_id     TEXT NOT NULL REFERENCES persons(person_id),
    crop_type     TEXT NOT NULL,
    path          TEXT NOT NULL,
    sharpness     REAL NOT NULL DEFAULT 0.0,
    session_date  DATE NOT NULL,
    video_source  TEXT,
    width         INTEGER,
    height        INTEGER,
    UNIQUE(person_id, crop_type, path)
);

CREATE INDEX IF NOT EXISTS idx_gallery_person     ON person_gallery(person_id);
CREATE INDEX IF NOT EXISTS idx_gallery_type       ON person_gallery(crop_type);
CREATE INDEX IF NOT EXISTS idx_gallery_sharpness  ON person_gallery(person_id, crop_type, sharpness DESC);

CREATE TABLE IF NOT EXISTS counters (
    key    TEXT PRIMARY KEY,
    value  INTEGER NOT NULL DEFAULT 0
);

INSERT INTO counters (key, value) VALUES ('person_count', 0)
    ON CONFLICT (key) DO NOTHING;

-- Async clothing-description queue -- see vlm_worker.py. Idempotent on
-- (person_id, segment_id): re-enqueuing for the same segment is a silent no-op.
CREATE TABLE IF NOT EXISTS clothing_jobs (
    job_id            TEXT PRIMARY KEY,
    person_id         TEXT NOT NULL REFERENCES persons(person_id),
    segment_id        TEXT NOT NULL,
    crop_path         TEXT NOT NULL,
    pipeline_version  TEXT NOT NULL,
    status            TEXT NOT NULL DEFAULT 'pending',
    attempts          INTEGER NOT NULL DEFAULT 0,
    next_attempt_at   TIMESTAMPTZ DEFAULT NULL,
    error             TEXT DEFAULT NULL,
    created_at        TIMESTAMPTZ NOT NULL,
    updated_at        TIMESTAMPTZ NOT NULL,
    UNIQUE(person_id, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_clothing_jobs_status       ON clothing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_clothing_jobs_person       ON clothing_jobs(person_id);
CREATE INDEX IF NOT EXISTS idx_clothing_jobs_next_attempt ON clothing_jobs(next_attempt_at);

-- Segment reliability state machine: CAPTURING -> READY -> PROCESSING -> SUCCEEDED
-- / FAILED_RETRYABLE (retried, loops back to PROCESSING) / FAILED_FINAL.
-- `status` reflects core detection/identity completion ONLY -- clothing enrichment
-- status lives in clothing_jobs/appearances and is never conflated with this table
-- (see the /api/segments* monitoring endpoints in person_creation/service.py).
--
-- segment_id (UUID, generated by the ingestion side) stays the surrogate PK so
-- clothing_jobs/camera_events FKs stay simple. The idempotency key requested for
-- this migration -- (segment_start_ts, pipeline_version), no camera_id needed at
-- single-camera scope -- is enforced separately below: a crash-recovery replay of
-- the same capture window (new segment_id, same start time + pipeline version)
-- collapses onto the same logical segment instead of creating a duplicate.
CREATE TABLE IF NOT EXISTS segments (
    segment_id          TEXT PRIMARY KEY,
    seq_num             INTEGER NOT NULL DEFAULT 0,
    codec               TEXT,
    pipeline_version    TEXT NOT NULL DEFAULT 'unknown',
    segment_start_ts    TIMESTAMPTZ NOT NULL,
    segment_end_ts      TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'CAPTURING',
    retry_count         INTEGER NOT NULL DEFAULT 0,
    segment_incomplete  BOOLEAN NOT NULL DEFAULT FALSE,
    error               TEXT DEFAULT NULL,
    created_at          TIMESTAMPTZ NOT NULL,
    updated_at          TIMESTAMPTZ NOT NULL,
    UNIQUE(segment_start_ts, pipeline_version)
);

CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);

-- camera_events: connected / disconnected / reconnect_attempt / reconnected_at, per
-- the realtime architecture's outage log. Not yet written by any producer -- see
-- this file's module docstring.
CREATE TABLE IF NOT EXISTS camera_events (
    id          INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_type  TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    reason      TEXT
);

CREATE INDEX IF NOT EXISTS idx_camera_events_ts   ON camera_events(ts);
CREATE INDEX IF NOT EXISTS idx_camera_events_type ON camera_events(event_type);
