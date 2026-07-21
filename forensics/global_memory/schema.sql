CREATE TABLE IF NOT EXISTS persons (
    person_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    embedding        BLOB NOT NULL,
    embedding_count  INTEGER NOT NULL DEFAULT 1,
    enrolled_at      TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    cameras          TEXT NOT NULL DEFAULT '[]',
    profile_image    TEXT DEFAULT NULL,
    profile_image_source TEXT NOT NULL DEFAULT 'auto'
);

CREATE TABLE IF NOT EXISTS appearances (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id        TEXT NOT NULL REFERENCES persons(person_id),
    date             TEXT NOT NULL,
    segment_id       TEXT DEFAULT NULL,
    top              TEXT,
    bottom           TEXT,
    shoes            TEXT,
    full_description TEXT,
    top_color        TEXT,
    bottom_color     TEXT,
    best_body_crops  TEXT NOT NULL DEFAULT '[]',
    video_sources    TEXT NOT NULL DEFAULT '[]',
    UNIQUE(person_id, date)
);

CREATE TABLE IF NOT EXISTS recognition_log (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id              TEXT NOT NULL,
    event_type             TEXT NOT NULL,
    similarity             REAL,
    embedding_count_before INTEGER,
    embedding_count_after  INTEGER NOT NULL,
    video_sources          TEXT NOT NULL DEFAULT '[]',
    best_face_crop         TEXT DEFAULT NULL,
    ts                     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person_gallery (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id    TEXT NOT NULL REFERENCES persons(person_id),
    crop_type    TEXT NOT NULL,
    path         TEXT NOT NULL,
    sharpness    REAL NOT NULL DEFAULT 0.0,
    session_date TEXT NOT NULL,
    video_source TEXT,
    width        INTEGER,
    height       INTEGER,
    UNIQUE(person_id, crop_type, path)
);

CREATE TABLE IF NOT EXISTS counters (
    key    TEXT PRIMARY KEY,
    value  INTEGER NOT NULL DEFAULT 0
);

INSERT OR IGNORE INTO counters (key, value) VALUES ('person_count', 0);

-- Async clothing-description queue. finalize() inserts a 'pending' row per
-- resolved person_id right after GlobalMemory.register() -- a fast, non-blocking
-- INSERT, no VLM call on the synchronous path. A separate worker (not
-- implemented yet) polls for status='pending', runs the VLM, and writes results
-- into `appearances`, then marks the row 'done' / 'failed_retryable' / 'failed_final'.
-- UNIQUE(person_id, segment_id) makes the insert idempotent: re-processing the same
-- segment (e.g. after a crash-recovery replay) never enqueues a duplicate VLM job.
CREATE TABLE IF NOT EXISTS clothing_jobs (
    job_id           TEXT PRIMARY KEY,
    person_id        TEXT NOT NULL REFERENCES persons(person_id),
    segment_id       TEXT NOT NULL,
    crop_path        TEXT NOT NULL,
    pipeline_version TEXT NOT NULL,
    status           TEXT NOT NULL DEFAULT 'pending',
    attempts         INTEGER NOT NULL DEFAULT 0,
    next_attempt_at  TEXT DEFAULT NULL,
    error            TEXT DEFAULT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    UNIQUE(person_id, segment_id)
);

CREATE INDEX IF NOT EXISTS idx_clothing_jobs_status ON clothing_jobs(status);
CREATE INDEX IF NOT EXISTS idx_clothing_jobs_person ON clothing_jobs(person_id);
CREATE INDEX IF NOT EXISTS idx_clothing_jobs_next_attempt ON clothing_jobs(next_attempt_at);

-- Segment reliability state machine: CAPTURING -> READY -> PROCESSING -> SUCCEEDED
-- / FAILED_RETRYABLE (retried, loops back to PROCESSING) / FAILED_FINAL. CAPTURING
-- and READY are set by the segment accumulator (presence_segmentation.py) as frames
-- are gathered and the segment closes; PROCESSING/SUCCEEDED/FAILED_* are set by
-- whoever runs the sync detection graph against the segment (service.py).
-- `status` here reflects core detection/identity completion ONLY -- clothing
-- enrichment status lives in clothing_jobs/appearances and is never conflated with
-- this table (see the /api/segments monitoring endpoints in service.py).
CREATE TABLE IF NOT EXISTS segments (
    segment_id         TEXT PRIMARY KEY,
    seq_num            INTEGER NOT NULL DEFAULT 0,
    codec              TEXT,
    segment_start_ts   TEXT NOT NULL,
    segment_end_ts     TEXT,
    status             TEXT NOT NULL DEFAULT 'CAPTURING',
    retry_count        INTEGER NOT NULL DEFAULT 0,
    segment_incomplete INTEGER NOT NULL DEFAULT 0,
    error              TEXT DEFAULT NULL,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_status ON segments(status);

-- Per-segment idempotency for appearances, in addition to the existing per-day
-- (person_id, date) uniqueness used by GlobalMemory.register()'s immediate
-- placeholder row. The async VLM worker upserts on (person_id, segment_id) instead
-- via GlobalMemory.upsert_appearance_for_segment() -- a person seen in two segments
-- on the same day gets two appearance rows (they may be wearing different clothes),
-- and re-processing the same segment (crash-recovery replay) overwrites in place
-- rather than duplicating.
CREATE UNIQUE INDEX IF NOT EXISTS idx_appearances_person_segment
    ON appearances(person_id, segment_id);

CREATE INDEX IF NOT EXISTS idx_appearances_date     ON appearances(date);
CREATE INDEX IF NOT EXISTS idx_appearances_person   ON appearances(person_id);
CREATE INDEX IF NOT EXISTS idx_log_person           ON recognition_log(person_id);
CREATE INDEX IF NOT EXISTS idx_log_ts               ON recognition_log(ts);
CREATE INDEX IF NOT EXISTS idx_log_event            ON recognition_log(event_type);
CREATE INDEX IF NOT EXISTS idx_gallery_person       ON person_gallery(person_id);
CREATE INDEX IF NOT EXISTS idx_gallery_type         ON person_gallery(crop_type);
CREATE INDEX IF NOT EXISTS idx_gallery_sharpness    ON person_gallery(person_id, crop_type, sharpness DESC);
