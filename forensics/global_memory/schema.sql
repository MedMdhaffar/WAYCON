CREATE TABLE IF NOT EXISTS persons (
    person_id        TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    embedding        BLOB NOT NULL,
    embedding_count  INTEGER NOT NULL DEFAULT 1,
    enrolled_at      TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    cameras          TEXT NOT NULL DEFAULT '[]',
    profile_image    TEXT DEFAULT NULL,
    profile_image_source TEXT NOT NULL DEFAULT 'auto',
    is_active        INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    merged_into_person_id TEXT DEFAULT NULL
);

CREATE TABLE IF NOT EXISTS appearances (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id        TEXT NOT NULL REFERENCES persons(person_id),
    date             TEXT NOT NULL,
    top              TEXT,
    bottom           TEXT,
    shoes            TEXT,
    full_description TEXT,
    top_color        TEXT,
    bottom_color     TEXT,
    clothing_status  TEXT NOT NULL DEFAULT 'not_attempted',
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

CREATE TABLE IF NOT EXISTS identity_match_suggestions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_person_id    TEXT NOT NULL REFERENCES persons(person_id),
    candidate_person_id TEXT NOT NULL REFERENCES persons(person_id),
    similarity          REAL NOT NULL,
    second_similarity   REAL,
    margin              REAL,
    reason              TEXT,
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'accepted', 'rejected', 'stale')),
    created_at          TEXT NOT NULL,
    reviewed_at         TEXT,
    reviewed_by         TEXT,
    CHECK (source_person_id <> candidate_person_id)
);

CREATE TABLE IF NOT EXISTS identity_merge_audit (
    merge_id               INTEGER PRIMARY KEY AUTOINCREMENT,
    source_person_id       TEXT NOT NULL,
    target_person_id       TEXT NOT NULL,
    reason                 TEXT,
    decision_source        TEXT NOT NULL,
    similarity             REAL,
    source_embedding       BLOB NOT NULL,
    source_embedding_count INTEGER NOT NULL
                           CHECK (source_embedding_count > 0),
    source_name            TEXT,
    target_name_before     TEXT,
    created_at             TEXT NOT NULL,
    CHECK (source_person_id <> target_person_id)
);

INSERT OR IGNORE INTO counters (key, value) VALUES ('person_count', 0);

CREATE INDEX IF NOT EXISTS idx_appearances_date     ON appearances(date);
CREATE INDEX IF NOT EXISTS idx_appearances_person   ON appearances(person_id);
CREATE INDEX IF NOT EXISTS idx_log_person           ON recognition_log(person_id);
CREATE INDEX IF NOT EXISTS idx_log_ts               ON recognition_log(ts);
CREATE INDEX IF NOT EXISTS idx_log_event            ON recognition_log(event_type);
CREATE INDEX IF NOT EXISTS idx_gallery_person       ON person_gallery(person_id);
CREATE INDEX IF NOT EXISTS idx_gallery_type         ON person_gallery(crop_type);
CREATE INDEX IF NOT EXISTS idx_gallery_sharpness    ON person_gallery(person_id, crop_type, sharpness DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_suggestion_pending
ON identity_match_suggestions(source_person_id, candidate_person_id)
WHERE status = 'pending';
CREATE INDEX IF NOT EXISTS idx_suggestions_status
ON identity_match_suggestions(status);
CREATE INDEX IF NOT EXISTS idx_suggestions_source
ON identity_match_suggestions(source_person_id);
CREATE INDEX IF NOT EXISTS idx_suggestions_candidate
ON identity_match_suggestions(candidate_person_id);
CREATE INDEX IF NOT EXISTS idx_merge_audit_source
ON identity_merge_audit(source_person_id);
CREATE INDEX IF NOT EXISTS idx_merge_audit_target
ON identity_merge_audit(target_person_id);
CREATE TRIGGER IF NOT EXISTS trg_identity_merge_audit_no_update
BEFORE UPDATE ON identity_merge_audit
BEGIN
    SELECT RAISE(ABORT, 'identity_merge_audit is append-only: UPDATE is not allowed');
END;
CREATE TRIGGER IF NOT EXISTS trg_identity_merge_audit_no_delete
BEFORE DELETE ON identity_merge_audit
BEGIN
    SELECT RAISE(ABORT, 'identity_merge_audit is append-only: DELETE is not allowed');
END;
