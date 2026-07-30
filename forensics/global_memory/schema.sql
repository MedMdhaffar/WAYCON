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
    notes            TEXT NOT NULL DEFAULT '',
    identity_source  TEXT NOT NULL DEFAULT 'video',
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

CREATE TABLE IF NOT EXISTS face_photo_sources (
    source_id          TEXT PRIMARY KEY,
    person_id          TEXT NOT NULL REFERENCES persons(person_id),
    original_image_path TEXT NOT NULL,
    face_crop_path     TEXT NOT NULL,
    face_bbox          TEXT NOT NULL,
    quality_info       TEXT NOT NULL DEFAULT '{}',
    embedding          BLOB NOT NULL,
    created_at         TEXT NOT NULL,
    source_filename    TEXT NOT NULL,
    import_batch_id    TEXT,
    is_primary         INTEGER NOT NULL DEFAULT 0
                       CHECK (is_primary IN (0, 1)),
    -- 1 marks a primary chosen by a supervisor.  Automatic selection never
    -- overwrites a supervisor choice; see profile_management.resolve_profile_image.
    is_supervisor_selected INTEGER NOT NULL DEFAULT 0
                       CHECK (is_supervisor_selected IN (0, 1)),
    content_sha256     TEXT
);

-- commit_key is a content fingerprint, not an ephemeral batch identifier, so a
-- retry after a lost response or a process restart replays the same durable row.
CREATE TABLE IF NOT EXISTS profile_import_commits (
    commit_key       TEXT PRIMARY KEY,
    content_set_key  TEXT,
    batch_id         TEXT NOT NULL,
    identity_id      TEXT NOT NULL,
    person_id        TEXT NOT NULL,
    outcome          TEXT NOT NULL CHECK (outcome IN ('created', 'updated', 'review')),
    created_at       TEXT NOT NULL,
    result_json      TEXT NOT NULL DEFAULT '{}'
);

-- One durable row owns one canonical set of uploaded bytes.  The key depends
-- only on the sorted, deduplicated SHA-256 values, while action/target/name are
-- stored separately so a retry can be distinguished from a changed intent.
CREATE TABLE IF NOT EXISTS profile_import_content_sets (
    content_set_key        TEXT PRIMARY KEY,
    semantic_action        TEXT NOT NULL
                           CHECK (semantic_action IN (
                               'create_new', 'attach_existing',
                               'review_required', 'skip'
                           )),
    target_person_id       TEXT,
    approved_name_component TEXT NOT NULL DEFAULT '',
    durable_result_json    TEXT NOT NULL DEFAULT '{}',
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);

-- Durable state for a confirmed review_required import.  No synthetic person is
-- created: identity_match_suggestions requires a real source_person_id, so
-- pending phone-import reviews live here and are merged into the same
-- "uncertain suggestions" API surface.
CREATE TABLE IF NOT EXISTS profile_import_reviews (
    review_key          TEXT PRIMARY KEY,
    content_set_key     TEXT NOT NULL,
    candidate_person_id TEXT REFERENCES persons(person_id),
    proposed_name       TEXT NOT NULL,
    similarity          REAL,
    second_similarity   REAL,
    margin              REAL,
    reason              TEXT,
    observation_count   INTEGER NOT NULL DEFAULT 1,
    identity_source     TEXT NOT NULL DEFAULT 'phone',
    embedding           BLOB NOT NULL,
    evidence_json       TEXT NOT NULL DEFAULT '[]',
    status              TEXT NOT NULL DEFAULT 'pending'
                        CHECK (status IN ('pending', 'accepted', 'rejected', 'stale')),
    created_at          TEXT NOT NULL,
    reviewed_at         TEXT,
    reviewed_by         TEXT,
    resolution_action   TEXT
                        CHECK (resolution_action IN (
                            'attach_existing', 'create_new', 'skip'
                        )),
    resolution_target_person_id TEXT,
    resolution_result_json TEXT NOT NULL DEFAULT '{}'
);

-- Individual review embeddings stay internal and are never serialized by the
-- API.  This preserves every accepted phone photo as distinct identity
-- evidence when the supervisor resolves a durable review.
CREATE TABLE IF NOT EXISTS profile_import_review_evidence (
    review_key TEXT NOT NULL REFERENCES profile_import_reviews(review_key),
    source_id  TEXT NOT NULL,
    embedding  BLOB NOT NULL,
    PRIMARY KEY (review_key, source_id)
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
CREATE INDEX IF NOT EXISTS idx_identity_evidence_person
ON identity_evidence(person_id);
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
CREATE INDEX IF NOT EXISTS idx_face_photo_sources_person
ON face_photo_sources(person_id, created_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_face_photo_primary
ON face_photo_sources(person_id) WHERE is_primary = 1;
CREATE UNIQUE INDEX IF NOT EXISTS uq_face_photo_batch_source
ON face_photo_sources(import_batch_id, source_id)
WHERE import_batch_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS uq_face_photo_supervisor_primary
ON face_photo_sources(person_id) WHERE is_supervisor_selected = 1;
CREATE INDEX IF NOT EXISTS idx_face_photo_content
ON face_photo_sources(person_id, content_sha256);
CREATE INDEX IF NOT EXISTS idx_profile_import_reviews_status
ON profile_import_reviews(status, created_at);
CREATE INDEX IF NOT EXISTS idx_profile_import_reviews_candidate
ON profile_import_reviews(candidate_person_id);
CREATE INDEX IF NOT EXISTS idx_profile_import_commits_content_set
ON profile_import_commits(content_set_key);
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
