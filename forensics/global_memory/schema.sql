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

CREATE INDEX IF NOT EXISTS idx_appearances_date     ON appearances(date);
CREATE INDEX IF NOT EXISTS idx_appearances_person   ON appearances(person_id);
CREATE INDEX IF NOT EXISTS idx_log_person           ON recognition_log(person_id);
CREATE INDEX IF NOT EXISTS idx_log_ts               ON recognition_log(ts);
CREATE INDEX IF NOT EXISTS idx_log_event            ON recognition_log(event_type);
CREATE INDEX IF NOT EXISTS idx_gallery_person       ON person_gallery(person_id);
CREATE INDEX IF NOT EXISTS idx_gallery_type         ON person_gallery(crop_type);
CREATE INDEX IF NOT EXISTS idx_gallery_sharpness    ON person_gallery(person_id, crop_type, sharpness DESC);
