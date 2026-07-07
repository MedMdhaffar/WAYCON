PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS people (
    person_id TEXT PRIMARY KEY,
    display_name TEXT,
    cluster_id INTEGER,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    source_profile_path TEXT
);

CREATE TABLE IF NOT EXISTS face_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    model_name TEXT,
    embedding_dim INTEGER,
    norm_type TEXT,
    embedding_json TEXT NOT NULL,
    face_count INTEGER,
    face_crop_count INTEGER,
    cluster_confidence REAL,
    cluster_face_count INTEGER,
    low_confidence INTEGER,
    created_at TEXT NOT NULL,

    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS appearances (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    appearance_date TEXT NOT NULL,
    top TEXT,
    bottom TEXT,
    shoes TEXT,
    full_description TEXT,
    clothing_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    UNIQUE(person_id, appearance_date),
    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS appearance_signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    appearance_date TEXT,
    signal_type TEXT NOT NULL,
    method TEXT,
    sample_count INTEGER,
    top_color TEXT,
    bottom_color TEXT,
    signal_json TEXT,
    created_at TEXT NOT NULL,

    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS reid_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    appearance_date TEXT,
    status TEXT,
    model_name TEXT,
    weights TEXT,
    embedding_dim INTEGER,
    body_embedding_json TEXT,
    aggregation TEXT,
    crop_count INTEGER,
    created_at TEXT NOT NULL,

    UNIQUE(person_id, appearance_date, model_name),
    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS camera_observations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    camera_id TEXT,
    video_path TEXT,
    appearance_date TEXT,
    frame_start INTEGER,
    frame_end INTEGER,
    created_at TEXT NOT NULL,

    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS crop_references (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    appearance_date TEXT,
    crop_type TEXT NOT NULL,
    crop_path TEXT NOT NULL,
    frame_idx INTEGER,
    video_path TEXT,
    created_at TEXT NOT NULL,

    UNIQUE(person_id, crop_type, crop_path),
    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS association_quality (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    source TEXT,
    association_count INTEGER,
    auto_pair_score_mean REAL,
    cluster_confidence REAL,
    low_confidence INTEGER,
    created_at TEXT NOT NULL,

    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE TABLE IF NOT EXISTS raw_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    person_id TEXT NOT NULL,
    profile_path TEXT,
    profile_json TEXT NOT NULL,
    created_at TEXT NOT NULL,

    FOREIGN KEY (person_id) REFERENCES people(person_id)
);

CREATE INDEX IF NOT EXISTS idx_appearances_date
ON appearances(appearance_date);

CREATE INDEX IF NOT EXISTS idx_camera_observations_camera
ON camera_observations(camera_id);

CREATE INDEX IF NOT EXISTS idx_camera_observations_date
ON camera_observations(appearance_date);

CREATE INDEX IF NOT EXISTS idx_crop_references_person
ON crop_references(person_id);

CREATE INDEX IF NOT EXISTS idx_reid_person_date
ON reid_embeddings(person_id, appearance_date);

CREATE INDEX IF NOT EXISTS idx_face_embeddings_person
ON face_embeddings(person_id);
