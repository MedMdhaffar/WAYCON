from __future__ import annotations

import json
import os
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


_FRAME_RE = re.compile(r"f(\d+)", re.IGNORECASE)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _frame_idx(path: str | Path | None) -> int | None:
    if not path:
        return None
    match = _FRAME_RE.search(Path(str(path).replace("\\", "/")).name)
    return int(match.group(1)) if match else None


def _camera_id(video_path: str | Path | None) -> str | None:
    if not video_path:
        return None
    stem = Path(str(video_path).replace("\\", "/")).stem
    return stem or None


def _normalize_embedding(embedding: list[float]) -> np.ndarray | None:
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        return None
    norm = np.linalg.norm(vector)
    if norm == 0:
        return None
    return vector / norm


class GlobalMemoryStore:
    def __init__(self, db_path: str | Path | None = None):
        env_path = os.getenv("WAYCON_GLOBAL_MEMORY_DB")
        self.db_path = Path(db_path or env_path or Path(__file__).resolve().parent / "waycon_memory.db")

    def init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        schema_path = Path(__file__).with_name("schema.sql")
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript(schema_path.read_text(encoding="utf-8"))

    def register_profile_file(self, profile_path: str | Path) -> dict:
        path = Path(profile_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and isinstance(payload.get("people"), list):
            person_ids: list[str] = []
            for profile in payload["people"]:
                result = self.register_profile(profile, path)
                person_ids.extend(result["person_ids"])
            return {
                "db_path": str(self.db_path),
                "registered_count": len(person_ids),
                "person_ids": person_ids,
            }
        return self.register_profile(payload, path)

    def register_profile(self, profile: dict, profile_path: str | Path | None = None) -> dict:
        self.init_db()
        person_id = str(profile.get("id") or profile.get("person_id") or "").strip()
        if not person_id:
            raise ValueError("profile must contain id or person_id")

        created_at = str(profile.get("created_at") or _now())
        updated_at = _now()
        path_value = str(Path(profile_path).resolve()) if profile_path else None
        appearance = profile.get("appearance") or {}
        appearance_date = str(appearance.get("date") or created_at[:10])
        face_meta = profile.get("face_embedding_meta") or {}
        color_signal = (profile.get("appearance_signals") or {}).get("color") or {}
        association = profile.get("association_meta") or {}
        reid = profile.get("reid") or {}

        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            with conn:
                conn.execute(
                    """
                    INSERT INTO people (
                        person_id, display_name, cluster_id, created_at, updated_at, source_profile_path
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(person_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        cluster_id = excluded.cluster_id,
                        updated_at = excluded.updated_at,
                        source_profile_path = excluded.source_profile_path
                    """,
                    (
                        person_id,
                        profile.get("name") or person_id,
                        _to_int(profile.get("cluster_id")),
                        created_at,
                        updated_at,
                        path_value,
                    ),
                )
                self._insert_face_embedding(conn, person_id, profile, face_meta, created_at)
                self._upsert_appearance(conn, person_id, appearance_date, appearance, created_at, updated_at)
                self._insert_color_signal(conn, person_id, appearance_date, color_signal, created_at)
                self._upsert_reid(conn, person_id, appearance_date, reid, created_at)
                self._insert_camera_observations(conn, person_id, appearance_date, profile, created_at)
                self._insert_crop_references(conn, person_id, appearance_date, profile, color_signal, created_at)
                self._insert_association_quality(conn, person_id, profile, association, created_at)
                conn.execute(
                    """
                    INSERT INTO raw_profiles (person_id, profile_path, profile_json, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (person_id, path_value, _json(profile), created_at),
                )

        return {
            "db_path": str(self.db_path),
            "registered_count": 1,
            "person_ids": [person_id],
        }

    def search_by_face(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        self.init_db()
        query = _normalize_embedding(embedding)
        if query is None:
            return []

        matches: list[dict] = []
        with closing(self._connect_rows()) as conn:
            rows = conn.execute(
                """
                SELECT person_id, model_name, embedding_json, created_at
                FROM face_embeddings
                """
            ).fetchall()

        for row in rows:
            stored = _normalize_embedding(json.loads(row["embedding_json"]))
            if stored is None or stored.shape != query.shape:
                continue
            similarity = float(np.dot(query, stored))
            if threshold is not None and similarity < threshold:
                continue
            matches.append({
                "person_id": row["person_id"],
                "similarity": similarity,
                "model_name": row["model_name"],
                "created_at": row["created_at"],
            })

        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:top_k]

    def get_by_date(self, appearance_date: str) -> list[dict]:
        self.init_db()
        with closing(self._connect_rows()) as conn:
            rows = conn.execute(
                """
                SELECT p.*, a.appearance_date, a.top, a.bottom, a.shoes,
                       a.full_description, a.clothing_json
                FROM appearances a
                JOIN people p ON p.person_id = a.person_id
                WHERE a.appearance_date = ?
                ORDER BY p.person_id
                """,
                (appearance_date,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_by_camera(self, camera_id: str) -> list[dict]:
        self.init_db()
        with closing(self._connect_rows()) as conn:
            rows = conn.execute(
                """
                SELECT p.*, c.camera_id, c.video_path, c.appearance_date,
                       c.frame_start, c.frame_end
                FROM camera_observations c
                JOIN people p ON p.person_id = c.person_id
                WHERE c.camera_id = ?
                ORDER BY c.appearance_date DESC, p.person_id
                """,
                (camera_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_person(self, person_id: str) -> dict | None:
        self.init_db()
        with closing(self._connect_rows()) as conn:
            person = conn.execute("SELECT * FROM people WHERE person_id = ?", (person_id,)).fetchone()
            if person is None:
                return None
            appearances = conn.execute(
                "SELECT * FROM appearances WHERE person_id = ? ORDER BY appearance_date DESC LIMIT 10",
                (person_id,),
            ).fetchall()
            cameras = conn.execute(
                "SELECT * FROM camera_observations WHERE person_id = ? ORDER BY created_at DESC LIMIT 20",
                (person_id,),
            ).fetchall()
            crops = conn.execute(
                "SELECT * FROM crop_references WHERE person_id = ? ORDER BY created_at DESC LIMIT 50",
                (person_id,),
            ).fetchall()
        return {
            "person": dict(person),
            "appearances": [dict(row) for row in appearances],
            "cameras": [dict(row) for row in cameras],
            "crops": [dict(row) for row in crops],
        }

    def _connect_rows(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _insert_face_embedding(self, conn: sqlite3.Connection, person_id: str, profile: dict, meta: dict, created_at: str) -> None:
        embedding = profile.get("face_embedding")
        if not embedding:
            return
        conn.execute(
            """
            INSERT INTO face_embeddings (
                person_id, model_name, embedding_dim, norm_type, embedding_json,
                face_count, face_crop_count, cluster_confidence, cluster_face_count,
                low_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                meta.get("model"),
                _to_int(meta.get("dim")),
                meta.get("norm"),
                _json(embedding),
                _to_int(profile.get("face_count")),
                _to_int(profile.get("face_crop_count")),
                _to_float(profile.get("cluster_confidence")),
                _to_int(profile.get("cluster_face_count")),
                int(bool(profile.get("low_confidence"))),
                created_at,
            ),
        )

    def _upsert_appearance(
        self,
        conn: sqlite3.Connection,
        person_id: str,
        appearance_date: str,
        appearance: dict,
        created_at: str,
        updated_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO appearances (
                person_id, appearance_date, top, bottom, shoes, full_description,
                clothing_json, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, appearance_date) DO UPDATE SET
                top = excluded.top,
                bottom = excluded.bottom,
                shoes = excluded.shoes,
                full_description = excluded.full_description,
                clothing_json = excluded.clothing_json,
                updated_at = excluded.updated_at
            """,
            (
                person_id,
                appearance_date,
                appearance.get("top"),
                appearance.get("bottom"),
                appearance.get("shoes"),
                appearance.get("full"),
                _json(appearance),
                created_at,
                updated_at,
            ),
        )

    def _insert_color_signal(self, conn: sqlite3.Connection, person_id: str, appearance_date: str, color: dict, created_at: str) -> None:
        if not color:
            return
        conn.execute(
            """
            INSERT INTO appearance_signals (
                person_id, appearance_date, signal_type, method, sample_count,
                top_color, bottom_color, signal_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                appearance_date,
                "color",
                color.get("method"),
                _to_int(color.get("sample_count")),
                color.get("top"),
                color.get("bottom"),
                _json(color),
                created_at,
            ),
        )

    def _upsert_reid(self, conn: sqlite3.Connection, person_id: str, appearance_date: str, reid: dict, created_at: str) -> None:
        model_name = reid.get("model") or "not_computed"
        conn.execute(
            """
            INSERT INTO reid_embeddings (
                person_id, appearance_date, status, model_name, weights,
                embedding_dim, body_embedding_json, aggregation, crop_count, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, appearance_date, model_name) DO UPDATE SET
                status = excluded.status,
                weights = excluded.weights,
                embedding_dim = excluded.embedding_dim,
                body_embedding_json = excluded.body_embedding_json,
                aggregation = excluded.aggregation,
                crop_count = excluded.crop_count,
                created_at = excluded.created_at
            """,
            (
                person_id,
                appearance_date,
                reid.get("status"),
                model_name,
                reid.get("weights"),
                _to_int(reid.get("embedding_dim")),
                _json(reid.get("body_embedding")) if reid.get("body_embedding") is not None else None,
                reid.get("aggregation"),
                _to_int(reid.get("crop_count")),
                created_at,
            ),
        )

    def _insert_camera_observations(
        self,
        conn: sqlite3.Connection,
        person_id: str,
        appearance_date: str,
        profile: dict,
        created_at: str,
    ) -> None:
        crop_frames = [
            _frame_idx(path)
            for path in [
                *profile.get("face_crops", []),
                *profile.get("body_crops", []),
                *profile.get("best_body_crops", []),
            ]
        ]
        crop_frames = [frame for frame in crop_frames if frame is not None]
        frame_start = min(crop_frames) if crop_frames else None
        frame_end = max(crop_frames) if crop_frames else None
        for video_path in profile.get("video_sources", []) or []:
            camera_id = _camera_id(video_path)
            exists = conn.execute(
                """
                SELECT 1 FROM camera_observations
                WHERE person_id = ? AND camera_id IS ? AND video_path = ? AND appearance_date = ?
                  AND frame_start IS ? AND frame_end IS ?
                """,
                (person_id, camera_id, video_path, appearance_date, frame_start, frame_end),
            ).fetchone()
            if exists:
                continue
            conn.execute(
                """
                INSERT INTO camera_observations (
                    person_id, camera_id, video_path, appearance_date, frame_start, frame_end, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (person_id, camera_id, video_path, appearance_date, frame_start, frame_end, created_at),
            )

    def _insert_crop_references(
        self,
        conn: sqlite3.Connection,
        person_id: str,
        appearance_date: str,
        profile: dict,
        color_signal: dict,
        created_at: str,
    ) -> None:
        seen: set[tuple[str, str]] = set()
        for crop_type, paths in (
            ("face", profile.get("face_crops", [])),
            ("body", profile.get("body_crops", [])),
            ("best_body", profile.get("best_body_crops", [])),
        ):
            for path in paths or []:
                self._insert_crop(conn, person_id, appearance_date, crop_type, path, created_at, seen)

        for sample in color_signal.get("samples", []) or []:
            path = sample.get("path") if isinstance(sample, dict) else None
            if path:
                self._insert_crop(conn, person_id, appearance_date, "color_sample", path, created_at, seen)

    def _insert_crop(
        self,
        conn: sqlite3.Connection,
        person_id: str,
        appearance_date: str,
        crop_type: str,
        crop_path: str,
        created_at: str,
        seen: set[tuple[str, str]],
    ) -> None:
        key = (crop_type, crop_path)
        if key in seen:
            return
        seen.add(key)
        conn.execute(
            """
            INSERT OR IGNORE INTO crop_references (
                person_id, appearance_date, crop_type, crop_path, frame_idx, video_path, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (person_id, appearance_date, crop_type, crop_path, _frame_idx(crop_path), None, created_at),
        )

    def _insert_association_quality(
        self,
        conn: sqlite3.Connection,
        person_id: str,
        profile: dict,
        association: dict,
        created_at: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO association_quality (
                person_id, source, association_count, auto_pair_score_mean,
                cluster_confidence, low_confidence, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                person_id,
                association.get("source"),
                _to_int(association.get("association_count")),
                _to_float(association.get("auto_pair_score_mean")),
                _to_float(profile.get("cluster_confidence")),
                int(bool(profile.get("low_confidence"))),
                created_at,
            ),
        )
