from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class ReadOnlyGlobalMemoryError(RuntimeError):
    """A mutating operation was attempted through a read-only memory handle."""


class GlobalMemory:
    def __init__(self, db_path: str | None = None, *, read_only: bool = False):
        self.db_path = Path(db_path or config.DB_PATH)
        self.read_only = bool(read_only)
        self._lock = threading.RLock()
        if self.read_only:
            uri = f"{self.db_path.resolve().as_uri()}?mode=ro"
            self._conn = sqlite3.connect(
                uri,
                uri=True,
                check_same_thread=False,
                isolation_level=None,
            )
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                isolation_level=None,
            )
        self._conn.row_factory = sqlite3.Row
        if self.read_only:
            self._conn.execute("PRAGMA query_only=ON")
        else:
            schema_path = Path(__file__).with_name("schema.sql")
            self._conn.executescript(schema_path.read_text(encoding="utf-8"))
            self._ensure_schema_columns()

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
                "SELECT person_id, name, embedding FROM persons"
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
                 WHERE a.date = ?
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
                "SELECT person_id, name, enrolled_at, cameras FROM persons"
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
                "profile_image": row["profile_image"],
                "profile_image_source": row["profile_image_source"],
                "latest_appearance": self._latest_appearance(row["person_id"]),
            }

    def list_all(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
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
                    "cameras": self._json_list(row["cameras"]),
                    "profile_image": row["profile_image"],
                    "profile_image_source": row["profile_image_source"],
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
                    "best_face_crop": row["best_face_crop"],
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
                (str(image_path), source, person_id),
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
            if row is not None and Path(row["path"]).exists():
                return {"path": row["path"], "sharpness": row["sharpness"]}

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
                return {"path": row["best_face_crop"], "sharpness": 0.0}
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
                    "path": row["path"],
                    "sharpness": row["sharpness"],
                    "session_date": row["session_date"],
                    "video_source": row["video_source"],
                    "width": row["width"],
                    "height": row["height"],
                }
                for row in rows
            ]

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

    def _find_existing_person(self, embedding: np.ndarray, threshold: float | None = None) -> dict | None:
        threshold = config.SIMILARITY_THRESHOLD if threshold is None else float(threshold)
        rows = self._conn.execute(
            "SELECT person_id, name, embedding, embedding_count FROM persons"
        ).fetchall()
        if not rows:
            return None

        matrix = np.vstack([
            np.frombuffer(row["embedding"], dtype=np.float32).copy() for row in rows
        ])
        sims = matrix @ embedding
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim < threshold:
            return None

        row = rows[best_idx]
        return {
            "person_id": row["person_id"],
            "name": row["name"],
            "similarity": best_sim,
            "embedding_count": row["embedding_count"],
        }

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
                str(resolved.resolve()),
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
        missing = [row["id"] for row in rows if not Path(row["path"]).exists()]
        if not missing:
            return
        placeholders = ",".join("?" for _ in missing)
        self._conn.execute(
            f"DELETE FROM person_gallery WHERE id IN ({placeholders})",
            missing,
        )

    def _upsert_appearance(self, person_id: str, appearance_date: str, profile: dict) -> None:
        appearance = profile.get("appearance") or {}
        clothing_fields = [appearance.get("top"), appearance.get("bottom"), appearance.get("shoes")]
        if not any(clothing_fields):
            return

        color = ((profile.get("appearance_signals") or {}).get("color") or {})
        self._conn.execute(
            """
            INSERT OR REPLACE INTO appearances (
                person_id, date, top, bottom, shoes, full_description,
                top_color, bottom_color, best_body_crops, video_sources
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        person_id: str,
        event_type: str,
        similarity: float | None,
        embedding_count_before: int | None,
        embedding_count_after: int,
        video_sources: list[str],
        best_face_crop: str | None = None,
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
            "best_body_crops": self._json_list(row["best_body_crops"]),
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
