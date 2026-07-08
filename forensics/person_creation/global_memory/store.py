from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from forensics.person_creation.global_memory.face_photo_registration import (
    embed_face_photo_records,
)
from forensics.person_creation.global_memory.config import (
    FACE_AUTO_MATCH_THRESHOLD,
    FACE_NO_MATCH_THRESHOLD,
)
from forensics.person_creation.global_memory.schema import ensure_schema
from forensics.person_creation.global_memory.similarity import (
    cosine_similarity,
    mean_normalized_embeddings,
    normalize_embedding,
)
from forensics.person_creation.global_memory.media_paths import media_path_for_api

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DB = _PROJECT_ROOT / "forensics" / "person_db" / "global_memory.sqlite"
_NS = uuid.UUID("0fbf8d1c-36ff-41af-9f39-e1e9f3638d83")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True)


def _loads(raw: str | None, default: Any = None) -> Any:
    if raw is None:
        return default
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return default


def _stable_id(prefix: str, *parts: Any) -> str:
    key = "|".join("" if p is None else str(p) for p in parts)
    return f"{prefix}_{uuid.uuid5(_NS, key).hex}"


def _new_person_id(name: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")
    slug = "_".join(part for part in slug.split("_") if part)[:40] or "person"
    return f"{slug}_{uuid.uuid4().hex[:12]}"


def _existing_path(raw: str | None) -> str | None:
    return media_path_for_api(raw, project_root=_PROJECT_ROOT)


def _with_existing_media(row: dict, *path_keys: str, output_key: str = "media_path") -> dict | None:
    for key in path_keys:
        resolved = _existing_path(row.get(key))
        if resolved:
            item = dict(row)
            item[output_key] = resolved
            if len(path_keys) == 1:
                item[path_keys[0]] = resolved
            return item
    return None


def _source_after_merge(old: str, new: str) -> str:
    if old == new:
        return old
    if old == "mixed":
        return "mixed"
    return "mixed"


def _public_match(match: dict | None) -> dict | None:
    if not match:
        return None
    return {
        "person_id": match["person_id"],
        "name": match["name"],
        "identity_source": match["identity_source"],
        "similarity": match["similarity"],
    }


def _optional_embedding(profile: dict) -> list[float] | None:
    for key in ("body_reid_embedding", "reid_embedding"):
        value = profile.get(key)
        if value:
            return normalize_embedding(value)
    reid = profile.get("reid") or {}
    for key in ("body_reid_embedding", "embedding"):
        value = reid.get(key)
        if value:
            return normalize_embedding(value)
    return None


class GlobalMemoryStore:
    def __init__(self, db_path: str | None = None):
        raw_db_path = db_path or os.getenv("PERSON_GLOBAL_MEMORY_DB")
        self.db_path = Path(raw_db_path).expanduser().resolve() if raw_db_path else _DEFAULT_DB
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_registration_result: dict = {}
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        ensure_schema(self._conn)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "GlobalMemoryStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def register_profile(
        self,
        profile: dict,
        profile_path: str | None = None,
        output_dir: str | None = None,
        threshold: float = FACE_AUTO_MATCH_THRESHOLD,
    ) -> str:
        result = self.register_profile_with_result(
            profile,
            profile_path=profile_path,
            output_dir=output_dir,
            auto_match_threshold=threshold,
        )
        return result["person_id"]

    def register_profile_with_result(
        self,
        profile: dict,
        profile_path: str | None = None,
        output_dir: str | None = None,
        auto_match_threshold: float = FACE_AUTO_MATCH_THRESHOLD,
        review_threshold: float = FACE_NO_MATCH_THRESHOLD,
    ) -> dict:
        embedding = normalize_embedding(profile.get("face_embedding") or [])
        name = str(profile.get("name") or profile.get("id") or "unknown").strip() or "unknown"
        all_matches = self.search_by_face(embedding, top_k=1, threshold=None)
        best_match = all_matches[0] if all_matches else None
        matched = bool(best_match and best_match["similarity"] >= auto_match_threshold)
        review_required = bool(
            best_match
            and review_threshold <= best_match["similarity"] < auto_match_threshold
        )
        now = _now()
        suggestion_id = None

        with self._conn:
            if matched:
                person_id = best_match["person_id"]
                row = self._person_row(person_id)
                source = _source_after_merge(row["identity_source"], "video_profile")
                merged_embedding = mean_normalized_embeddings([
                    _loads(row["face_embedding_json"], []),
                    embedding,
                ])
                self._conn.execute(
                    """
                    UPDATE persons
                    SET name = ?, face_embedding_json = ?, identity_source = ?, updated_at = ?
                    WHERE person_id = ?
                    """,
                    (row["name"] or name, _json(merged_embedding), source, now, person_id),
                )
                action = "updated_existing"
            else:
                person_id = _new_person_id(name)
                self._conn.execute(
                    """
                    INSERT INTO persons
                    (person_id, name, face_embedding_json, identity_source, created_at, updated_at, is_active)
                    VALUES (?, ?, ?, 'video_profile', ?, ?, 1)
                    """,
                    (person_id, name, _json(embedding), now, now),
                )
                action = "created"

            self._upsert_appearance(person_id, profile, now)
            self._upsert_profile_run(person_id, profile, profile_path, output_dir, now)
            self._insert_crop_references(person_id, profile, now)

            if (
                not matched
                and best_match
                and best_match["similarity"] >= review_threshold
                and best_match["person_id"] != person_id
            ):
                suggestion_id = self._create_match_suggestion(
                    person_id,
                    best_match["person_id"],
                    best_match["similarity"],
                    auto_match_threshold,
                    now,
                )

        result = {
            "operation": "register_profile",
            "person_id": person_id,
            "action": action,
            "created": action == "created",
            "matched": matched,
            "review_required": review_required,
            "suggestion_id": suggestion_id,
            "best_match": _public_match(best_match),
            "similarity": best_match["similarity"] if best_match else None,
            "threshold": auto_match_threshold,
            "review_threshold": review_threshold,
            "auto_match_threshold": auto_match_threshold,
            "db_path": str(self.db_path),
        }
        self.last_registration_result = result
        return result

    def register_face_photo_identity(
        self,
        name: str,
        image_paths: list[str],
        threshold: float = FACE_AUTO_MATCH_THRESHOLD,
    ) -> str:
        result = self.register_face_photo_identity_with_result(
            name,
            image_paths,
            threshold=threshold,
            allow_update_existing=False,
        )
        return result["person_id"]

    def register_face_photo_identity_with_result(
        self,
        name: str,
        image_paths: list[str],
        threshold: float = FACE_AUTO_MATCH_THRESHOLD,
        review_threshold: float = FACE_NO_MATCH_THRESHOLD,
        allow_update_existing: bool = False,
        target_person_id: str | None = None,
    ) -> dict:
        crop_dir = self.db_path.parent / "global_memory_face_crops"
        records = embed_face_photo_records(image_paths, save_crops_dir=crop_dir)
        if not records["embeddings"]:
            reasons = "; ".join(f"{s['image_path']}: {s['reason']}" for s in records["skipped"])
            raise ValueError(f"no valid faces found in phone photos ({reasons})")

        embedding = mean_normalized_embeddings(records["embeddings"])
        matches = self.search_by_face(embedding, top_k=1, threshold=None)
        best_match = matches[0] if matches else None
        now = _now()
        clean_name = name.strip() or "unknown"
        suggestion_id = None

        with self._conn:
            if target_person_id:
                person_id = target_person_id
                row = self._person_row(person_id)
                merged = mean_normalized_embeddings([
                    _loads(row["face_embedding_json"], []),
                    embedding,
                ])
                source = _source_after_merge(row["identity_source"], "phone_photo")
                self._conn.execute(
                    """
                    UPDATE persons
                    SET face_embedding_json = ?, identity_source = ?, updated_at = ?
                    WHERE person_id = ?
                    """,
                    (_json(merged), source, now, person_id),
                )
                action = "updated_target"
                matched = False
            elif allow_update_existing and best_match and best_match["similarity"] >= threshold:
                person_id = best_match["person_id"]
                row = self._person_row(person_id)
                merged = mean_normalized_embeddings([
                    _loads(row["face_embedding_json"], []),
                    embedding,
                ])
                source = _source_after_merge(row["identity_source"], "phone_photo")
                self._conn.execute(
                    """
                    UPDATE persons
                    SET name = ?, face_embedding_json = ?, identity_source = ?, updated_at = ?
                    WHERE person_id = ?
                    """,
                    (row["name"] or clean_name, _json(merged), source, now, person_id),
                )
                action = "updated_existing"
                matched = True
            else:
                person_id = _new_person_id(clean_name)
                self._conn.execute(
                    """
                    INSERT INTO persons
                    (person_id, name, face_embedding_json, identity_source, created_at, updated_at, is_active)
                    VALUES (?, ?, ?, 'phone_photo', ?, ?, 1)
                    """,
                    (person_id, clean_name, _json(embedding), now, now),
                )
                action = "created"
                matched = False

            self._insert_face_photo_sources(person_id, records["sources"], now)

            if (
                action == "created"
                and best_match
                and best_match["person_id"] != person_id
                and best_match["similarity"] >= review_threshold
            ):
                suggestion_id = self._create_match_suggestion(
                    person_id,
                    best_match["person_id"],
                    best_match["similarity"],
                    threshold,
                    now,
                )

        result = {
            "operation": "register_face_photo_identity",
            "person_id": person_id,
            "action": action,
            "created": action == "created",
            "matched": matched,
            "review_required": bool(
                action == "created"
                and best_match
                and best_match["similarity"] >= review_threshold
            ),
            "suggestion_id": suggestion_id,
            "possible_duplicate": bool(best_match and best_match["similarity"] >= review_threshold),
            "best_match": _public_match(best_match),
            "similarity": best_match["similarity"] if best_match else None,
            "threshold": threshold,
            "review_threshold": review_threshold,
            "auto_match_threshold": threshold,
            "db_path": str(self.db_path),
            "registered_images": len(records["sources"]),
            "skipped": records["skipped"],
        }
        self.last_registration_result = result
        return result

    def add_face_photos_to_person(self, person_id: str, image_paths: list[str]) -> dict:
        result = self.register_face_photo_identity_with_result(
            name="",
            image_paths=image_paths,
            target_person_id=person_id,
        )
        result["operation"] = "add_face_photos_to_person"
        self.last_registration_result = result
        return result

    def search_by_face(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[dict]:
        query = normalize_embedding(embedding)
        matches: list[dict] = []
        for row in self._conn.execute("SELECT * FROM persons WHERE is_active = 1"):
            try:
                sim = cosine_similarity(query, _loads(row["face_embedding_json"], []))
            except ValueError:
                continue
            if threshold is not None and sim < threshold:
                continue
            matches.append({
                "person_id": row["person_id"],
                "name": row["name"],
                "similarity": sim,
                "identity_source": row["identity_source"],
            })
        matches.sort(key=lambda item: item["similarity"], reverse=True)
        return matches[:max(0, int(top_k))]

    def get_person_profile_image(self, person_id: str) -> str | None:
        for row in self._conn.execute(
            """
            SELECT face_crop_path AS path
            FROM face_photo_sources
            WHERE person_id = ?
              AND face_crop_path IS NOT NULL
              AND face_crop_path != ''
            ORDER BY created_at
            """,
            (person_id,),
        ):
            existing = _existing_path(row["path"])
            if existing:
                return existing

        for row in self._conn.execute(
            """
            SELECT image_path AS path
            FROM face_photo_sources
            WHERE person_id = ?
              AND image_path IS NOT NULL
              AND image_path != ''
            ORDER BY created_at
            """,
            (person_id,),
        ):
            existing = _existing_path(row["path"])
            if existing:
                return existing

        for row in self._conn.execute(
            """
            SELECT crop_path AS path
            FROM crop_references
            WHERE person_id = ? AND crop_type = 'face'
            ORDER BY created_at DESC
            """,
            (person_id,),
        ):
            existing = _existing_path(row["path"])
            if existing:
                return existing
        return None

    def get_person(self, person_id: str) -> dict | None:
        row = self._person_row(person_id, missing_ok=True)
        if row is None:
            return None
        crop_rows = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM crop_references WHERE person_id = ? ORDER BY crop_type, crop_path",
                (person_id,),
            )
        ]
        face_photo_rows = [
            dict(r)
            for r in self._conn.execute(
                "SELECT * FROM face_photo_sources WHERE person_id = ? ORDER BY created_at",
                (person_id,),
            )
        ]
        visible_face_photos = [
            item
            for item in (
                _with_existing_media(r, "face_crop_path", "image_path")
                for r in face_photo_rows
            )
            if item is not None
        ]
        total_video_face = sum(1 for r in crop_rows if r.get("crop_type") == "face")
        total_body = sum(1 for r in crop_rows if r.get("crop_type") == "body")
        total_best_body = sum(1 for r in crop_rows if r.get("crop_type") == "best_body")
        video_face_crops = [
            item
            for item in (
                _with_existing_media(r, "crop_path")
                for r in crop_rows
                if r.get("crop_type") == "face"
            )
            if item is not None
        ]
        body_crops = [
            item
            for item in (
                _with_existing_media(r, "crop_path")
                for r in crop_rows
                if r.get("crop_type") == "body"
            )
            if item is not None
        ]
        best_body_crops = [
            item
            for item in (
                _with_existing_media(r, "crop_path")
                for r in crop_rows
                if r.get("crop_type") == "best_body"
            )
            if item is not None
        ]
        suggestions = self._suggestions_for_person(person_id)
        person = self._row_to_dict(row, json_fields={"face_embedding_json": "face_embedding"})
        person["profile_image_path"] = self.get_person_profile_image(person_id)
        return {
            "person": person,
            "profile_image_path": person["profile_image_path"],
            "appearances": [
                self._row_to_dict(r, json_fields={
                    "color_signals_json": "color_signals",
                    "reid_json": "reid",
                    "body_reid_embedding_json": "body_reid_embedding",
                })
                for r in self._conn.execute(
                    "SELECT * FROM appearances WHERE person_id = ? ORDER BY date, created_at",
                    (person_id,),
                )
            ],
            "profile_runs": [
                self._row_to_dict(r, json_fields={"video_sources_json": "video_sources"})
                for r in self._conn.execute(
                    "SELECT * FROM profile_runs WHERE person_id = ? ORDER BY created_at",
                    (person_id,),
                )
            ],
            "face_photo_sources": visible_face_photos,
            "video_face_crops": video_face_crops,
            "body_crops": body_crops,
            "best_body_crops": best_body_crops,
            "crop_references": crop_rows,
            "total_video_face_crop_count": total_video_face,
            "valid_video_face_crop_count": len(video_face_crops),
            "missing_video_face_crop_count": total_video_face - len(video_face_crops),
            "total_body_crop_count": total_body,
            "valid_body_crop_count": len(body_crops),
            "missing_body_crop_count": total_body - len(body_crops),
            "total_best_body_crop_count": total_best_body,
            "valid_best_body_crop_count": len(best_body_crops),
            "missing_best_body_crop_count": total_best_body - len(best_body_crops),
            "pending_suggestions": suggestions,
        }

    def list_persons(self, include_inactive: bool = False) -> list[dict]:
        active_filter = "" if include_inactive else "WHERE p.is_active = 1"
        rows = self._conn.execute(
            f"""
            SELECT
                p.person_id,
                p.name,
                p.notes,
                p.identity_source,
                p.created_at,
                p.updated_at,
                p.is_active,
                p.merged_into_person_id,
                COUNT(DISTINCT a.id) AS appearance_count,
                COUNT(DISTINCT pr.id) AS profile_run_count,
                COUNT(DISTINCT fps.id) AS face_photo_count,
                COUNT(DISTINCT face_crops.id) AS video_face_crop_count,
                COUNT(DISTINCT body_crops.id) AS body_crop_count,
                COUNT(DISTINCT best_body_crops.id) AS best_body_crop_count,
                COUNT(DISTINCT ims.id) AS pending_suggestion_count
            FROM persons p
            LEFT JOIN appearances a ON a.person_id = p.person_id
            LEFT JOIN profile_runs pr ON pr.person_id = p.person_id
            LEFT JOIN face_photo_sources fps ON fps.person_id = p.person_id
            LEFT JOIN crop_references face_crops
                ON face_crops.person_id = p.person_id AND face_crops.crop_type = 'face'
            LEFT JOIN crop_references body_crops
                ON body_crops.person_id = p.person_id AND body_crops.crop_type = 'body'
            LEFT JOIN crop_references best_body_crops
                ON best_body_crops.person_id = p.person_id AND best_body_crops.crop_type = 'best_body'
            LEFT JOIN identity_match_suggestions ims
                ON ims.status = 'pending'
                AND (ims.new_person_id = p.person_id OR ims.candidate_person_id = p.person_id)
            {active_filter}
            GROUP BY p.person_id
            ORDER BY p.updated_at DESC, p.created_at DESC
            """
        )
        persons = []
        for row in rows:
            item = dict(row)
            item["profile_image_path"] = self.get_person_profile_image(item["person_id"])
            persons.append(item)
        return persons

    def update_person_details(
        self,
        person_id: str,
        name: str | None = None,
        notes: str | None = None,
    ) -> dict:
        row = self._person_row(person_id)
        new_name = row["name"] if name is None else (name.strip() or row["name"])
        new_notes = row["notes"] if notes is None else notes.strip()
        now = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE persons SET name = ?, notes = ?, updated_at = ? WHERE person_id = ?",
                (new_name, new_notes, now, person_id),
            )
        detail = self.get_person(person_id)
        return detail["person"] if detail else {}

    def merge_persons(
        self,
        source_person_id: str,
        target_person_id: str,
        new_name: str | None = None,
    ) -> dict:
        if source_person_id == target_person_id:
            raise ValueError("source_person_id and target_person_id must be different")
        now = _now()
        source = self._person_row(source_person_id)
        target = self._person_row(target_person_id)
        merged_embedding = mean_normalized_embeddings([
            _loads(source["face_embedding_json"], []),
            _loads(target["face_embedding_json"], []),
        ])

        with self._conn:
            self._move_table_rows("appearances", source_person_id, target_person_id)
            self._move_table_rows("profile_runs", source_person_id, target_person_id)
            self._move_table_rows("crop_references", source_person_id, target_person_id)
            self._move_table_rows("face_photo_sources", source_person_id, target_person_id)

            final_name = (new_name or "").strip() or target["name"]
            final_notes = target["notes"] or source["notes"]
            final_source = self._infer_identity_source(
                target_person_id,
                [source["identity_source"], target["identity_source"]],
            )
            self._conn.execute(
                """
                UPDATE persons
                SET name = ?, notes = ?, face_embedding_json = ?, identity_source = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (final_name, final_notes, _json(merged_embedding), final_source, now, target_person_id),
            )
            self._conn.execute(
                """
                UPDATE persons
                SET is_active = 0, merged_into_person_id = ?, updated_at = ?
                WHERE person_id = ?
                """,
                (target_person_id, now, source_person_id),
            )
            self._conn.execute(
                """
                UPDATE identity_match_suggestions
                SET status = 'accepted', resolved_at = ?
                WHERE status = 'pending'
                  AND new_person_id = ?
                  AND candidate_person_id = ?
                """,
                (now, source_person_id, target_person_id),
            )
            self._conn.execute(
                """
                UPDATE identity_match_suggestions
                SET status = 'rejected', resolved_at = ?
                WHERE status = 'pending'
                  AND (new_person_id = ? OR candidate_person_id = ?)
                """,
                (now, source_person_id, source_person_id),
            )

        return {
            "ok": True,
            "source_person_id": source_person_id,
            "target_person_id": target_person_id,
            "identity_source": final_source,
        }

    def list_suggestions(self, status: str = "pending") -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                s.*,
                np.name AS new_person_name,
                np.identity_source AS new_identity_source,
                cp.name AS candidate_person_name,
                cp.identity_source AS candidate_identity_source
            FROM identity_match_suggestions s
            JOIN persons np ON np.person_id = s.new_person_id
            JOIN persons cp ON cp.person_id = s.candidate_person_id
            WHERE s.status = ?
              AND np.is_active = 1
              AND cp.is_active = 1
            ORDER BY s.similarity DESC, s.created_at DESC
            """,
            (status,),
        )
        return [self._enrich_suggestion(dict(r)) for r in rows]

    def accept_suggestion(self, suggestion_id: str) -> dict:
        row = self._suggestion_row(suggestion_id)
        if row["status"] != "pending":
            raise ValueError("suggestion is not pending")
        result = self.merge_persons(row["new_person_id"], row["candidate_person_id"])
        now = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE identity_match_suggestions SET status = 'accepted', resolved_at = ? WHERE id = ?",
                (now, suggestion_id),
            )
        return result

    def reject_suggestion(self, suggestion_id: str) -> dict:
        row = self._suggestion_row(suggestion_id)
        now = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE identity_match_suggestions SET status = 'rejected', resolved_at = ? WHERE id = ?",
                (now, suggestion_id),
            )
        return {"ok": True, "suggestion_id": row["id"], "status": "rejected"}

    def search_by_date(self, date: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT p.person_id, p.name, p.identity_source, a.*
            FROM appearances a
            JOIN persons p ON p.person_id = a.person_id
            WHERE a.date = ?
            ORDER BY p.name
            """,
            (date,),
        )
        results = []
        for row in rows:
            item = dict(row)
            item["color_signals"] = _loads(item.pop("color_signals_json", None), {})
            item["reid"] = _loads(item.pop("reid_json", None), {})
            item["body_reid_embedding"] = _loads(item.pop("body_reid_embedding_json", None), None)
            results.append(item)
        return results

    def _person_row(self, person_id: str, missing_ok: bool = False):
        row = self._conn.execute("SELECT * FROM persons WHERE person_id = ?", (person_id,)).fetchone()
        if row is None and not missing_ok:
            raise KeyError(f"person not found: {person_id}")
        return row

    def _suggestion_row(self, suggestion_id: str):
        row = self._conn.execute(
            "SELECT * FROM identity_match_suggestions WHERE id = ?",
            (suggestion_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"suggestion not found: {suggestion_id}")
        return row

    def _create_match_suggestion(
        self,
        new_person_id: str,
        candidate_person_id: str,
        similarity: float,
        threshold: float,
        now: str,
    ) -> str:
        existing = self._conn.execute(
            """
            SELECT id
            FROM identity_match_suggestions
            WHERE new_person_id = ?
              AND candidate_person_id = ?
              AND status = 'pending'
            LIMIT 1
            """,
            (new_person_id, candidate_person_id),
        ).fetchone()
        if existing:
            return existing["id"]

        suggestion_id = f"suggestion_{uuid.uuid4().hex}"
        try:
            self._conn.execute(
                """
                INSERT INTO identity_match_suggestions
                (id, new_person_id, candidate_person_id, similarity, threshold, status, created_at)
                VALUES (?, ?, ?, ?, ?, 'pending', ?)
                """,
                (suggestion_id, new_person_id, candidate_person_id, float(similarity), float(threshold), now),
            )
        except sqlite3.IntegrityError:
            existing = self._conn.execute(
                """
                SELECT id
                FROM identity_match_suggestions
                WHERE new_person_id = ?
                  AND candidate_person_id = ?
                  AND status = 'pending'
                LIMIT 1
                """,
                (new_person_id, candidate_person_id),
            ).fetchone()
            if existing:
                return existing["id"]
            raise
        return suggestion_id

    def _suggestions_for_person(self, person_id: str) -> list[dict]:
        rows = self._conn.execute(
            """
            SELECT
                s.*,
                np.name AS new_person_name,
                np.identity_source AS new_identity_source,
                cp.name AS candidate_person_name,
                cp.identity_source AS candidate_identity_source
            FROM identity_match_suggestions s
            JOIN persons np ON np.person_id = s.new_person_id
            JOIN persons cp ON cp.person_id = s.candidate_person_id
            WHERE s.status = 'pending'
              AND (s.new_person_id = ? OR s.candidate_person_id = ?)
              AND np.is_active = 1
              AND cp.is_active = 1
            ORDER BY s.similarity DESC, s.created_at DESC
            """,
            (person_id, person_id),
        )
        return [self._enrich_suggestion(dict(r)) for r in rows]

    def _enrich_suggestion(self, suggestion: dict) -> dict:
        suggestion["new_profile_image_path"] = self.get_person_profile_image(suggestion["new_person_id"])
        suggestion["candidate_profile_image_path"] = self.get_person_profile_image(suggestion["candidate_person_id"])
        suggestion["review_threshold"] = FACE_NO_MATCH_THRESHOLD
        suggestion["auto_match_threshold"] = suggestion.get("threshold", FACE_AUTO_MATCH_THRESHOLD)
        return suggestion

    def _insert_face_photo_sources(self, person_id: str, sources: list[dict], now: str) -> None:
        for source in sources:
            source_id = _stable_id("photo", person_id, source["image_path"])
            self._conn.execute(
                """
                INSERT OR IGNORE INTO face_photo_sources
                (id, person_id, image_path, face_crop_path, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (source_id, person_id, source["image_path"], source.get("face_crop_path"), now),
            )

    def _move_table_rows(self, table: str, source_person_id: str, target_person_id: str) -> None:
        rows = list(self._conn.execute(f"SELECT id FROM {table} WHERE person_id = ?", (source_person_id,)))
        for row in rows:
            try:
                self._conn.execute(
                    f"UPDATE {table} SET person_id = ? WHERE id = ?",
                    (target_person_id, row["id"]),
                )
            except sqlite3.IntegrityError:
                self._conn.execute(f"DELETE FROM {table} WHERE id = ?", (row["id"],))

    def _infer_identity_source(self, person_id: str, sources: list[str]) -> str:
        if "mixed" in sources:
            return "mixed"
        counts = self._conn.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM face_photo_sources WHERE person_id = ?) AS photo_count,
                (SELECT COUNT(*) FROM profile_runs WHERE person_id = ?) AS run_count
            """,
            (person_id, person_id),
        ).fetchone()
        has_photo = int(counts["photo_count"] or 0) > 0 or "phone_photo" in sources
        has_video = int(counts["run_count"] or 0) > 0 or "video_profile" in sources
        if has_photo and has_video:
            return "mixed"
        if has_photo:
            return "phone_photo"
        return "video_profile"

    def _upsert_appearance(self, person_id: str, profile: dict, now: str) -> None:
        appearance = profile.get("appearance") or {}
        if not appearance:
            return
        date = appearance.get("date")
        app_id = _stable_id("appearance", person_id, date or profile.get("id") or now)
        color = ((profile.get("appearance_signals") or {}).get("color") or None)
        reid = profile.get("reid") or None
        body_reid_embedding = _optional_embedding(profile)
        self._conn.execute(
            """
            INSERT INTO appearances
            (id, person_id, date, top, bottom, shoes, full, color_signals_json,
             reid_json, body_reid_embedding_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, date) DO UPDATE SET
                top = excluded.top,
                bottom = excluded.bottom,
                shoes = excluded.shoes,
                full = excluded.full,
                color_signals_json = excluded.color_signals_json,
                reid_json = excluded.reid_json,
                body_reid_embedding_json = excluded.body_reid_embedding_json
            """,
            (
                app_id,
                person_id,
                date,
                appearance.get("top"),
                appearance.get("bottom"),
                appearance.get("shoes"),
                appearance.get("full"),
                _json(color) if color is not None else None,
                _json(reid) if reid is not None else None,
                _json(body_reid_embedding) if body_reid_embedding is not None else None,
                now,
            ),
        )

    def _upsert_profile_run(
        self,
        person_id: str,
        profile: dict,
        profile_path: str | None,
        output_dir: str | None,
        now: str,
    ) -> None:
        run_id = _stable_id(
            "run",
            profile_path or output_dir or profile.get("id"),
            profile.get("cluster_id"),
        )
        self._conn.execute(
            """
            INSERT INTO profile_runs
            (id, person_id, profile_path, output_dir, video_sources_json, cluster_id,
             cluster_confidence, cluster_face_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                person_id = excluded.person_id,
                profile_path = excluded.profile_path,
                output_dir = excluded.output_dir,
                video_sources_json = excluded.video_sources_json,
                cluster_id = excluded.cluster_id,
                cluster_confidence = excluded.cluster_confidence,
                cluster_face_count = excluded.cluster_face_count
            """,
            (
                run_id,
                person_id,
                profile_path,
                output_dir,
                _json(profile.get("video_sources") or []),
                str(profile.get("cluster_id")) if profile.get("cluster_id") is not None else None,
                float(profile.get("cluster_confidence", 0.0) or 0.0),
                int(profile.get("cluster_face_count", profile.get("face_crop_count", 0)) or 0),
                now,
            ),
        )

    def _insert_crop_references(self, person_id: str, profile: dict, now: str) -> None:
        crop_sets = [
            ("face", profile.get("face_crops") or []),
            ("body", profile.get("body_crops") or []),
            ("best_body", profile.get("best_body_crops") or []),
        ]
        for crop_type, paths in crop_sets:
            for raw in paths:
                crop_path = str(raw)
                crop_id = _stable_id("crop", person_id, crop_type, crop_path)
                self._conn.execute(
                    """
                    INSERT OR IGNORE INTO crop_references
                    (id, person_id, crop_type, crop_path, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (crop_id, person_id, crop_type, crop_path, now),
                )

    @staticmethod
    def _row_to_dict(row, json_fields: dict[str, str] | None = None) -> dict:
        item = dict(row)
        for old_key, new_key in (json_fields or {}).items():
            item[new_key] = _loads(item.pop(old_key, None), None)
        return item
