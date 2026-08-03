from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import re
import shutil
import sqlite3
import threading
import time
import uuid
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

import cv2
import numpy as np
from flask import Blueprint, current_app, jsonify, request

from forensics.face_engine.client import FaceEngineClient
from forensics.global_memory import GlobalMemory, PersonMergeError
from forensics.global_memory import store as global_memory_store
from forensics.global_memory.identity_policy import (
    IdentityCandidate,
    IdentityDecisionReason,
    IdentityDecisionType,
    IdentityPolicyConfig,
    evaluate_identity_decision,
)
from forensics.media_paths import (
    MediaPathError,
    get_media_root,
    normalize_media_path,
    resolve_media_path,
)
from forensics.person_creation.quality_config import load_quality_filter_config


bp = Blueprint("profile_management", __name__)

_TERMINAL_IMAGE_STATES = {
    "valid",
    "no_face",
    "multiple_faces",
    "quality_rejected",
    "failed",
    "skipped",
}
# States whose staged media may be reclaimed by the sweeper or by capacity
# pressure.  "processing" and "committing" are never evicted.
_EVICTABLE_BATCH_STATES = {"ready", "cancelled", "committed", "failed"}
_ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
_ACTION_VALUES = {"create_new", "attach_existing", "review_required", "skip"}
_SAFE_NAME = re.compile(r"[\s_-]+")
_SAFE_PERSON_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SAFE_KEY = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_FINGERPRINT_VERSION = "v1"
_CONTENT_SET_VERSION = "content-set-v1"
_TRUSTED_ENROLMENT_ENV = "PROFILE_IMPORT_TRUSTED_PHONE_ENROLMENT"
_TRUSTED_IDENTITY_SOURCE = "phone_supervised"

PROFILE_IMAGE_PRIORITY = (
    "supervisor_phone_crop",
    "newest_phone_crop",
    "video_crop",
    "placeholder",
)


class OperationConflict(RuntimeError):
    """The same uploaded bytes were previously committed under another intent."""


def _profile_memory(
    db_path: str | None = None,
    *,
    read_only: bool = False,
    media_root: str | Path | None = None,
) -> GlobalMemory:
    """Open Global Memory after enforcing the profile-test runtime DB guard.

    Production leaves ``PROFILE_IMPORT_PROTECTED_RUNTIME_DB`` unset.  The
    profile-management tests set it to the real runtime database, making every
    helper, worker, restart simulation and API construction fail closed before
    SQLite can open the protected DB or its WAL/SHM sidecars.
    """
    selected = Path(
        db_path or os.getenv("FORENSICS_MEMORY_DB", global_memory_store.config.DB_PATH)
    ).expanduser().resolve(strict=False)
    protected_value = os.getenv("PROFILE_IMPORT_PROTECTED_RUNTIME_DB", "").strip()
    if protected_value:
        protected = Path(protected_value).expanduser().resolve(strict=False)
        if selected == protected:
            raise RuntimeError("profile management refused to open the protected runtime database")
    return GlobalMemory(
        db_path=str(selected),
        read_only=read_only,
        media_root=media_root,
    )


def filename_to_profile_name(filename: str) -> str:
    """Convert a browser filename to an editable, human-readable proposal."""
    leaf = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    stem = Path(leaf).stem
    words = [part for part in _SAFE_NAME.split(stem.strip()) if part]
    return " ".join(word[:1].upper() + word[1:].lower() for word in words) or "Unnamed"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _media_url(path: str | None) -> str | None:
    return f"/api/images?path={quote(path, safe='')}" if path else None


def _normalize_embedding(value: Any) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.shape != (512,) or not np.isfinite(vector).all():
        raise ValueError("Face Engine returned an invalid embedding")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Face Engine returned a zero embedding")
    return vector / norm


def _sharpness(image: np.ndarray) -> float:
    return float(cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def _supervisor_id() -> str:
    return (
        request.headers.get("X-WAYCON-Supervisor", "").strip()
        or request.remote_addr
        or "local"
    )[:200]


def _trusted_phone_enrolment() -> bool:
    """Whether a supervisor-approved phone crop counts as trusted enrolment."""
    return os.getenv(_TRUSTED_ENROLMENT_ENV, "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


# --- media-path safety -------------------------------------------------------

def prepare_media_directory(media_root: Path, *parts: str) -> Path:
    """Create and validate a destination directory inside the media root.

    Every component is checked for symlinks *before* it is used, so a planted
    symlink cannot redirect a later write outside the configured root.
    """
    root = Path(media_root).resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    current = root
    for part in parts:
        if not _SAFE_KEY.fullmatch(str(part)):
            raise ValueError("media destination component is not a safe name")
        current = current / part
        if current.is_symlink():
            raise ValueError("media destination path contains a symlink")
        current.mkdir(exist_ok=True)
        if current.resolve(strict=False) != current:
            raise ValueError("media destination path escapes the media root")
    try:
        current.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ValueError("media destination path escapes the media root") from exc
    return current


def safe_copy_into(source: Path, destination: Path, created: list[Path]) -> Path:
    """Copy to a temporary file inside the validated directory, then rename.

    Every path this creates is appended to ``created`` so a caller can remove
    them all when its transaction rolls back.
    """
    if destination.is_symlink():
        raise ValueError("media destination file is a symlink")
    temporary = destination.with_name(destination.name + ".part")
    if temporary.is_symlink():
        raise ValueError("media staging file is a symlink")
    created.append(temporary)
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    created.remove(temporary)
    created.append(destination)
    return destination


def discard_created_media(created: list[Path]) -> None:
    """Remove files written for a transaction that did not commit."""
    while created:
        path = created.pop()
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _media_file_exists(relative: str | None, media_root: Path) -> bool:
    if not relative:
        return False
    try:
        resolve_media_path(
            relative,
            media_root=media_root,
            require_exists=True,
            image_only=True,
        )
        return True
    except (MediaPathError, FileNotFoundError, OSError):
        return False


# --- profile-image resolution ------------------------------------------------

def resolve_profile_image(
    connection: sqlite3.Connection,
    person_id: str,
    media_root: Path,
) -> dict:
    """Resolve the effective profile image using the documented priority.

    1. supervisor-selected phone face crop
    2. newest valid phone face crop
    3. best valid video face crop
    4. placeholder
    """
    supervisor = connection.execute(
        "SELECT face_crop_path FROM face_photo_sources "
        "WHERE person_id=? AND is_supervisor_selected=1",
        (person_id,),
    ).fetchone()
    if supervisor is not None and _media_file_exists(supervisor["face_crop_path"], media_root):
        return {"path": supervisor["face_crop_path"], "origin": "supervisor_phone_crop"}

    phone_rows = connection.execute(
        "SELECT face_crop_path FROM face_photo_sources "
        "WHERE person_id=? ORDER BY is_primary DESC, created_at DESC, rowid DESC",
        (person_id,),
    ).fetchall()
    for row in phone_rows:
        if _media_file_exists(row["face_crop_path"], media_root):
            return {"path": row["face_crop_path"], "origin": "newest_phone_crop"}

    video_rows = connection.execute(
        "SELECT path FROM person_gallery "
        "WHERE person_id=? AND crop_type='face' "
        "  AND (video_source IS NULL OR video_source <> 'phone_photo') "
        "ORDER BY sharpness DESC, id DESC",
        (person_id,),
    ).fetchall()
    for row in video_rows:
        if _media_file_exists(row["path"], media_root):
            return {"path": row["path"], "origin": "video_crop"}

    log_rows = connection.execute(
        "SELECT best_face_crop FROM recognition_log "
        "WHERE person_id=? AND best_face_crop IS NOT NULL ORDER BY id DESC",
        (person_id,),
    ).fetchall()
    for row in log_rows:
        if _media_file_exists(row["best_face_crop"], media_root):
            return {"path": row["best_face_crop"], "origin": "video_crop"}

    return {"path": None, "origin": "placeholder"}


# --- durable commit fingerprint ---------------------------------------------

def commit_fingerprint(
    *,
    action: str,
    scope: str,
    content_hashes: list[str],
    name_component: str = "",
) -> str:
    """Stable content fingerprint for one durable import operation.

    Deliberately independent of batch_id/identity_id so a retry after a lost
    response, a coordinator restart or a re-upload of the same files replays the
    original durable row instead of creating a duplicate.
    """
    payload = "|".join(
        [
            _FINGERPRINT_VERSION,
            action,
            scope,
            ",".join(sorted(content_hashes)),
            name_component,
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def content_set_fingerprint(content_hashes: list[str]) -> str:
    """Return the canonical identity of a set of uploaded file bytes.

    Filenames, upload order, batch UUIDs and preview UUIDs are deliberately
    absent. Duplicate hashes collapse because this is a content *set*.
    """
    canonical = sorted({str(value).strip().lower() for value in content_hashes if value})
    if not canonical or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in canonical):
        raise ValueError("every imported source requires a valid SHA-256 content hash")
    payload = "\n".join([_CONTENT_SET_VERSION, *canonical])
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


class ProfileImportManager:
    """Bounded, process-local preview coordinator.

    Preview state is deliberately separate from Global Memory.  Durable writes
    happen only in ``_write_identity`` / ``_write_review`` transactions, keyed by
    a content fingerprint so they survive a restart.
    """

    def __init__(self, *, media_root: Path | None = None, face_engine=None):
        self.media_root = get_media_root(media_root)
        self.face_engine = face_engine or FaceEngineClient()
        workers = max(1, min(int(os.getenv("PROFILE_IMPORT_WORKERS", "2")), 2))
        self.max_file_bytes = int(os.getenv("PROFILE_IMPORT_MAX_FILE_BYTES", str(15 * 1024 * 1024)))
        self.max_pixels = int(os.getenv("PROFILE_IMPORT_MAX_PIXELS", "40000000"))
        self.max_files = max(1, int(os.getenv("PROFILE_IMPORT_MAX_FILES", "500")))
        self.max_batches = max(1, int(os.getenv("PROFILE_IMPORT_MAX_BATCHES", "64")))
        self.batch_ttl_seconds = max(60.0, float(os.getenv("PROFILE_IMPORT_BATCH_TTL_SECONDS", "3600")))
        self.trusted_phone_enrolment = _trusted_phone_enrolment()
        self.batch_duplicate_threshold = IdentityPolicyConfig.from_environment().maximum_similarity
        # The queue carries only (batch_id, source_id) references.  Staged bytes
        # stay on disk and decoded arrays never outlive one worker iteration, so
        # a maximum-size batch is bounded by disk, not RAM.
        capacity = max(workers, int(os.getenv("PROFILE_IMPORT_QUEUE_CAPACITY", "64")))
        self._queue: queue.Queue = queue.Queue(maxsize=capacity)
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._batches: dict[str, dict] = {}
        self._closed = False
        self._workers = [
            threading.Thread(
                target=self._worker_loop,
                name=f"profile-import-{index}",
                daemon=True,
            )
            for index in range(workers)
        ]
        for worker in self._workers:
            worker.start()

    # --- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        for _ in self._workers:
            self._queue.put(None)
        for worker in self._workers:
            worker.join(timeout=5.0)

    def _worker_loop(self) -> None:
        while True:
            task = self._queue.get()
            try:
                if task is None:
                    return
                self._process_item(*task)
            except Exception:  # a worker must never die on one bad image
                pass
            finally:
                self._queue.task_done()

    # --- batch bookkeeping ---------------------------------------------------

    def _batch_directory(self, batch_id: str) -> Path:
        return self.media_root / "_profile_imports" / batch_id

    def _discard_batch_locked(self, batch_id: str) -> None:
        """Drop in-memory state and staged media. Caller holds the lock."""
        self._batches.pop(batch_id, None)
        directory = self._batch_directory(batch_id)
        try:
            if directory.is_dir() and not directory.is_symlink():
                shutil.rmtree(directory, ignore_errors=True)
        except OSError:
            pass

    def sweep(self) -> int:
        """Reclaim evictable batches past their TTL. Returns the count removed."""
        removed = 0
        now = time.monotonic()
        with self._lock:
            expired = [
                key
                for key, value in self._batches.items()
                if value["state"] in _EVICTABLE_BATCH_STATES
                and now - value["monotonic_created_at"] >= self.batch_ttl_seconds
            ]
            for key in expired:
                self._discard_batch_locked(key)
                removed += 1
        return removed

    def _make_room_locked(self) -> None:
        """Evict the oldest evictable batches until there is capacity."""
        while len(self._batches) >= self.max_batches:
            candidates = [
                (value["monotonic_created_at"], key)
                for key, value in self._batches.items()
                if value["state"] in _EVICTABLE_BATCH_STATES
            ]
            if not candidates:
                raise ValueError(
                    "too many profile-import batches are still processing; retry shortly"
                )
            candidates.sort()
            self._discard_batch_locked(candidates[0][1])

    # --- batch creation ------------------------------------------------------

    def create_batch(self, files, owner: str, *, kind: str = "batch") -> dict:
        uploads = list(files)
        if not uploads:
            raise ValueError("at least one image is required")
        if len(uploads) > self.max_files:
            raise ValueError(f"a batch may contain at most {self.max_files} images")
        self.sweep()
        batch_id = uuid.uuid4().hex
        batch_dir = prepare_media_directory(self.media_root, "_profile_imports", batch_id)
        prepare_media_directory(self.media_root, "_profile_imports", batch_id, "uploads")
        prepare_media_directory(self.media_root, "_profile_imports", batch_id, "crops")
        batch = {
            "batch_id": batch_id,
            "owner": owner,
            "kind": kind,
            "state": "processing",
            "cancel_requested": False,
            "created_at": _now(),
            "monotonic_created_at": time.monotonic(),
            "items": [],
            "identities": [],
            "commit_results": {},
        }
        with self._lock:
            if self._closed:
                shutil.rmtree(batch_dir, ignore_errors=True)
                raise ValueError("the import coordinator is shutting down")
            try:
                self._make_room_locked()
            except ValueError:
                shutil.rmtree(batch_dir, ignore_errors=True)
                raise
            self._batches[batch_id] = batch

        try:
            for index, upload in enumerate(uploads):
                batch["items"].append(self._stage_upload(batch, upload, index))
        except BaseException:
            with self._lock:
                self._discard_batch_locked(batch_id)
            raise

        threading.Thread(
            target=self._feed_batch,
            args=(batch_id,),
            name=f"profile-import-feed-{batch_id[:8]}",
            daemon=True,
        ).start()
        with self._condition:
            self._finalize_if_ready(batch)
            self._condition.notify_all()
        return self.public_batch(batch_id, owner)

    def _feed_batch(self, batch_id: str) -> None:
        """Hand staged items to the bounded queue, blocking when it is full.

        Blocking here is what keeps a maximum-size batch fully processed: back
        pressure slows the producer instead of failing valid images.
        """
        with self._lock:
            batch = self._batches.get(batch_id)
            if batch is None:
                return
            pending = [item["source_id"] for item in batch["items"] if item["state"] == "waiting"]
        for source_id in pending:
            while True:
                with self._lock:
                    batch = self._batches.get(batch_id)
                    if batch is None or batch["cancel_requested"] or self._closed:
                        return
                try:
                    self._queue.put((batch_id, source_id), timeout=0.5)
                    break
                except queue.Full:
                    continue

    def _stage_upload(self, batch: dict, upload, index: int) -> dict:
        source_id = uuid.uuid4().hex
        browser_name = str(upload.filename or "")
        leaf = browser_name.replace("\\", "/").rsplit("/", 1)[-1]
        ext = Path(leaf).suffix.lower()
        item = {
            "source_id": source_id,
            "index": index,
            "source_filename": leaf[:255],
            "proposed_name": filename_to_profile_name(browser_name),
            "state": "waiting",
            "error": None,
            "quality": None,
            "bbox": None,
        }
        if ext not in _ALLOWED_EXTENSIONS:
            item.update(state="failed", error="unsupported image type")
            return item
        if "executable" in str(upload.mimetype or "").lower():
            item.update(state="failed", error="executable uploads are not allowed")
            return item
        generated = f"{source_id}{ext}"
        path = self._batch_directory(batch["batch_id"]) / "uploads" / generated
        digest = hashlib.sha256()
        total = 0
        try:
            with path.open("wb") as destination:
                while True:
                    chunk = upload.stream.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > self.max_file_bytes:
                        raise ValueError("image exceeds configured file-size limit")
                    digest.update(chunk)
                    destination.write(chunk)
        except Exception as exc:
            path.unlink(missing_ok=True)
            item.update(state="failed", error=str(exc))
            return item
        item["_upload_path"] = path
        item["content_sha256"] = digest.hexdigest()
        item["original_photo"] = normalize_media_path(path, media_root=self.media_root)
        return item

    # --- processing ----------------------------------------------------------

    def _process_item(self, batch_id: str, source_id: str) -> None:
        with self._condition:
            batch = self._batches.get(batch_id)
            if batch is None:
                return
            item = next((row for row in batch["items"] if row["source_id"] == source_id), None)
            if item is None or item["state"] != "waiting":
                return
            if batch["cancel_requested"]:
                item["state"] = "skipped"
                self._finalize_if_ready(batch)
                self._condition.notify_all()
                return
            item["state"] = "processing"
        try:
            path = item["_upload_path"]
            raw = np.fromfile(str(path), dtype=np.uint8)
            image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            if image is None or image.ndim != 3 or image.size == 0:
                raise ValueError("image could not be decoded")
            height, width = image.shape[:2]
            if height < 1 or width < 1 or height * width > self.max_pixels:
                raise ValueError("decoded image dimensions are invalid")
            faces = self.face_engine.detect(image)
            if len(faces) == 0:
                result = {"state": "no_face", "error": "exactly one face is required"}
            elif len(faces) > 1:
                result = {"state": "multiple_faces", "error": "exactly one face is required"}
            else:
                raw_bbox = faces[0].get("bbox") or []
                if len(raw_bbox) != 4 or not all(math.isfinite(float(v)) for v in raw_bbox):
                    raise ValueError("Face Engine returned an invalid face bounding box")
                x1, y1, x2, y2 = (int(round(float(value))) for value in raw_bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)
                if x2 <= x1 or y2 <= y1:
                    raise ValueError("detected face crop is empty")
                crop = image[y1:y2, x1:x2].copy()
                quality_config = load_quality_filter_config()
                sharpness = _sharpness(crop)
                quality = {
                    "width": int(crop.shape[1]),
                    "height": int(crop.shape[0]),
                    "sharpness": sharpness,
                    "brightness": float(crop.mean()),
                    "detector_confidence": float(
                        faces[0].get("confidence", faces[0].get("score", 0.0))
                    ),
                }
                if (
                    crop.shape[1] < quality_config.face_min_width
                    or crop.shape[0] < quality_config.face_min_height
                    or sharpness < quality_config.face_min_sharpness
                ):
                    result = {
                        "state": "quality_rejected",
                        "error": "face crop did not meet the existing quality boundary",
                        "bbox": [x1, y1, x2, y2],
                        "quality": quality,
                    }
                else:
                    crop_path = self._batch_directory(batch_id) / "crops" / f"{source_id}.jpg"
                    if not cv2.imwrite(str(crop_path), crop):
                        raise ValueError("failed to save face crop")
                    embedding = _normalize_embedding(self.face_engine.embed(crop))
                    result = {
                        "state": "valid",
                        "bbox": [x1, y1, x2, y2],
                        "quality": quality,
                        "face_crop": normalize_media_path(crop_path, media_root=self.media_root),
                        "_crop_path": crop_path,
                        "_embedding": embedding,
                    }
        except Exception as exc:
            result = {"state": "failed", "error": str(exc) or "image processing failed"}
        with self._condition:
            batch = self._batches.get(batch_id)
            if batch is None:
                return
            current = next(
                (row for row in batch["items"] if row["source_id"] == source_id), None
            )
            if current is None:
                return
            current.update(result)
            self._finalize_if_ready(batch)
            self._condition.notify_all()

    def _finalize_if_ready(self, batch: dict) -> None:
        if any(item["state"] not in _TERMINAL_IMAGE_STATES for item in batch["items"]):
            return
        if batch["cancel_requested"]:
            batch["state"] = "cancelled"
            return
        if batch["state"] not in {"processing", "ready"}:
            return
        batch["identities"] = self._build_identities(batch["items"])
        batch["state"] = "ready"

    # --- grouping ------------------------------------------------------------

    def group_by_complete_linkage(self, valid: list[dict]) -> list[list[dict]]:
        """Group photos so that EVERY pair inside a group meets the threshold.

        Transitive (single-linkage) union can chain A-B and B-C into one group
        even when A and C are far apart, which would fuse two different people
        into one Global Memory profile.  Complete linkage forbids that: a photo
        joins a group only when it is close enough to every existing member.

        Photos are processed in stable input order, so grouping is deterministic
        and independent of worker completion order.  If a candidate satisfies
        multiple groups, it joins the first group created by that stable input
        order.
        """
        ordered = sorted(valid, key=lambda row: row["index"])
        groups: list[list[dict]] = []
        for item in ordered:
            for group in groups:
                if all(
                    float(item["_embedding"] @ member["_embedding"])
                    >= self.batch_duplicate_threshold
                    for member in group
                ):
                    group.append(item)
                    break
            else:
                groups.append([item])
        return groups

    def _build_identities(self, items: list[dict]) -> list[dict]:
        valid = [item for item in items if item["state"] == "valid"]
        identities = []
        for grouped in self.group_by_complete_linkage(valid):
            primary = max(
                grouped,
                key=lambda row: (float((row.get("quality") or {}).get("sharpness", 0)), -row["index"]),
            )
            mean = _normalize_embedding(np.mean([row["_embedding"] for row in grouped], axis=0))
            match = self._global_match(mean, len(grouped))
            identities.append({
                "identity_id": uuid.uuid4().hex,
                "source_ids": [row["source_id"] for row in grouped],
                "photos": [self._public_item(row) for row in grouped],
                "primary_source_id": primary["source_id"],
                "proposed_name": primary["proposed_name"],
                "quality": primary["quality"],
                "grouped_photo_count": len(grouped),
                "memory_match": match,
                "similarity": match.get("similarity"),
                "existing_candidate": match.get("candidate"),
                "proposed_action": match["action"],
                "observation_count": len(grouped),
                "identity_source": _TRUSTED_IDENTITY_SOURCE,
                "trusted_enrolment_applied": bool(match.get("trusted_enrolment_applied")),
                "validation_error": None,
                "_embedding": mean,
                "_order": min(row["index"] for row in grouped),
            })
        for item in items:
            if item["state"] == "valid":
                continue
            identities.append({
                "identity_id": uuid.uuid4().hex,
                "source_ids": [item["source_id"]],
                "photos": [self._public_item(item)],
                "primary_source_id": item["source_id"],
                "proposed_name": item["proposed_name"],
                "quality": item.get("quality"),
                "grouped_photo_count": 1,
                "memory_match": {"action": "skip", "reason": item["state"]},
                "similarity": None,
                "existing_candidate": None,
                "proposed_action": "skip",
                "observation_count": 0,
                "identity_source": _TRUSTED_IDENTITY_SOURCE,
                "trusted_enrolment_applied": False,
                "validation_error": item.get("error") or item["state"],
                "_embedding": None,
                "_order": item["index"],
            })
        identities.sort(key=lambda row: row["_order"])
        return identities

    def _global_match(self, embedding: np.ndarray, observation_count: int) -> dict:
        config = IdentityPolicyConfig.from_environment()
        memory = _profile_memory(read_only=True, media_root=self.media_root)
        try:
            candidates = memory.query_by_face(embedding, top_k=2, threshold=0.0)
            top = (
                IdentityCandidate(candidates[0]["person_id"], candidates[0]["similarity"])
                if candidates else None
            )
            second = (
                IdentityCandidate(candidates[1]["person_id"], candidates[1]["similarity"])
                if len(candidates) > 1 else None
            )
            # The true observation count is passed through unchanged; the policy's
            # similarity and margin rules are applied exactly as configured.
            decision = evaluate_identity_decision(
                top_candidate=top,
                second_candidate=second,
                observation_count=observation_count,
                low_confidence=False,
                configuration=config,
            )
            action = {
                IdentityDecisionType.NEW_PERSON: "create_new",
                IdentityDecisionType.ATTACH_EXISTING: "attach_existing",
                IdentityDecisionType.REVIEW_REQUIRED: "review_required",
            }[decision.decision]
            # A supervisor-supplied phone crop has already passed the
            # exactly-one-face and quality gates.  When trusted phone enrolment
            # is enabled, that explicit evidence satisfies the observation-count
            # prerequisite -- and ONLY that prerequisite.  Similarity, margin and
            # low-confidence outcomes are never overridden.
            trusted_applied = False
            if (
                self.trusted_phone_enrolment
                and decision.decision is IdentityDecisionType.REVIEW_REQUIRED
                and decision.reason is IdentityDecisionReason.INSUFFICIENT_FACE_OBSERVATIONS
            ):
                action = "attach_existing"
                trusted_applied = True
            candidate = None
            if candidates:
                person = memory.get_person(candidates[0]["person_id"]) or {}
                candidate = {
                    "person_id": candidates[0]["person_id"],
                    "name": candidates[0]["name"],
                    "profile_image": person.get("profile_image"),
                    "profile_image_url": _media_url(person.get("profile_image")),
                }
            return {
                "action": action,
                "reason": decision.reason.value,
                "policy_decision": decision.decision.value,
                "similarity": decision.top_similarity,
                "second_similarity": decision.second_similarity,
                "margin": decision.margin,
                "observation_count": observation_count,
                "minimum_face_observations": config.minimum_face_observations,
                "identity_source": _TRUSTED_IDENTITY_SOURCE,
                "trusted_enrolment_enabled": self.trusted_phone_enrolment,
                "trusted_enrolment_applied": trusted_applied,
                "candidate": candidate,
            }
        finally:
            memory.close()

    # --- preview access ------------------------------------------------------

    def wait_ready(self, batch_id: str, owner: str, timeout: float = 120.0) -> dict:
        deadline = time.monotonic() + timeout
        with self._condition:
            while True:
                batch = self._owned(batch_id, owner)
                if batch["state"] != "processing":
                    return self.public_batch(batch_id, owner)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("profile preview timed out")
                self._condition.wait(min(remaining, 1.0))

    def cancel(self, batch_id: str, owner: str) -> dict:
        with self._condition:
            batch = self._owned(batch_id, owner)
            if batch["state"] in {"committed", "committing"}:
                raise ValueError("a committing or committed batch cannot be cancelled")
            batch["cancel_requested"] = True
            for item in batch["items"]:
                if item["state"] == "waiting":
                    item["state"] = "skipped"
            self._finalize_if_ready(batch)
            batch["state"] = "cancelled"
            snapshot = self.public_batch(batch_id, owner)
            self._condition.notify_all()
        # Drain outside the lock so a worker mid-decode cannot race the removal.
        self._wait_for_quiet_batch(batch_id)
        with self._lock:
            self._discard_batch_locked(batch_id)
        snapshot["state"] = "cancelled"
        return snapshot

    def _wait_for_quiet_batch(self, batch_id: str, timeout: float = 10.0) -> None:
        deadline = time.monotonic() + timeout
        with self._condition:
            while time.monotonic() < deadline:
                batch = self._batches.get(batch_id)
                if batch is None:
                    return
                if not any(item["state"] == "processing" for item in batch["items"]):
                    return
                self._condition.wait(0.1)

    def discard(self, batch_id: str) -> None:
        """Drop a preview and its staged media (used for failed single previews)."""
        self._wait_for_quiet_batch(batch_id, timeout=2.0)
        with self._lock:
            self._discard_batch_locked(batch_id)

    # --- commit --------------------------------------------------------------

    def commit(self, batch_id: str, owner: str, changes: list[dict]) -> dict:
        with self._condition:
            batch = self._owned(batch_id, owner)
            # A concurrent duplicate request waits for the in-flight commit and
            # then replays its stored results instead of failing with a
            # misleading batch-state error.
            deadline = time.monotonic() + 120.0
            while batch["state"] == "committing" and time.monotonic() < deadline:
                self._condition.wait(0.2)
                batch = self._owned(batch_id, owner)
            if batch["state"] == "committing":
                raise TimeoutError("a concurrent commit is still running")
            if batch["state"] not in {"ready", "committed"}:
                raise ValueError("batch must finish processing before commit")
            batch["state"] = "committing"
        try:
            by_id = {str(row.get("identity_id")): row for row in changes if isinstance(row, dict)}
            results = []
            for identity in batch["identities"]:
                override = by_id.get(identity["identity_id"], {})
                try:
                    result = self._commit_identity(batch, identity, override)
                except OperationConflict:
                    raise
                except Exception as exc:
                    result = {
                        "identity_id": identity["identity_id"],
                        "status": "failed",
                        "error": str(exc) or "identity save failed",
                    }
                results.append(result)
        finally:
            with self._condition:
                batch["state"] = "committed"
                self._condition.notify_all()
        with self._lock:
            batch["commit_results"] = {row["identity_id"]: row for row in results}
        summary = {
            "profiles_created": sum(row["status"] == "created" for row in results),
            "profiles_updated": sum(row["status"] == "updated" for row in results),
            "items_sent_to_review": sum(row["status"] == "review_required" for row in results),
            "skipped_items": sum(row["status"] == "skipped" for row in results),
            "failed_items": sum(row["status"] == "failed" for row in results),
        }
        return {"batch_id": batch_id, "state": "committed", "summary": summary, "results": results}

    def _commit_identity(self, batch: dict, identity: dict, override: dict) -> dict:
        identity_id = identity["identity_id"]
        previous = batch["commit_results"].get(identity_id)
        action = "skip" if override.get("skip") else str(
            override.get("action") or identity["proposed_action"]
        )
        if action not in _ACTION_VALUES:
            raise ValueError("invalid proposed action")
        if previous and previous.get("status") == "skipped" and action == "skip":
            return {**previous, "idempotent_replay": True}
        if identity["validation_error"] or action == "skip":
            return {"identity_id": identity_id, "status": "skipped"}
        name = str(override.get("name") or identity["proposed_name"]).strip()
        if not name or len(name) > 100:
            raise ValueError("name is required and must be at most 100 characters")
        existing_person_id = str(
            override.get("existing_person_id")
            or ((identity.get("existing_candidate") or {}).get("person_id") or "")
        ).strip()
        primary_source_id = str(
            override.get("primary_source_id") or identity["primary_source_id"]
        )
        if primary_source_id not in identity["source_ids"]:
            raise ValueError("primary photo must belong to the grouped identity")
        selected_items = [
            next(row for row in batch["items"] if row["source_id"] == source_id)
            for source_id in identity["source_ids"]
        ]
        notes = str(override.get("notes") or identity.get("notes") or "")[:4000]

        if action == "review_required":
            return self._write_review(
                batch=batch,
                identity=identity,
                items=selected_items,
                name=name,
            )
        if action == "attach_existing" and not existing_person_id:
            raise ValueError("an existing person must be selected")
        # Renaming an existing profile is an explicit, separate consent.  The
        # mere presence of a name field never grants it.
        update_existing_name = bool(override.get("update_existing_name"))
        return self._write_identity(
            batch=batch,
            identity=identity,
            items=selected_items,
            name=name,
            update_existing_name=update_existing_name,
            notes=notes,
            action=action,
            existing_person_id=existing_person_id or None,
            primary_source_id=primary_source_id,
        )

    def _identity_fingerprint(
        self,
        *,
        action: str,
        items: list[dict],
        existing_person_id: str | None,
        name: str,
        update_existing_name: bool,
    ) -> str:
        hashes = [str(item.get("content_sha256") or item["source_id"]) for item in items]
        if action == "create_new":
            scope, name_component = "new", name
        elif action == "review_required":
            scope = f"review:{existing_person_id or 'none'}"
            name_component = name
        else:
            scope = f"person:{existing_person_id}"
            name_component = f"rename:{name}" if update_existing_name else ""
        return commit_fingerprint(
            action=action,
            scope=scope,
            content_hashes=hashes,
            name_component=name_component,
        )

    @staticmethod
    def _content_set_key(items: list[dict]) -> str:
        return content_set_fingerprint(
            [str(item.get("content_sha256") or "") for item in items]
        )

    @staticmethod
    def _claim_content_intent(
        connection: sqlite3.Connection,
        *,
        content_set_key: str,
        action: str,
        target_person_id: str | None,
        approved_name_component: str,
    ) -> dict | None:
        """Claim bytes for one semantic action, or replay/conflict atomically.

        The approved name is retained for audit, but does not turn a later name
        edit into a new operation: same bytes + same action + same target always
        replays the first approved result.
        """
        target = str(target_person_id or "").strip() or None
        row = connection.execute(
            "SELECT * FROM profile_import_content_sets WHERE content_set_key=?",
            (content_set_key,),
        ).fetchone()
        if row is None:
            now = _now()
            connection.execute(
                """
                INSERT INTO profile_import_content_sets(
                    content_set_key, semantic_action, target_person_id,
                    approved_name_component, durable_result_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, '{}', ?, ?)
                """,
                (
                    content_set_key, action, target,
                    str(approved_name_component or ""), now, now,
                ),
            )
            return None

        stored_target = str(row["target_person_id"] or "").strip() or None
        if row["semantic_action"] != action or stored_target != target:
            raise OperationConflict(
                "operation_conflict: uploaded content was already committed "
                f"as {row['semantic_action']} for {stored_target or 'no target'}"
            )
        try:
            durable = json.loads(row["durable_result_json"] or "{}")
        except (TypeError, ValueError):
            durable = {}
        if durable:
            durable["idempotent_replay"] = True
            return durable
        return None

    @staticmethod
    def _store_content_result(
        connection: sqlite3.Connection,
        content_set_key: str,
        result: dict,
    ) -> None:
        connection.execute(
            "UPDATE profile_import_content_sets "
            "SET durable_result_json=?, updated_at=? WHERE content_set_key=?",
            (
                json.dumps({key: value for key, value in result.items() if key != "identity_id"}),
                _now(),
                content_set_key,
            ),
        )

    @staticmethod
    def _replay(connection: sqlite3.Connection, commit_key: str) -> dict | None:
        row = connection.execute(
            "SELECT person_id, outcome, result_json FROM profile_import_commits "
            "WHERE commit_key=?",
            (commit_key,),
        ).fetchone()
        if row is None:
            return None
        try:
            stored = json.loads(row["result_json"]) or {}
        except (TypeError, ValueError):
            stored = {}
        stored.setdefault("person_id", row["person_id"])
        stored.setdefault(
            "status",
            {"created": "created", "updated": "updated"}.get(row["outcome"], "review_required"),
        )
        stored["idempotent_replay"] = True
        return stored

    def _write_identity(
        self,
        *,
        batch: dict,
        identity: dict,
        items: list[dict],
        name: str,
        update_existing_name: bool,
        notes: str,
        action: str,
        existing_person_id: str | None,
        primary_source_id: str,
    ) -> dict:
        commit_key = self._identity_fingerprint(
            action=action,
            items=items,
            existing_person_id=existing_person_id,
            name=name,
            update_existing_name=update_existing_name,
        )
        content_set_key = self._content_set_key(items)
        memory = _profile_memory(media_root=self.media_root)
        connection = memory._conn
        created: list[Path] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            replay = self._replay(connection, commit_key)
            if replay is not None:
                connection.execute("COMMIT")
                return {**replay, "identity_id": identity["identity_id"]}

            content_hashes = [str(item.get("content_sha256") or "") for item in items]
            durable = self._claim_content_intent(
                connection,
                content_set_key=content_set_key,
                action=action,
                target_person_id=existing_person_id if action == "attach_existing" else None,
                approved_name_component=name,
            )
            if durable is not None:
                connection.execute("COMMIT")
                return {**durable, "identity_id": identity["identity_id"]}

            # Content ownership is never treated as permission to change intent
            # or target.  It is only a safety check for legacy evidence that
            # predates the durable content-set ledger.  A partially overlapping
            # attach to the same target may still append genuinely new photos.
            owned: dict[str, set[str]] = {}
            for value in content_hashes:
                rows = connection.execute(
                    "SELECT DISTINCT person_id FROM face_photo_sources "
                    "WHERE content_sha256=?",
                    (value,),
                ).fetchall()
                if rows:
                    owned[value] = {str(row["person_id"]) for row in rows}
            if owned:
                wrong_target = action == "create_new" or any(
                    set(owners) != {str(existing_person_id)} for owners in owned.values()
                )
                if wrong_target:
                    raise OperationConflict(
                        "operation_conflict: uploaded content is already identity "
                        "evidence and has no matching durable intent"
                    )

            if action == "create_new":
                person_id, _default_name = memory._next_person_id()
                outcome = "created"
                old_count = 0
            else:
                person_id = str(existing_person_id)
                person = connection.execute(
                    "SELECT embedding, embedding_count FROM persons "
                    "WHERE person_id=? AND is_active=1",
                    (person_id,),
                ).fetchone()
                if person is None:
                    raise ValueError("selected existing person does not exist or is inactive")
                outcome = "updated"
                old_count = int(person["embedding_count"])

            # Partial overlap: attach only the photos this person does not
            # already hold, so a retry that adds one new image cannot duplicate
            # the ones already stored.
            existing_hashes = {
                row["content_sha256"]
                for row in connection.execute(
                    "SELECT content_sha256 FROM face_photo_sources "
                    "WHERE person_id=? AND content_sha256 IS NOT NULL",
                    (person_id,),
                ).fetchall()
            }
            retained_items = [
                item for item in items
                if str(item.get("content_sha256") or "") not in existing_hashes
            ]
            if not retained_items:
                result = {
                    "identity_id": identity["identity_id"],
                    "person_id": person_id,
                    "status": "updated",
                    "duplicate_evidence_skipped": True,
                    "commit_key": commit_key,
                    "content_set_key": content_set_key,
                    "idempotent_replay": True,
                }
                self._store_content_result(connection, content_set_key, result)
                connection.execute(
                    """
                    INSERT INTO profile_import_commits(
                        commit_key, content_set_key, batch_id, identity_id,
                        person_id, outcome, created_at, result_json
                    ) VALUES (?, ?, ?, ?, ?, 'updated', ?, ?)
                    """,
                    (
                        commit_key, content_set_key, batch["batch_id"],
                        identity["identity_id"], person_id, _now(),
                        json.dumps({
                            key: value for key, value in result.items()
                            if key != "identity_id"
                        }),
                    ),
                )
                connection.execute("COMMIT")
                return result

            items = retained_items
            observation_count = len(retained_items)
            if primary_source_id not in {item["source_id"] for item in items}:
                primary_source_id = items[0]["source_id"]

            canonical = self._materialize_phone_files(person_id, items, created)
            embedding = _normalize_embedding(
                np.mean([item["_embedding"] for item in retained_items], axis=0)
            )
            created_at = _now()

            # A supervisor-selected primary is durable state: an automatic
            # attach never replaces it.
            supervisor = connection.execute(
                "SELECT face_crop_path FROM face_photo_sources "
                "WHERE person_id=? AND is_supervisor_selected=1",
                (person_id,),
            ).fetchone()
            keep_supervisor = supervisor is not None and _media_file_exists(
                supervisor["face_crop_path"], self.media_root
            )
            new_primary_path = canonical[primary_source_id]["crop"]

            if action == "create_new":
                connection.execute(
                    """
                    INSERT INTO persons(
                        person_id, name, embedding, embedding_count, enrolled_at,
                        updated_at, cameras, profile_image, profile_image_source,
                        notes, identity_source, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, '[]', ?, 'phone', ?, 'phone', 1)
                    """,
                    (
                        person_id, name, embedding.astype(np.float32).tobytes(),
                        observation_count, created_at, created_at, new_primary_path, notes,
                    ),
                )
            else:
                stored = connection.execute(
                    "SELECT embedding, embedding_count FROM persons WHERE person_id=?",
                    (person_id,),
                ).fetchone()
                old = np.frombuffer(stored["embedding"], dtype=np.float32).copy()
                combined = _normalize_embedding(
                    old * old_count + embedding * observation_count
                )
                connection.execute(
                    """
                    UPDATE persons
                       SET embedding=?, embedding_count=?, updated_at=?,
                           profile_image=CASE WHEN ? THEN profile_image ELSE ? END,
                           profile_image_source='phone',
                           name=CASE WHEN ? THEN ? ELSE name END,
                           notes=CASE WHEN ? <> '' THEN ? ELSE notes END,
                           identity_source=CASE
                               WHEN identity_source='video' THEN 'phone+video'
                               ELSE identity_source
                           END
                     WHERE person_id=?
                    """,
                    (
                        combined.astype(np.float32).tobytes(),
                        old_count + observation_count,
                        created_at,
                        int(keep_supervisor), new_primary_path,
                        int(update_existing_name), name,
                        notes, notes,
                        person_id,
                    ),
                )
            if not keep_supervisor:
                connection.execute(
                    "UPDATE face_photo_sources SET is_primary=0 WHERE person_id=?",
                    (person_id,),
                )
            for item in items:
                paths = canonical[item["source_id"]]
                is_primary = int(
                    not keep_supervisor and item["source_id"] == primary_source_id
                )
                connection.execute(
                    """
                    INSERT INTO face_photo_sources(
                        source_id, person_id, original_image_path, face_crop_path,
                        face_bbox, quality_info, embedding, created_at,
                        source_filename, import_batch_id, is_primary,
                        is_supervisor_selected, content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        item["source_id"], person_id, paths["original"], paths["crop"],
                        json.dumps(item["bbox"]), json.dumps(item["quality"] or {}),
                        item["_embedding"].astype(np.float32).tobytes(), created_at,
                        item["source_filename"], batch["batch_id"], is_primary,
                        item.get("content_sha256"),
                    ),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO identity_evidence(
                        person_id, evidence_key, crop_type, canonical_path,
                        embedding_applied, observation_weight, created_at
                    ) VALUES (?, ?, 'face', ?, 1, 1, ?)
                    """,
                    (person_id, f"phone:{item['source_id']}", paths["crop"], created_at),
                )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO person_gallery(
                        person_id, crop_type, path, sharpness, session_date,
                        video_source, width, height
                    ) VALUES (?, 'face', ?, ?, ?, 'phone_photo', ?, ?)
                    """,
                    (
                        person_id, paths["crop"],
                        float((item["quality"] or {}).get("sharpness", 0)),
                        date.today().isoformat(),
                        int((item["quality"] or {}).get("width", 0)),
                        int((item["quality"] or {}).get("height", 0)),
                    ),
                )
            connection.execute(
                """
                INSERT INTO recognition_log(
                    person_id, event_type, similarity, embedding_count_before,
                    embedding_count_after, video_sources, best_face_crop, ts
                ) VALUES (?, ?, NULL, ?, ?, '[]', ?, ?)
                """,
                (
                    person_id,
                    "phone_profile_created" if outcome == "created" else "phone_evidence_appended",
                    None if outcome == "created" else old_count,
                    old_count + observation_count,
                    new_primary_path,
                    created_at,
                ),
            )
            result = {
                "identity_id": identity["identity_id"],
                "person_id": person_id,
                "status": outcome,
                "primary_preserved": bool(keep_supervisor),
                "renamed_existing": bool(update_existing_name and action != "create_new"),
                "commit_key": commit_key,
                "content_set_key": content_set_key,
                "idempotent_replay": False,
            }
            self._store_content_result(connection, content_set_key, result)
            connection.execute(
                """
                INSERT INTO profile_import_commits(
                    commit_key, content_set_key, batch_id, identity_id, person_id, outcome,
                    created_at, result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    commit_key, content_set_key, batch["batch_id"], identity["identity_id"],
                    person_id, outcome, created_at,
                    json.dumps({k: v for k, v in result.items() if k != "identity_id"}),
                ),
            )
            connection.execute("COMMIT")
            return result
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            discard_created_media(created)
            raise
        finally:
            memory.close()

    def _write_review(
        self,
        *,
        batch: dict,
        identity: dict,
        items: list[dict],
        name: str,
    ) -> dict:
        """Persist an uncertain import so it survives eviction and restart."""
        candidate = (identity.get("existing_candidate") or {}).get("person_id") or None
        match = identity.get("memory_match") or {}
        commit_key = self._identity_fingerprint(
            action="review_required",
            items=items,
            existing_person_id=candidate,
            name=name,
            update_existing_name=False,
        )
        content_set_key = self._content_set_key(items)
        memory = _profile_memory(media_root=self.media_root)
        connection = memory._conn
        created: list[Path] = []
        try:
            connection.execute("BEGIN IMMEDIATE")
            durable = self._claim_content_intent(
                connection,
                content_set_key=content_set_key,
                action="review_required",
                target_person_id=candidate,
                approved_name_component=name,
            )
            if durable is not None:
                connection.execute("COMMIT")
                return {**durable, "identity_id": identity["identity_id"]}
            replay = self._replay(connection, commit_key)
            if replay is not None:
                self._store_content_result(connection, content_set_key, replay)
                connection.execute("COMMIT")
                return {**replay, "identity_id": identity["identity_id"]}

            # Durable, controlled evidence outside any person directory: no
            # synthetic person is created for an unresolved identity.
            directory = prepare_media_directory(
                self.media_root, "_profile_reviews", commit_key[:32]
            )
            evidence = []
            for item in items:
                crop_destination = directory / f"crop_{item['source_id']}.jpg"
                original_destination = directory / (
                    f"original_{item['source_id']}{Path(item['_upload_path']).suffix.lower()}"
                )
                safe_copy_into(item["_crop_path"], crop_destination, created)
                safe_copy_into(item["_upload_path"], original_destination, created)
                evidence.append({
                    "source_id": item["source_id"],
                    "source_filename": item["source_filename"],
                    "face_crop_path": normalize_media_path(
                        crop_destination, media_root=self.media_root
                    ),
                    "original_image_path": normalize_media_path(
                        original_destination, media_root=self.media_root
                    ),
                    "quality": item.get("quality") or {},
                    "face_bbox": item.get("bbox"),
                    "content_sha256": item.get("content_sha256"),
                })
            created_at = _now()
            connection.execute(
                """
                INSERT INTO profile_import_reviews(
                    review_key, content_set_key, candidate_person_id,
                    proposed_name, similarity,
                    second_similarity, margin, reason, observation_count,
                    identity_source, embedding, evidence_json, status, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    commit_key, content_set_key, candidate, name, match.get("similarity"),
                    match.get("second_similarity"), match.get("margin"),
                    match.get("reason"), int(identity.get("observation_count") or len(items)),
                    _TRUSTED_IDENTITY_SOURCE,
                    _normalize_embedding(identity["_embedding"]).astype(np.float32).tobytes(),
                    json.dumps(evidence), created_at,
                ),
            )
            connection.executemany(
                "INSERT INTO profile_import_review_evidence("
                "review_key, source_id, embedding) VALUES (?, ?, ?)",
                [
                    (
                        commit_key,
                        item["source_id"],
                        _normalize_embedding(item["_embedding"]).astype(np.float32).tobytes(),
                    )
                    for item in items
                ],
            )
            result = {
                "identity_id": identity["identity_id"],
                "person_id": candidate or "",
                "status": "review_required",
                "review_key": commit_key,
                "candidate_person_id": candidate,
                "commit_key": commit_key,
                "content_set_key": content_set_key,
                "idempotent_replay": False,
            }
            self._store_content_result(connection, content_set_key, result)
            connection.execute(
                """
                INSERT INTO profile_import_commits(
                    commit_key, content_set_key, batch_id, identity_id, person_id, outcome,
                    created_at, result_json
                ) VALUES (?, ?, ?, ?, ?, 'review', ?, ?)
                """,
                (
                    commit_key, content_set_key, batch["batch_id"], identity["identity_id"],
                    candidate or "", created_at,
                    json.dumps({k: v for k, v in result.items() if k != "identity_id"}),
                ),
            )
            connection.execute("COMMIT")
            return result
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            discard_created_media(created)
            raise
        finally:
            memory.close()

    def _materialize_phone_files(
        self, person_id: str, items: list[dict], created: list[Path]
    ) -> dict[str, dict[str, str]]:
        if not _SAFE_PERSON_ID.fullmatch(person_id):
            raise ValueError("person ID is not safe for media storage")
        originals = prepare_media_directory(self.media_root, person_id, "phone_originals")
        crops = prepare_media_directory(self.media_root, person_id, "face_crops")
        result = {}
        for item in items:
            source_id = item["source_id"]
            original_ext = Path(item["_upload_path"]).suffix.lower()
            original_destination = originals / f"phone_{source_id}{original_ext}"
            crop_destination = crops / f"phone_{source_id}.jpg"
            safe_copy_into(item["_upload_path"], original_destination, created)
            safe_copy_into(item["_crop_path"], crop_destination, created)
            result[source_id] = {
                "original": normalize_media_path(original_destination, media_root=self.media_root),
                "crop": normalize_media_path(crop_destination, media_root=self.media_root),
            }
        return result

    # --- public view ---------------------------------------------------------

    def _owned(self, batch_id: str, owner: str) -> dict:
        batch = self._batches.get(str(batch_id))
        if batch is None:
            raise KeyError("batch not found")
        if batch["owner"] != owner:
            raise PermissionError("batch does not belong to this supervisor")
        return batch

    @staticmethod
    def _public_item(item: dict) -> dict:
        return {
            key: value
            for key, value in item.items()
            if not key.startswith("_")
        } | {
            "original_photo_url": _media_url(item.get("original_photo")),
            "face_crop_url": _media_url(item.get("face_crop")),
        }

    def public_batch(self, batch_id: str, owner: str) -> dict:
        with self._lock:
            batch = self._owned(batch_id, owner)
            items = [self._public_item(item) for item in batch["items"]]
            identities = [
                {key: value for key, value in identity.items() if not key.startswith("_")}
                for identity in batch["identities"]
            ]
            processed = sum(item["state"] in _TERMINAL_IMAGE_STATES for item in batch["items"])
            summary = {
                "processed": processed,
                "total": len(items),
                "valid": sum(item["state"] == "valid" for item in batch["items"]),
                "existing_matches": sum(
                    row["proposed_action"] == "attach_existing" for row in batch["identities"]
                ),
                "new_profiles": sum(
                    row["proposed_action"] == "create_new" for row in batch["identities"]
                ),
                "review_required": sum(
                    row["proposed_action"] == "review_required" for row in batch["identities"]
                ),
                "failed": sum(
                    item["state"] not in {"valid", "waiting", "processing"}
                    for item in batch["items"]
                ),
            }
            return {
                "batch_id": batch_id,
                "state": batch["state"],
                "created_at": batch["created_at"],
                "progress": summary,
                "items": items,
                "identities": identities,
                "can_commit": batch["state"] in {"ready", "committed"},
            }


# --- merge helpers (all run inside the caller-owned transaction) -------------

def _merge_move_phone_sources(connection, source: str, target: str) -> None:
    connection.execute(
        "UPDATE face_photo_sources SET is_primary=0, is_supervisor_selected=0 "
        "WHERE person_id=?",
        (source,),
    )
    connection.execute(
        "UPDATE face_photo_sources SET person_id=? WHERE person_id=?",
        (target, source),
    )


def _merge_move_identity_evidence(connection, source: str, target: str) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO identity_evidence(
            person_id, evidence_key, crop_type, canonical_path,
            embedding_applied, observation_weight, created_at
        )
        SELECT ?, evidence_key, crop_type, canonical_path,
               embedding_applied, observation_weight, created_at
          FROM identity_evidence WHERE person_id=?
        """,
        (target, source),
    )
    connection.execute("DELETE FROM identity_evidence WHERE person_id=?", (source,))


def _appearance_json_list(value: Any) -> list:
    try:
        parsed = json.loads(value or "[]")
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _appearance_union(first: list, second: list) -> list:
    """Deduplicate JSON-compatible media/source values in stable order."""
    result = []
    seen = set()
    for value in [*first, *second]:
        marker = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if marker not in seen:
            seen.add(marker)
            result.append(value)
    return result


def _meaningful_appearance_value(value: Any) -> bool:
    return value is not None and str(value).strip() not in {"", "unknown", "unavailable"}


def _merge_same_date_appearance(
    connection: sqlite3.Connection,
    source_row: sqlite3.Row,
    target_row: sqlite3.Row,
) -> None:
    """Reconcile one date collision without leaving source-owned state.

    Appearance rows have no per-row update timestamp.  The row with more
    non-empty descriptive/color fields is therefore authoritative for
    conflicting values; ties preserve the established target.  Empty fields are
    filled from the other row.  Body/ReID crop references and video sources use
    the existing append-and-deduplicate storage policy, target values first.
    """
    scalar_columns = (
        "top", "bottom", "shoes", "full_description", "top_color", "bottom_color"
    )
    source_score = sum(
        _meaningful_appearance_value(source_row[column]) for column in scalar_columns
    )
    target_score = sum(
        _meaningful_appearance_value(target_row[column]) for column in scalar_columns
    )
    preferred, fallback = (
        (source_row, target_row) if source_score > target_score else (target_row, source_row)
    )
    merged = {
        column: (
            preferred[column]
            if _meaningful_appearance_value(preferred[column])
            else fallback[column]
        )
        for column in scalar_columns
    }
    status = str(preferred["clothing_status"] or "not_attempted")
    fallback_status = str(fallback["clothing_status"] or "not_attempted")
    if status == "not_attempted" and fallback_status != "not_attempted":
        status = fallback_status
    body = _appearance_union(
        _appearance_json_list(target_row["best_body_crops"]),
        _appearance_json_list(source_row["best_body_crops"]),
    )
    videos = _appearance_union(
        _appearance_json_list(target_row["video_sources"]),
        _appearance_json_list(source_row["video_sources"]),
    )
    connection.execute(
        """
        UPDATE appearances
           SET top=?, bottom=?, shoes=?, full_description=?,
               top_color=?, bottom_color=?, clothing_status=?,
               best_body_crops=?, video_sources=?
         WHERE id=?
        """,
        (
            merged["top"], merged["bottom"], merged["shoes"],
            merged["full_description"], merged["top_color"],
            merged["bottom_color"], status,
            json.dumps(body), json.dumps(videos), target_row["id"],
        ),
    )
    connection.execute("DELETE FROM appearances WHERE id=?", (source_row["id"],))


def _merge_reconcile_appearances(
    connection: sqlite3.Connection,
    source: str,
    target: str,
) -> None:
    for source_row in connection.execute(
        "SELECT * FROM appearances WHERE person_id=? ORDER BY date, id",
        (source,),
    ).fetchall():
        target_row = connection.execute(
            "SELECT * FROM appearances WHERE person_id=? AND date=?",
            (target, source_row["date"]),
        ).fetchone()
        if target_row is None:
            connection.execute(
                "UPDATE appearances SET person_id=? WHERE id=?",
                (target, source_row["id"]),
            )
        else:
            _merge_same_date_appearance(connection, source_row, target_row)
    remaining = connection.execute(
        "SELECT COUNT(*) FROM appearances WHERE person_id=?", (source,)
    ).fetchone()[0]
    if remaining:
        raise sqlite3.IntegrityError("source appearances remained after reconciliation")


def _merge_move_gallery(connection, source: str, target: str) -> None:
    connection.execute(
        """
        INSERT OR IGNORE INTO person_gallery(
            person_id, crop_type, path, sharpness, session_date,
            video_source, width, height
        )
        SELECT ?, crop_type, path, sharpness, session_date,
               video_source, width, height
          FROM person_gallery WHERE person_id=?
        """,
        (target, source),
    )
    connection.execute("DELETE FROM person_gallery WHERE person_id=?", (source,))
    connection.execute(
        "UPDATE recognition_log SET person_id=? WHERE person_id=?", (target, source)
    )
    _merge_reconcile_appearances(connection, source, target)


def _merge_reconcile_primary(connection, target: str, media_root: Path) -> None:
    """Keep a valid supervisor choice, otherwise apply the documented priority."""
    resolved = resolve_profile_image(connection, target, media_root)
    if resolved["origin"] != "supervisor_phone_crop":
        connection.execute(
            "UPDATE face_photo_sources SET is_primary=0 WHERE person_id=?", (target,)
        )
        if resolved["origin"] == "newest_phone_crop":
            connection.execute(
                "UPDATE face_photo_sources SET is_primary=1 "
                "WHERE source_id = (SELECT source_id FROM face_photo_sources "
                "                    WHERE person_id=? AND face_crop_path=? LIMIT 1)",
                (target, resolved["path"]),
            )
    if resolved["path"]:
        connection.execute(
            "UPDATE persons SET profile_image=?, profile_image_source=? WHERE person_id=?",
            (
                resolved["path"],
                "phone" if resolved["origin"].endswith("phone_crop") else "auto",
                target,
            ),
        )


def merge_phone_evidence(connection, source: str, target: str, media_root: Path) -> None:
    _merge_move_phone_sources(connection, source, target)
    _merge_move_identity_evidence(connection, source, target)
    _merge_move_gallery(connection, source, target)
    _merge_reconcile_primary(connection, target, media_root)


# --- HTTP surface ------------------------------------------------------------

def _manager() -> ProfileImportManager:
    manager = current_app.config.get("PROFILE_IMPORT_MANAGER")
    if manager is None:
        manager = ProfileImportManager()
        current_app.config["PROFILE_IMPORT_MANAGER"] = manager
    return manager


def _json_error(exc: Exception):
    if isinstance(exc, OperationConflict):
        return jsonify({"error": str(exc), "code": "operation_conflict"}), 409
    if isinstance(exc, KeyError):
        return jsonify({"error": str(exc.args[0])}), 404
    if isinstance(exc, PermissionError):
        return jsonify({"error": str(exc)}), 403
    if isinstance(exc, (ValueError, TimeoutError)):
        return jsonify({"error": str(exc)}), 400
    return jsonify({"error": "profile management operation failed"}), 500


@bp.post("/api/profiles/import/preview")
def import_preview():
    try:
        files = request.files.getlist("images") or request.files.getlist("files")
        return jsonify(_manager().create_batch(files, _supervisor_id())), 202
    except Exception as exc:
        return _json_error(exc)


@bp.get("/api/profiles/import/<batch_id>/status")
def import_status(batch_id: str):
    try:
        return jsonify(_manager().public_batch(batch_id, _supervisor_id()))
    except Exception as exc:
        return _json_error(exc)


@bp.post("/api/profiles/import/<batch_id>/commit")
def import_commit(batch_id: str):
    try:
        body = request.get_json(silent=True) or {}
        identities = body.get("identities") or []
        if not isinstance(identities, list):
            raise ValueError("identities must be an array")
        return jsonify(_manager().commit(batch_id, _supervisor_id(), identities))
    except Exception as exc:
        return _json_error(exc)


@bp.post("/api/profiles/import/<batch_id>/cancel")
def import_cancel(batch_id: str):
    try:
        return jsonify(_manager().cancel(batch_id, _supervisor_id()))
    except Exception as exc:
        return _json_error(exc)


def _single_preview(kind: str):
    files = request.files.getlist("photo")
    if len(files) != 1:
        raise ValueError("exactly one phone photo is required")
    manager = _manager()
    batch = manager.create_batch(files, _supervisor_id(), kind=kind)
    try:
        ready = manager.wait_ready(batch["batch_id"], _supervisor_id())
    except BaseException:
        manager.discard(batch["batch_id"])
        raise
    ready["preview_id"] = ready["batch_id"]
    return ready


@bp.post("/api/profiles/manual/preview")
def manual_preview():
    try:
        # Validate before anything is staged so a rejected request leaves no
        # orphan batch or staged media behind.
        name = str(request.form.get("name") or "").strip()
        if not name:
            raise ValueError("name is required")
        if len(name) > 100:
            raise ValueError("name must be at most 100 characters")
        notes = str(request.form.get("notes") or "")[:4000]
        result = _single_preview("manual")
        manager = _manager()
        with manager._lock:
            batch = manager._owned(result["batch_id"], _supervisor_id())
            if batch["identities"]:
                batch["identities"][0]["proposed_name"] = name
                batch["identities"][0]["notes"] = notes
            result = manager.public_batch(result["batch_id"], _supervisor_id())
            result["preview_id"] = result["batch_id"]
        return jsonify(result)
    except Exception as exc:
        return _json_error(exc)


@bp.post("/api/profiles/manual/commit")
def manual_commit():
    return _commit_single_preview()


def _commit_single_preview(body: dict | None = None):
    try:
        body = body if body is not None else (request.get_json(silent=True) or {})
        preview_id = str(body.get("preview_id") or "")
        identity = body.get("identity") or {}
        batch = _manager().public_batch(preview_id, _supervisor_id())
        if len(batch["identities"]) != 1:
            raise ValueError("preview does not contain one identity")
        identity = {"identity_id": batch["identities"][0]["identity_id"], **identity}
        return jsonify(_manager().commit(preview_id, _supervisor_id(), [identity]))
    except Exception as exc:
        return _json_error(exc)


def _pending_import_reviews(connection, media_root: Path, person_id: str | None = None) -> list[dict]:
    if person_id is None:
        rows = connection.execute(
            "SELECT * FROM profile_import_reviews WHERE status='pending' "
            "ORDER BY created_at ASC, rowid ASC LIMIT 200"
        ).fetchall()
    else:
        rows = connection.execute(
            "SELECT * FROM profile_import_reviews "
            "WHERE status='pending' AND candidate_person_id=? "
            "ORDER BY created_at ASC, rowid ASC LIMIT 200",
            (person_id,),
        ).fetchall()
    result = []
    for row in rows:
        try:
            evidence = json.loads(row["evidence_json"]) or []
        except (TypeError, ValueError):
            evidence = []
        for entry in evidence:
            entry["face_crop_url"] = _media_url(entry.get("face_crop_path"))
            entry.pop("content_sha256", None)
        result.append({
            "kind": "phone_import",
            "review_key": row["review_key"],
            "candidate_person_id": row["candidate_person_id"],
            "proposed_name": row["proposed_name"],
            "similarity": row["similarity"],
            "second_similarity": row["second_similarity"],
            "margin": row["margin"],
            "reason": row["reason"],
            "observation_count": row["observation_count"],
            "identity_source": row["identity_source"],
            "status": row["status"],
            "created_at": row["created_at"],
            "evidence": evidence,
        })
    return result


def _review_resolution_items(
    connection: sqlite3.Connection,
    row: sqlite3.Row,
    media_root: Path,
) -> list[dict]:
    try:
        evidence = json.loads(row["evidence_json"] or "[]")
    except (TypeError, ValueError) as exc:
        raise ValueError("review evidence metadata is invalid") from exc
    embeddings = {
        item["source_id"]: item["embedding"]
        for item in connection.execute(
            "SELECT source_id, embedding FROM profile_import_review_evidence "
            "WHERE review_key=?",
            (row["review_key"],),
        ).fetchall()
    }
    if not evidence or len(embeddings) != len(evidence):
        raise ValueError("review has no complete durable embedding evidence")
    items = []
    for entry in evidence:
        source_id = str(entry.get("source_id") or "")
        original = resolve_media_path(
            str(entry.get("original_image_path") or ""),
            media_root=media_root,
            require_exists=True,
            image_only=True,
        )
        crop = resolve_media_path(
            str(entry.get("face_crop_path") or ""),
            media_root=media_root,
            require_exists=True,
            image_only=True,
        )
        items.append({
            "source_id": source_id,
            "source_filename": str(entry.get("source_filename") or "phone-photo"),
            "bbox": entry.get("face_bbox") or [],
            "quality": entry.get("quality") or {},
            "content_sha256": str(entry.get("content_sha256") or ""),
            "_upload_path": original,
            "_crop_path": crop,
            "_embedding": _normalize_embedding(
                np.frombuffer(embeddings[source_id], dtype=np.float32).copy()
            ),
        })
    return items


def _materialize_review_phone_files(
    media_root: Path,
    person_id: str,
    items: list[dict],
    created: list[Path],
) -> dict[str, dict[str, str]]:
    if not _SAFE_PERSON_ID.fullmatch(person_id):
        raise ValueError("person ID is not safe for media storage")
    originals = prepare_media_directory(media_root, person_id, "phone_originals")
    crops = prepare_media_directory(media_root, person_id, "face_crops")
    result = {}
    for item in items:
        source_id = item["source_id"]
        extension = Path(item["_upload_path"]).suffix.lower()
        original_destination = originals / f"phone_{source_id}{extension}"
        crop_destination = crops / f"phone_{source_id}.jpg"
        safe_copy_into(item["_upload_path"], original_destination, created)
        safe_copy_into(item["_crop_path"], crop_destination, created)
        result[source_id] = {
            "original": normalize_media_path(original_destination, media_root=media_root),
            "crop": normalize_media_path(crop_destination, media_root=media_root),
        }
    return result


def resolve_import_review(
    review_key: str,
    body: dict,
    *,
    supervisor_id: str,
) -> dict:
    action = str(body.get("action") or "").strip()
    if action not in {"attach_existing", "create_new", "skip"}:
        raise ValueError("action must be attach_existing, create_new, or skip")
    target = str(body.get("target_person_id") or "").strip() or None
    name = str(body.get("name") or "").strip()
    notes = str(body.get("notes") or "")
    if len(name) > 100:
        raise ValueError("name must be at most 100 characters")
    if len(notes) > 4000:
        raise ValueError("notes must be at most 4000 characters")
    if action == "attach_existing" and target is None:
        raise ValueError("attach_existing requires target_person_id")
    if action == "create_new" and not name:
        raise ValueError("create_new requires an approved name")
    if action != "attach_existing":
        target = None

    memory = _profile_memory()
    connection = memory._conn
    created: list[Path] = []
    try:
        connection.execute("BEGIN IMMEDIATE")
        review = connection.execute(
            "SELECT * FROM profile_import_reviews WHERE review_key=?",
            (review_key,),
        ).fetchone()
        if review is None:
            raise KeyError("review not found")

        if review["status"] != "pending":
            stored_action = str(review["resolution_action"] or "")
            stored_target = str(review["resolution_target_person_id"] or "").strip() or None
            if stored_action != action or stored_target != target:
                raise OperationConflict(
                    "operation_conflict: review was already resolved with another action or target"
                )
            try:
                result = json.loads(review["resolution_result_json"] or "{}")
            except (TypeError, ValueError):
                result = {}
            result["idempotent_replay"] = True
            connection.execute("COMMIT")
            return result

        content_set_key = str(review["content_set_key"] or "")
        content_row = connection.execute(
            "SELECT * FROM profile_import_content_sets WHERE content_set_key=?",
            (content_set_key,),
        ).fetchone()
        if (
            content_row is None
            or content_row["semantic_action"] != "review_required"
            or str(content_row["target_person_id"] or "").strip()
            != str(review["candidate_person_id"] or "").strip()
        ):
            raise OperationConflict(
                "operation_conflict: review content intent no longer matches its pending state"
            )

        now = _now()
        if action == "skip":
            person_id = ""
            outcome = "skipped"
            result = {
                "review_key": review_key,
                "action": action,
                "person_id": person_id,
                "status": outcome,
                "content_set_key": content_set_key,
                "idempotent_replay": False,
            }
        else:
            items = _review_resolution_items(connection, review, memory.media_root)
            identity_embedding = _normalize_embedding(
                np.frombuffer(review["embedding"], dtype=np.float32).copy()
            )
            if action == "create_new":
                person_id, _default_name = memory._next_person_id()
                old_count = 0
                outcome = "created"
            else:
                person_id = str(target)
                person = connection.execute(
                    "SELECT embedding, embedding_count FROM persons "
                    "WHERE person_id=? AND is_active=1",
                    (person_id,),
                ).fetchone()
                if person is None:
                    raise ValueError("target person does not exist or is inactive")
                old_count = int(person["embedding_count"])
                outcome = "updated"

            canonical = _materialize_review_phone_files(
                memory.media_root, person_id, items, created
            )
            primary_source_id = items[0]["source_id"]
            new_primary_path = canonical[primary_source_id]["crop"]
            supervisor = connection.execute(
                "SELECT face_crop_path FROM face_photo_sources "
                "WHERE person_id=? AND is_supervisor_selected=1",
                (person_id,),
            ).fetchone()
            keep_supervisor = supervisor is not None and _media_file_exists(
                supervisor["face_crop_path"], memory.media_root
            )

            if action == "create_new":
                connection.execute(
                    """
                    INSERT INTO persons(
                        person_id, name, embedding, embedding_count, enrolled_at,
                        updated_at, cameras, profile_image, profile_image_source,
                        notes, identity_source, is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, '[]', ?, 'phone', ?, 'phone', 1)
                    """,
                    (
                        person_id, name, identity_embedding.astype(np.float32).tobytes(),
                        len(items), now, now, new_primary_path, notes,
                    ),
                )
            else:
                stored = connection.execute(
                    "SELECT embedding FROM persons WHERE person_id=?", (person_id,)
                ).fetchone()
                old_embedding = np.frombuffer(stored["embedding"], dtype=np.float32).copy()
                combined = _normalize_embedding(
                    old_embedding * old_count + identity_embedding * len(items)
                )
                connection.execute(
                    """
                    UPDATE persons
                       SET embedding=?, embedding_count=?, updated_at=?,
                           profile_image=CASE WHEN ? THEN profile_image ELSE ? END,
                           profile_image_source='phone',
                           notes=CASE WHEN ? <> '' THEN ? ELSE notes END,
                           identity_source=CASE
                               WHEN identity_source='video' THEN 'phone+video'
                               ELSE identity_source
                           END
                     WHERE person_id=?
                    """,
                    (
                        combined.astype(np.float32).tobytes(),
                        old_count + len(items), now,
                        int(keep_supervisor), new_primary_path,
                        notes, notes, person_id,
                    ),
                )
            if not keep_supervisor:
                connection.execute(
                    "UPDATE face_photo_sources SET is_primary=0 WHERE person_id=?",
                    (person_id,),
                )
            for item in items:
                paths = canonical[item["source_id"]]
                is_primary = int(
                    not keep_supervisor and item["source_id"] == primary_source_id
                )
                connection.execute(
                    """
                    INSERT INTO face_photo_sources(
                        source_id, person_id, original_image_path, face_crop_path,
                        face_bbox, quality_info, embedding, created_at,
                        source_filename, import_batch_id, is_primary,
                        is_supervisor_selected, content_sha256
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        item["source_id"], person_id, paths["original"], paths["crop"],
                        json.dumps(item["bbox"]), json.dumps(item["quality"]),
                        item["_embedding"].astype(np.float32).tobytes(), now,
                        item["source_filename"], f"review:{review_key}", is_primary,
                        item["content_sha256"],
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO identity_evidence(
                        person_id, evidence_key, crop_type, canonical_path,
                        embedding_applied, observation_weight, created_at
                    ) VALUES (?, ?, 'face', ?, 1, 1, ?)
                    """,
                    (person_id, f"phone:{item['source_id']}", paths["crop"], now),
                )
                connection.execute(
                    """
                    INSERT INTO person_gallery(
                        person_id, crop_type, path, sharpness, session_date,
                        video_source, width, height
                    ) VALUES (?, 'face', ?, ?, ?, 'phone_photo', ?, ?)
                    """,
                    (
                        person_id, paths["crop"],
                        float(item["quality"].get("sharpness", 0)),
                        date.today().isoformat(),
                        int(item["quality"].get("width", 0)),
                        int(item["quality"].get("height", 0)),
                    ),
                )
            connection.execute(
                """
                INSERT INTO recognition_log(
                    person_id, event_type, similarity, embedding_count_before,
                    embedding_count_after, video_sources, best_face_crop, ts
                ) VALUES (?, ?, NULL, ?, ?, '[]', ?, ?)
                """,
                (
                    person_id,
                    "phone_profile_created" if outcome == "created"
                    else "phone_evidence_appended",
                    None if outcome == "created" else old_count,
                    old_count + len(items), new_primary_path, now,
                ),
            )
            result = {
                "review_key": review_key,
                "action": action,
                "person_id": person_id,
                "status": outcome,
                "primary_preserved": bool(keep_supervisor),
                "content_set_key": content_set_key,
                "idempotent_replay": False,
            }

        status = "rejected" if action == "skip" else "accepted"
        connection.execute(
            """
            UPDATE profile_import_reviews
               SET status=?, reviewed_at=?, reviewed_by=?,
                   resolution_action=?, resolution_target_person_id=?,
                   resolution_result_json=?
             WHERE review_key=? AND status='pending'
            """,
            (
                status, now, supervisor_id, action, target,
                json.dumps(result), review_key,
            ),
        )
        connection.execute(
            """
            UPDATE profile_import_content_sets
               SET semantic_action=?, target_person_id=?,
                   approved_name_component=?, durable_result_json=?, updated_at=?
             WHERE content_set_key=?
            """,
            (
                action, target, name,
                json.dumps(result), now, content_set_key,
            ),
        )
        connection.execute("COMMIT")
        return result
    except BaseException:
        if connection.in_transaction:
            connection.execute("ROLLBACK")
        discard_created_media(created)
        raise
    finally:
        memory.close()


@bp.get("/api/profiles/reviews")
def profiles_reviews():
    memory = _profile_memory(read_only=True)
    try:
        return jsonify({
            "reviews": _pending_import_reviews(memory._conn, memory.media_root)
        })
    finally:
        memory.close()


@bp.post("/api/profiles/reviews/<review_key>/resolve")
def profiles_review_resolve(review_key: str):
    try:
        body = request.get_json(silent=True) or {}
        return jsonify(
            resolve_import_review(
                review_key,
                body,
                supervisor_id=_supervisor_id(),
            )
        )
    except Exception as exc:
        return _json_error(exc)


def _profile_rows(include_inactive: bool, query: str) -> list[dict]:
    memory = _profile_memory(read_only=True)
    try:
        rows = memory.list_all(include_inactive=include_inactive)
        result = []
        for row in rows:
            if query and query not in row["name"].lower() and query not in row["person_id"].lower():
                continue
            phone = memory._conn.execute(
                "SELECT COUNT(*) AS count FROM face_photo_sources WHERE person_id=?",
                (row["person_id"],),
            ).fetchone()
            video = memory._conn.execute(
                "SELECT COUNT(*) AS count FROM person_gallery "
                "WHERE person_id=? AND crop_type='face' AND video_source<>'phone_photo'",
                (row["person_id"],),
            ).fetchone()
            resolved = resolve_profile_image(
                memory._conn, row["person_id"], memory.media_root
            )
            row["phone_photo_count"] = int(phone["count"])
            row["video_evidence_count"] = int(video["count"])
            row["profile_image_url"] = _media_url(row.get("profile_image"))
            row["effective_profile_image"] = resolved["path"]
            row["effective_profile_image_url"] = _media_url(resolved["path"])
            row["profile_image_origin"] = resolved["origin"]
            result.append(row)
        return result
    finally:
        memory.close()


@bp.get("/api/profiles")
def profiles_list():
    active_filter = request.args.get("state", "active").strip().lower()
    if active_filter not in {"active", "archived", "all"}:
        return jsonify({"error": "state must be active, archived, or all"}), 400
    rows = _profile_rows(active_filter != "active", request.args.get("q", "").strip().lower())
    if active_filter == "archived":
        rows = [row for row in rows if not row["is_active"] and not row["merged_into_person_id"]]
    elif active_filter == "active":
        rows = [row for row in rows if row["is_active"]]
    return jsonify({"profiles": rows})


def _profile_detail(person_id: str) -> dict | None:
    memory = _profile_memory(read_only=True)
    try:
        person = memory.get_person(person_id)
        if person is None:
            return None
        person.pop("embedding", None)
        phone_rows = memory._conn.execute(
            """
            SELECT source_id, original_image_path, face_crop_path, face_bbox,
                   quality_info, created_at, source_filename, import_batch_id,
                   is_primary, is_supervisor_selected
              FROM face_photo_sources WHERE person_id=? ORDER BY created_at DESC, rowid DESC
            """,
            (person_id,),
        ).fetchall()
        phone_photos = []
        for row in phone_rows:
            item = dict(row)
            item["face_bbox"] = json.loads(item["face_bbox"])
            item["quality"] = json.loads(item.pop("quality_info"))
            item["is_primary"] = bool(item["is_primary"])
            item["is_supervisor_selected"] = bool(item["is_supervisor_selected"])
            item["original_image_url"] = _media_url(item["original_image_path"])
            item["face_crop_url"] = _media_url(item["face_crop_path"])
            phone_photos.append(item)
        gallery = memory.get_gallery(person_id, limit=100)
        video = [row for row in gallery if row.get("video_source") != "phone_photo"]
        suggestions = [
            row.as_dict()
            for row in memory.list_pending_identity_reviews(limit=100, offset=0)
            if row.source_person_id == person_id or row.candidate_person_id == person_id
        ]
        for row in suggestions:
            row["kind"] = "identity_suggestion"
        suggestions.extend(
            _pending_import_reviews(memory._conn, memory.media_root, person_id)
        )
        resolved = resolve_profile_image(memory._conn, person_id, memory.media_root)
        person.update({
            "phone_photos": phone_photos,
            "video_evidence": video,
            "recent_appearances": memory.get_recognition_history(person_id=person_id, limit=20),
            "pending_review_suggestions": suggestions,
            "effective_profile_image": resolved["path"],
            "effective_profile_image_url": _media_url(resolved["path"]),
            "profile_image_origin": resolved["origin"],
            "primary_face_crop_url": _media_url(resolved["path"]),
            "state": "active" if person["is_active"] else (
                "merged" if person["merged_into_person_id"] else "archived"
            ),
        })
        return person
    finally:
        memory.close()


@bp.get("/api/profiles/<person_id>")
def profiles_detail(person_id: str):
    detail = _profile_detail(person_id)
    return jsonify(detail) if detail is not None else (jsonify({"error": "profile not found"}), 404)


@bp.patch("/api/profiles/<person_id>")
def profiles_patch(person_id: str):
    body = request.get_json(silent=True) or {}
    unknown = set(body) - {"name", "notes"}
    if unknown:
        return jsonify({"error": "unsupported fields: " + ", ".join(sorted(unknown))}), 400
    name = body.get("name")
    notes = body.get("notes")
    if name is not None and (not str(name).strip() or len(str(name).strip()) > 100):
        return jsonify({"error": "name must contain 1 to 100 characters"}), 400
    if notes is not None and len(str(notes)) > 4000:
        return jsonify({"error": "notes must be at most 4000 characters"}), 400
    memory = _profile_memory()
    try:
        if memory.get_person(person_id) is None:
            return jsonify({"error": "profile not found"}), 404
        memory._conn.execute(
            "UPDATE persons SET name=COALESCE(?, name), notes=COALESCE(?, notes), updated_at=? "
            "WHERE person_id=?",
            (
                str(name).strip() if name is not None else None,
                str(notes) if notes is not None else None,
                _now(), person_id,
            ),
        )
    finally:
        memory.close()
    return jsonify(_profile_detail(person_id))


@bp.post("/api/profiles/<person_id>/photos/preview")
def profile_photo_preview(person_id: str):
    if _profile_detail(person_id) is None:
        return jsonify({"error": "profile not found"}), 404
    try:
        result = _single_preview("profile_photo")
        result["target_person_id"] = person_id
        return jsonify(result)
    except Exception as exc:
        return _json_error(exc)


@bp.post("/api/profiles/<person_id>/photos/commit")
def profile_photo_commit(person_id: str):
    body = request.get_json(silent=True) or {}
    supplied = body.get("identity") or {}
    body["identity"] = {
        **supplied,
        "action": "attach_existing",
        "existing_person_id": person_id,
        # Attaching evidence never renames the target profile.
        "update_existing_name": False,
    }
    body["identity"].pop("name", None)
    return _commit_single_preview(body)


@bp.post("/api/profiles/<person_id>/primary-photo")
def profile_primary_photo(person_id: str):
    body = request.get_json(silent=True) or {}
    source_id = str(body.get("source_id") or "")
    memory = _profile_memory()
    try:
        row = memory._conn.execute(
            "SELECT face_crop_path FROM face_photo_sources WHERE source_id=? AND person_id=?",
            (source_id, person_id),
        ).fetchone()
        if row is None:
            return jsonify({"error": "phone photo not found"}), 404
        memory._conn.execute("BEGIN IMMEDIATE")
        memory._conn.execute(
            "UPDATE face_photo_sources SET is_primary=0, is_supervisor_selected=0 "
            "WHERE person_id=?",
            (person_id,),
        )
        memory._conn.execute(
            "UPDATE face_photo_sources SET is_primary=1, is_supervisor_selected=1 "
            "WHERE source_id=?",
            (source_id,),
        )
        memory._conn.execute(
            "UPDATE persons SET profile_image=?, profile_image_source='phone', updated_at=? "
            "WHERE person_id=?",
            (row["face_crop_path"], _now(), person_id),
        )
        memory._conn.execute("COMMIT")
    except BaseException:
        if memory._conn.in_transaction:
            memory._conn.execute("ROLLBACK")
        raise
    finally:
        memory.close()
    return jsonify(_profile_detail(person_id))


def _set_archive(person_id: str, active: bool):
    memory = _profile_memory()
    try:
        row = memory._conn.execute(
            "SELECT is_active, merged_into_person_id FROM persons WHERE person_id=?",
            (person_id,),
        ).fetchone()
        if row is None:
            return jsonify({"error": "profile not found"}), 404
        if active and row["merged_into_person_id"]:
            return jsonify({"error": "a merged profile cannot be restored"}), 409
        memory._conn.execute(
            "UPDATE persons SET is_active=?, updated_at=? WHERE person_id=?",
            (int(active), _now(), person_id),
        )
    finally:
        memory.close()
    return jsonify(_profile_detail(person_id))


@bp.post("/api/profiles/<person_id>/archive")
def profile_archive(person_id: str):
    return _set_archive(person_id, False)


@bp.post("/api/profiles/<person_id>/restore")
def profile_restore(person_id: str):
    return _set_archive(person_id, True)


@bp.post("/api/profiles/merge")
def profile_merge():
    body = request.get_json(silent=True) or {}
    source = str(body.get("source_person_id") or "")
    target = str(body.get("target_person_id") or "")
    if not body.get("confirm"):
        return jsonify({"error": "explicit merge confirmation is required"}), 400
    memory = _profile_memory()
    try:
        def participant(connection, _result):
            merge_phone_evidence(connection, source, target, memory.media_root)

        result = memory.merge_persons_atomic(
            source,
            target,
            reason=str(body.get("reason") or "supervisor confirmed duplicate"),
            decision_source="profile_management",
            participant=participant,
        )
        return jsonify(asdict(result))
    except (PersonMergeError, sqlite3.Error, ValueError) as exc:
        return jsonify({"error": str(exc)}), 409
    finally:
        memory.close()
