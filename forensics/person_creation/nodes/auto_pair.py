"""Automatic face<->body association for single-subject enrollment.

Replaces the old manual `human_in_the_loop` pairing pause with a three-stage,
multi-cue pipeline:

  Stage 1 - per-frame multi-cue cost matrix (geometry + face embedding +
            body ReID + pose + detector confidence/sharpness), fused with
            renormalized weights and solved with Hungarian assignment.
  Stage 2 - track-level identity propagation: a Kalman-filtered body track
            per detected person, carrying rolling face/ReID embedding
            history, bridging short occlusions.
  Stage 3 - confidence-gated output: a pair is only kept if it clears both
            the per-frame fused-score threshold AND its owning track's
            aggregate confidence threshold.

Geometry, and detector confidence/sharpness, are always available. Face
embedding reuses the FaceNet model `load_models` already loads. Body ReID
(OSNet-x0.25 / torchreid) and pose (RTMPose / rtmlib) are optional cues:
if their dependency isn't installed the cue is disabled for the run and its
weight is redistributed across the remaining active cues — the pipeline
never fails because an optional cue is missing.

The running face-embedding reference fixes multi-video identity continuity
(the same enrolled subject appearing in more than one input video used to
have all but one video's associations discarded by the old single "longest
track wins" selection).

Output contract (unchanged, so no downstream node needs modification):
    associations, frame_groups, human_feedback_path,
    quality_body_crops, quality_face_crops
"""

from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Stage 1
_MIN_MATCH_SCORE = 0.52          # per-frame fused-score acceptance threshold
_INFEASIBLE_COST = 1e6           # geometry-impossible pairs never win Hungarian assignment
_BASE_CUE_WEIGHTS = {
    "geometry": 0.45,
    "face": 0.25,
    "reid": 0.10,
    "pose": 0.10,
    "conf_sharp": 0.10,
}

# Stage 2
_MAX_TRACK_GAP = 45               # frames a track can go unseen before it's considered finished
_MAX_TRACK_DISTANCE = 0.85        # track-continuity gate (predicted-vs-observed box distance)
_TRACK_MISMATCH_SIMILARITY = 0.15 # below this cosine sim, treat as a confident identity mismatch
_EMBED_HISTORY = 12               # rolling window size for per-track embedding history
_REFERENCE_MOMENTUM = 0.95        # EMA momentum for the running subject-identity reference

# Stage 3
_MIN_TRACK_SCORE = 0.55           # track-level confidence gate
# FaceNet vggface2 embeddings are L2-normalized; same-identity pairs typically
# cosine >= ~0.5, different-identity centers near/below 0 — 0.45 is a
# deliberately conservative merge threshold, tune against real footage.
_IDENTITY_SIMILARITY_THRESHOLD = 0.45


# ─── geometry (kept from the v1 implementation, split into hard-reject + score) ──

def _bbox_size(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1), max(0.0, y2 - y1)


def _center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, v))


def _geometry_score(face: dict, body: dict) -> tuple[bool, float, dict]:
    fx1, fy1, fx2, fy2 = face["bbox"]
    bx1, by1, bx2, by2 = body["bbox"]
    face_w, face_h = _bbox_size(face["bbox"])
    body_w, body_h = _bbox_size(body["bbox"])
    if face_w <= 0 or face_h <= 0 or body_w <= 0 or body_h <= 0:
        return False, 0.0, {"reject": "empty_bbox"}

    fcx, fcy = _center(face["bbox"])
    bcx = (bx1 + bx2) / 2.0
    upper_y = by1 + 0.20 * body_h

    inside = bx1 <= fcx <= bx2 and by1 <= fcy <= by2
    face_y_rel = (fcy - by1) / body_h
    upper_region = -0.05 <= face_y_rel <= 0.48
    ratio = face_h / body_h

    if not inside or ratio < 0.055 or ratio > 0.55:
        reason = "face_center_outside_body" if not inside else "face_body_ratio_out_of_range"
        return False, 0.0, {"reject": reason, "face_height_ratio": round(ratio, 4)}

    dx = abs(fcx - bcx) / max(body_w * 0.5, 1.0)
    dy = abs(fcy - upper_y) / max(body_h * 0.35, 1.0)
    distance_score = _clamp01(1.0 - math.sqrt(dx * dx + dy * dy) / 1.6)
    ratio_score = _clamp01(1.0 - abs(ratio - 0.24) / 0.26)
    upper_score = _clamp01(1.0 - abs(face_y_rel - 0.22) / 0.32) if upper_region else 0.0

    score = _clamp01(0.30 * float(inside) + 0.30 * upper_score + 0.20 * ratio_score + 0.20 * distance_score)
    parts = {
        "face_y_rel": round(face_y_rel, 4),
        "upper_region": upper_region,
        "face_height_ratio": round(ratio, 4),
        "distance_score": round(distance_score, 4),
        "ratio_score": round(ratio_score, 4),
        "upper_score": round(upper_score, 4),
    }
    return True, score, parts


def _conf_sharp_score(face: dict, body: dict) -> float:
    conf_score = _clamp01((float(face.get("score", 0.5)) + float(body.get("score", 0.5))) / 2.0)
    sharp_score = _clamp01(float(body.get("sharpness", 0.0)) / 1200.0)
    return _clamp01(0.65 * conf_score + 0.35 * sharp_score)


def _pose_cue_score(face_bbox: list[float], body_bbox: list[float], head_local_center) -> float | None:
    """`head_local_center` is in the body crop's own pixel space (as returned
    by PoseEstimator.head_center); translate it into frame space using the
    body bbox's top-left corner before comparing to the face bbox center.
    """
    if head_local_center is None:
        return None
    bx1, by1 = body_bbox[0], body_bbox[1]
    body_w, body_h = _bbox_size(body_bbox)
    if body_w <= 0 or body_h <= 0:
        return None
    hx, hy = head_local_center
    head_fx, head_fy = bx1 + hx, by1 + hy
    fcx, fcy = _center(face_bbox)
    dx = abs(fcx - head_fx) / max(body_w * 0.5, 1.0)
    dy = abs(fcy - head_fy) / max(body_h * 0.35, 1.0)
    return _clamp01(1.0 - math.sqrt(dx * dx + dy * dy) / 1.6)


def _cosine(a, b) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def _fuse(raw: dict[str, float | None]) -> tuple[float, dict]:
    active = {k: v for k, v in raw.items() if v is not None}
    if not active:
        return 0.0, {"weights_used": {}, "raw": {}}
    weight_total = sum(_BASE_CUE_WEIGHTS[k] for k in active)
    fused = sum(_BASE_CUE_WEIGHTS[k] * v for k, v in active.items()) / weight_total
    return _clamp01(fused), {
        "weights_used": {k: round(_BASE_CUE_WEIGHTS[k] / weight_total, 4) for k in active},
        "raw": {k: round(v, 4) for k, v in active.items()},
    }


# ─── running identity reference (progressively built subject signature) ─────

class _RunningReference:
    """EMA-smoothed mean embedding of the enrolled subject for one modality
    (face or body-ReID). Bootstraps from the first qualifying observation,
    then drifts slowly so a handful of early bystander frames can't hijack
    the identity. Persists across frame groups AND across videos within one
    auto_pair() run, which is what lets the same subject be recognized in a
    second/third input video instead of only the first.
    """

    def __init__(self, min_confidence: float, momentum: float = _REFERENCE_MOMENTUM) -> None:
        self._vec: np.ndarray | None = None
        self._min_confidence = min_confidence
        self._momentum = momentum

    @property
    def ready(self) -> bool:
        return self._vec is not None

    def similarity(self, embedding) -> float | None:
        if self._vec is None or embedding is None:
            return None
        return _cosine(self._vec, embedding)

    def update(self, embedding, confidence: float) -> None:
        if embedding is None or confidence < self._min_confidence:
            return
        vec = np.asarray(embedding, dtype=np.float64)
        self._vec = vec if self._vec is None else (self._momentum * self._vec + (1 - self._momentum) * vec)


# ─── optional-cue models + per-run caches ────────────────────────────────────

class _ScoringContext:
    def __init__(self) -> None:
        self.face_embedder = None
        self.reid_model = None
        self.pose_model = None
        self.cues_enabled = {"geometry": True, "conf_sharp": True, "face": False, "reid": False, "pose": False}

        self.image_cache: dict[str, np.ndarray | None] = {}
        self.face_embed_cache: dict[str, np.ndarray | None] = {}
        self.reid_embed_cache: dict[str, np.ndarray | None] = {}
        self.pose_cache: dict[str, tuple[float, float] | None] = {}

        self.face_ref = _RunningReference(min_confidence=_MIN_TRACK_SCORE)
        self.reid_ref = _RunningReference(min_confidence=_MIN_TRACK_SCORE)


def _attach_available_cues(ctx: _ScoringContext) -> None:
    try:
        from forensics.face_engine.local_client import LocalFaceEngine

        embedder = LocalFaceEngine()
        embedder.ensure_healthy()
        ctx.face_embedder = embedder
        ctx.cues_enabled["face"] = True
    except Exception:
        pass
    try:
        from forensics.person_creation.models.body_reid import get_body_reid

        reid = get_body_reid()
        if reid.is_available():
            ctx.reid_model = reid
            ctx.cues_enabled["reid"] = True
    except Exception:
        pass
    try:
        from forensics.person_creation.models.pose_estimator import get_pose_estimator

        pose = get_pose_estimator()
        if pose.is_available():
            ctx.pose_model = pose
            ctx.cues_enabled["pose"] = True
    except Exception:
        pass

    for name in ("face", "reid", "pose"):
        if not ctx.cues_enabled[name]:
            print(f"[auto_pair] cue disabled: {name} (model unavailable) — weight redistributed to remaining cues")


def _read_image_cached(path: str, cache: dict[str, np.ndarray | None]) -> np.ndarray | None:
    if path in cache:
        return cache[path]
    img = None
    try:
        p = Path(path)
        if p.exists():
            img = cv2.imread(str(p.resolve()))
    except Exception:
        img = None
    cache[path] = img
    return img


def _seed_face_embed_cache(ctx: _ScoringContext, all_face_embeddings: list[dict]) -> None:
    """Reuse embeddings already computed by embed_all_faces.py instead of recomputing
    them here -- same crop, same embedder, embedding_all_faces already ran first in the
    graph (embed_all_faces -> cluster_identities -> assign_bodies_to_clusters).
    """
    for record in all_face_embeddings:
        path = record.get("crop_path")
        embedding = record.get("embedding")
        if not path or embedding is None:
            continue
        ctx.face_embed_cache[path] = np.asarray(embedding, dtype=np.float64)


def _face_embedding_cached(path: str, ctx: _ScoringContext) -> np.ndarray | None:
    if path in ctx.face_embed_cache:
        return ctx.face_embed_cache[path]
    emb = None
    if ctx.face_embedder is not None:
        img = _read_image_cached(path, ctx.image_cache)
        if img is not None:
            try:
                emb = np.asarray(ctx.face_embedder.embed(img), dtype=np.float64)
            except Exception:
                emb = None
    ctx.face_embed_cache[path] = emb
    return emb


def _body_reid_embedding_cached(path: str, ctx: _ScoringContext) -> np.ndarray | None:
    if path in ctx.reid_embed_cache:
        return ctx.reid_embed_cache[path]
    emb = None
    if ctx.reid_model is not None:
        img = _read_image_cached(path, ctx.image_cache)
        if img is not None:
            vec = ctx.reid_model.embed(img)
            emb = np.asarray(vec, dtype=np.float64) if vec is not None else None
    ctx.reid_embed_cache[path] = emb
    return emb


def _pose_head_center_cached(path: str, ctx: _ScoringContext) -> tuple[float, float] | None:
    if path in ctx.pose_cache:
        return ctx.pose_cache[path]
    center = None
    if ctx.pose_model is not None:
        img = _read_image_cached(path, ctx.image_cache)
        if img is not None:
            center = ctx.pose_model.head_center(img)
    ctx.pose_cache[path] = center
    return center


# ─── Stage 1: per-frame multi-cue scoring + Hungarian assignment ────────────

def _score_pair(face: dict, body: dict, ctx: _ScoringContext) -> tuple[bool, float, dict]:
    hard_ok, geo_score, geo_parts = _geometry_score(face, body)
    if not hard_ok:
        return False, 0.0, {"reject": geo_parts.get("reject", "geometry_infeasible"), "geometry": geo_parts}

    raw: dict[str, float | None] = {"geometry": geo_score, "conf_sharp": _conf_sharp_score(face, body)}

    face_embedding = _face_embedding_cached(face["path"], ctx) if ctx.face_embedder is not None else None
    raw["face"] = ctx.face_ref.similarity(face_embedding) if face_embedding is not None else None

    body_embedding = _body_reid_embedding_cached(body["path"], ctx) if ctx.reid_model is not None else None
    raw["reid"] = ctx.reid_ref.similarity(body_embedding) if body_embedding is not None else None

    pose_center = _pose_head_center_cached(body["path"], ctx) if ctx.pose_model is not None else None
    raw["pose"] = _pose_cue_score(face["bbox"], body["bbox"], pose_center) if pose_center is not None else None

    fused, fusion_info = _fuse(raw)
    return True, fused, {
        "face_embedding": face_embedding,
        "body_embedding": body_embedding,
        "parts": {"geometry": geo_parts, **fusion_info},
    }


def _association(face: dict, body: dict, score: float, parts: dict) -> dict:
    fx1, fy1, fx2, fy2 = face["bbox"]
    bx1, by1, bx2, by2 = body["bbox"]
    face_h = fy2 - fy1
    body_h = by2 - by1
    body_area = (bx2 - bx1) * body_h
    return {
        "face_path": face["path"],
        "body_path": body["path"],
        "frame_idx": face["frame_idx"],
        "video": face["video"],
        "video_name": Path(face["video"]).name,
        "body_sharpness": body["sharpness"],
        "iou_score": None,
        "auto_score": round(float(score), 4),
        "auto_score_components": parts,
        "face_bbox": [fx1, fy1, fx2, fy2],
        "body_bbox": [bx1, by1, bx2, by2],
        "face_w": int(fx2 - fx1),
        "face_h": int(face_h),
        "body_w": int(bx2 - bx1),
        "body_h": int(body_h),
        "body_area": int(body_area),
        "face_height_ratio": round(face_h / body_h, 3) if body_h > 0 else 0,
        "confirmed_by_human": False,
        "confirmed_automatically": True,
    }


def _public_assoc(a: dict) -> dict:
    """Strip internal-only keys (raw embeddings) before an association is
    written to pairing_feedback.json or returned to the graph state.
    """
    return {k: v for k, v in a.items() if not k.startswith("_")}


def _assignment(costs: list[list[float]]) -> list[tuple[int, int]]:
    if not costs or not costs[0]:
        return []
    try:
        from scipy.optimize import linear_sum_assignment

        rows, cols = linear_sum_assignment(np.asarray(costs, dtype=np.float64))
        return list(zip((int(r) for r in rows), (int(c) for c in cols)))
    except Exception:
        pass

    row_count = len(costs)
    col_count = len(costs[0])
    transposed = False
    matrix = costs
    if row_count > col_count:
        transposed = True
        matrix = [[costs[r][c] for r in range(row_count)] for c in range(col_count)]
        row_count, col_count = col_count, row_count

    memo: dict[tuple[int, int], tuple[float, list[tuple[int, int]]]] = {}

    def solve(row: int, used_cols: int) -> tuple[float, list[tuple[int, int]]]:
        if row >= row_count:
            return 0.0, []
        key = (row, used_cols)
        if key in memo:
            return memo[key]
        best_cost = float("inf")
        best_pairs: list[tuple[int, int]] = []
        for col in range(col_count):
            if used_cols & (1 << col):
                continue
            tail_cost, tail_pairs = solve(row + 1, used_cols | (1 << col))
            total = matrix[row][col] + tail_cost
            if total < best_cost:
                best_cost = total
                best_pairs = [(row, col)] + tail_pairs
        memo[key] = (best_cost, best_pairs)
        return memo[key]

    _cost, pairs = solve(0, 0)
    if transposed:
        return [(col, row) for row, col in pairs]
    return pairs


def _build_frame_groups(quality_face: list[dict], quality_body: list[dict]) -> list[dict]:
    groups: dict[tuple, dict] = {}
    for item, key_name in ((f, "faces") for f in quality_face):
        key = (item["frame_idx"], item["video"])
        groups.setdefault(key, {
            "frame_idx": item["frame_idx"],
            "video": item["video"],
            "video_name": Path(item["video"]).name,
            "faces": [],
            "bodies": [],
        })[key_name].append(item)
    for item, key_name in ((b, "bodies") for b in quality_body):
        key = (item["frame_idx"], item["video"])
        groups.setdefault(key, {
            "frame_idx": item["frame_idx"],
            "video": item["video"],
            "video_name": Path(item["video"]).name,
            "faces": [],
            "bodies": [],
        })[key_name].append(item)

    frame_groups = [g for g in groups.values() if g["faces"] and g["bodies"]]
    frame_groups.sort(key=lambda x: (x["video"], x["frame_idx"]))
    return frame_groups


def _match_group(group: dict, ctx: _ScoringContext) -> tuple[list[dict], list[dict]]:
    faces = group["faces"]
    bodies = group["bodies"]

    pair_info: list[list[tuple]] = []
    costs: list[list[float]] = []
    for face in faces:
        info_row = []
        cost_row = []
        for body in bodies:
            hard_ok, score, extra = _score_pair(face, body, ctx)
            info_row.append((hard_ok, score, extra))
            cost_row.append(-score if hard_ok else _INFEASIBLE_COST)
        pair_info.append(info_row)
        costs.append(cost_row)

    accepted: list[dict] = []
    rejected: list[dict] = []
    for face_idx, body_idx in _assignment(costs):
        hard_ok, score, extra = pair_info[face_idx][body_idx]
        face = faces[face_idx]
        body = bodies[body_idx]
        if hard_ok and score >= _MIN_MATCH_SCORE:
            assoc = _association(face, body, score, extra["parts"])
            assoc["_face_embedding"] = extra.get("face_embedding")
            assoc["_body_embedding"] = extra.get("body_embedding")
            accepted.append(assoc)
        else:
            rejected.append({
                "face_path": face["path"],
                "body_path": body["path"],
                "frame_idx": group["frame_idx"],
                "video": group["video"],
                "auto_score": round(float(score), 4),
                "reason": "weak_match" if hard_ok else extra.get("reject", "geometry_infeasible"),
            })
    return accepted, rejected


# ─── Stage 2: Kalman-tracked, embedding-aware track-level identity ──────────

def _new_kalman(cx: float, cy: float, w: float, h: float) -> cv2.KalmanFilter:
    """Constant-velocity Kalman filter over (cx, cy, w, h). Reuses cv2's
    built-in implementation instead of adding a new dependency."""
    kf = cv2.KalmanFilter(8, 4)
    kf.transitionMatrix = np.eye(8, dtype=np.float32)
    for i in range(4):
        kf.transitionMatrix[i, i + 4] = 1.0
    kf.measurementMatrix = np.zeros((4, 8), dtype=np.float32)
    for i in range(4):
        kf.measurementMatrix[i, i] = 1.0
    kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
    kf.errorCovPost = np.eye(8, dtype=np.float32)
    kf.statePost = np.array([[cx], [cy], [w], [h], [0.0], [0.0], [0.0], [0.0]], dtype=np.float32)
    return kf


class _Track:
    def __init__(self, track_id: int, video: str, first_assoc: dict) -> None:
        self.id = track_id
        self.video = video
        self.assocs: list[dict] = []
        self.last_frame_idx = first_assoc["frame_idx"]
        cx, cy = _center(first_assoc["body_bbox"])
        w, h = _bbox_size(first_assoc["body_bbox"])
        self._kf = _new_kalman(cx, cy, w, h)
        self.face_embeds: deque = deque(maxlen=_EMBED_HISTORY)
        self.reid_embeds: deque = deque(maxlen=_EMBED_HISTORY)
        self.scores: deque = deque(maxlen=_EMBED_HISTORY)
        self._absorb(first_assoc)

    def predict(self) -> tuple[float, float, float, float]:
        state = self._kf.predict()
        flat = state.reshape(-1)
        return float(flat[0]), float(flat[1]), float(flat[2]), float(flat[3])

    def correct_and_add(self, assoc: dict) -> None:
        cx, cy = _center(assoc["body_bbox"])
        w, h = _bbox_size(assoc["body_bbox"])
        self._kf.correct(np.array([[cx], [cy], [w], [h]], dtype=np.float32))
        self.last_frame_idx = assoc["frame_idx"]
        self._absorb(assoc)

    def _absorb(self, assoc: dict) -> None:
        self.assocs.append(assoc)
        self.scores.append(float(assoc.get("auto_score", 0.0)))
        if assoc.get("_face_embedding") is not None:
            self.face_embeds.append(assoc["_face_embedding"])
        if assoc.get("_body_embedding") is not None:
            self.reid_embeds.append(assoc["_body_embedding"])

    @property
    def mean_face_embedding(self) -> np.ndarray | None:
        return np.mean(np.array(self.face_embeds), axis=0) if self.face_embeds else None

    @property
    def mean_reid_embedding(self) -> np.ndarray | None:
        return np.mean(np.array(self.reid_embeds), axis=0) if self.reid_embeds else None

    @property
    def confidence(self) -> float:
        if not self.scores:
            return 0.0
        avg_score = float(np.mean(self.scores))
        persistence_bonus = min(len(self.assocs) / 20.0, 1.0) * 0.15
        return _clamp01(avg_score * 0.85 + persistence_bonus)


class _TrackManager:
    """Maintains per-video Kalman tracks and bridges short gaps (occlusion /
    momentarily-missed detections) using the predicted box position instead
    of the last raw observation. Tracks never merge across videos here —
    cross-video identity merging happens afterwards in
    _select_dominant_identity(), using the accumulated face embeddings.
    """

    def __init__(self) -> None:
        self._active: dict[str, list[_Track]] = {}
        self._finished: list[_Track] = []
        self._next_id = 0

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def active_tracks(self, video: str) -> list[_Track]:
        return self._active.get(video, [])

    def all_tracks(self) -> list[_Track]:
        tracks = list(self._finished)
        for lst in self._active.values():
            tracks.extend(lst)
        return tracks

    def step(self, video: str, frame_idx: int, accepted: list[dict]) -> None:
        active = self._active.setdefault(video, [])
        alive = [t for t in active if frame_idx - t.last_frame_idx <= _MAX_TRACK_GAP]
        self._finished.extend(t for t in active if t not in alive)
        active[:] = alive

        if not accepted:
            return
        if not active:
            for a in accepted:
                active.append(_Track(self._new_id(), video, a))
            return

        predicted = [t.predict() for t in active]
        costs: list[list[float]] = []
        for t, (pcx, pcy, _pw, ph) in zip(active, predicted):
            row = []
            for a in accepted:
                acx, acy = _center(a["body_bbox"])
                _aw, ah = _bbox_size(a["body_bbox"])
                scale = max((ph + ah) / 2.0, 1.0)
                center_dist = math.hypot(acx - pcx, acy - pcy) / scale
                ratio_dist = abs(math.log(max(ah, 1.0) / max(ph, 1.0)))
                dist = center_dist + 0.35 * ratio_dist
                face_emb = a.get("_face_embedding")
                if t.mean_face_embedding is not None and face_emb is not None:
                    sim = _cosine(t.mean_face_embedding, face_emb)
                    if sim < _TRACK_MISMATCH_SIMILARITY:
                        # Confident identity mismatch: never continue this
                        # track onto a different person no matter how close
                        # the geometry looks (e.g. a bystander walking into
                        # the gap right after the subject moves on).
                        dist = _MAX_TRACK_DISTANCE + 1.0
                    else:
                        # Symmetric adjustment centered at sim=0.5 (neutral):
                        # rewards agreement, penalizes disagreement, instead
                        # of only ever bonusing agreement.
                        dist += (0.5 - sim) * 0.6
                row.append(dist)
            costs.append(row)

        matched_assocs: set[int] = set()
        for ti, ai in _assignment(costs):
            if costs[ti][ai] > _MAX_TRACK_DISTANCE:
                continue
            active[ti].correct_and_add(accepted[ai])
            matched_assocs.add(ai)

        for ai, a in enumerate(accepted):
            if ai not in matched_assocs:
                active.append(_Track(self._new_id(), video, a))


def _select_dominant_identity(tracks: list[_Track]) -> tuple[list[dict], list[dict], dict]:
    if not tracks:
        return [], [], {"method": "none", "reason": "no_tracks"}

    tracks_with_face = [t for t in tracks if t.mean_face_embedding is not None]
    if tracks_with_face:
        reference_track = max(tracks_with_face, key=lambda t: t.confidence)
        reference_embedding = reference_track.mean_face_embedding
        same_identity = [
            t for t in tracks_with_face
            if _cosine(t.mean_face_embedding, reference_embedding) >= _IDENTITY_SIMILARITY_THRESHOLD
        ]
        keep_ids = {t.id for t in same_identity if t.confidence >= _MIN_TRACK_SCORE}
        if keep_ids:
            same_identity_ids = {t.id for t in same_identity}
            primary: list[dict] = []
            for t in tracks:
                if t.id not in keep_ids:
                    continue
                for a in t.assocs:
                    pub = _public_assoc(a)
                    pub["track_id"] = t.id
                    pub["track_confidence"] = round(t.confidence, 4)
                    primary.append(pub)
            rejected: list[dict] = []
            for t in tracks:
                if t.id in keep_ids:
                    continue
                reason = "below_track_confidence" if t.id in same_identity_ids else "different_identity"
                rejected.extend({**_public_assoc(a), "reject_reason": reason} for a in t.assocs)
            info = {
                "method": "face_embedding_identity_merge_v1",
                "reference_track_id": reference_track.id,
                "identity_track_ids": sorted(keep_ids),
                "identity_videos": sorted({t.video for t in tracks if t.id in keep_ids}),
            }
            return primary, rejected, info

    # Fallback: legacy "single largest/highest-confidence track wins" — used
    # when face embeddings are entirely unusable (cue disabled or the
    # embedder failed on every crop) or no identity-merged group clears the
    # track-confidence bar. Matches the pre-multi-cue v1 behaviour exactly.
    def _track_score(t: _Track) -> float:
        return len(t.assocs) * 10.0 + t.confidence

    primary_track = max(tracks, key=_track_score)
    primary = []
    for a in primary_track.assocs:
        pub = _public_assoc(a)
        pub["track_id"] = primary_track.id
        pub["track_confidence"] = round(primary_track.confidence, 4)
        primary.append(pub)
    rejected = [
        {**_public_assoc(a), "reject_reason": "non_primary_track"}
        for t in tracks if t.id != primary_track.id
        for a in t.assocs
    ]
    info = {"method": "legacy_dominant_track_v1", "reference_track_id": primary_track.id}
    return primary, rejected, info


# ─── feedback-file helpers (unchanged from v1) ──────────────────────────────

def _load_reference_feedback(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _basename(path: str) -> str:
    return str(path).replace("\\", "/").rsplit("/", 1)[-1]


def _validate_against_reference(associations: list[dict], reference: dict | None) -> dict:
    if not reference:
        return {"available": False}
    ref_pairs = {
        (_basename(p.get("face_path", "")), _basename(p.get("body_path", "")))
        for p in reference.get("confirmed_pairs", []) or []
    }
    auto_pairs = {
        (_basename(p.get("face_path", "")), _basename(p.get("body_path", "")))
        for p in associations
    }
    if not ref_pairs:
        return {"available": False, "reason": "reference_has_no_pairs"}
    tp = len(auto_pairs & ref_pairs)
    fp = len(auto_pairs - ref_pairs)
    fn = len(ref_pairs - auto_pairs)
    return {
        "available": True,
        "reference_pairs": len(ref_pairs),
        "auto_pairs": len(auto_pairs),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": round(tp / max(len(auto_pairs), 1), 4),
        "recall": round(tp / max(len(ref_pairs), 1), 4),
    }


# ─── entry point ─────────────────────────────────────────────────────────────

def auto_pair(state: dict) -> dict:
    quality_face = state["quality_face_crops"]
    quality_body = state["quality_body_crops"]
    output_dir = Path(state["output_dir"])
    feedback_path = output_dir / "pairing_feedback.json"
    reference_feedback = _load_reference_feedback(feedback_path)

    ctx = _ScoringContext()
    _attach_available_cues(ctx)
    _seed_face_embed_cache(ctx, state.get("all_face_embeddings", []))

    frame_groups = _build_frame_groups(quality_face, quality_body)
    tm = _TrackManager()
    weak_rejections: list[dict] = []

    # frame_groups is already sorted by (video, frame_idx); processing in that
    # order lets the running identity reference bootstrap on the first video
    # and stay valid when a later video shows the same subject again.
    for group in frame_groups:
        accepted, rejected = _match_group(group, ctx)
        weak_rejections.extend(rejected)
        tm.step(group["video"], group["frame_idx"], accepted)
        for t in tm.active_tracks(group["video"]):
            if t.confidence >= _MIN_TRACK_SCORE:
                ctx.face_ref.update(t.mean_face_embedding, t.confidence)
                ctx.reid_ref.update(t.mean_reid_embedding, t.confidence)

    all_tracks = tm.all_tracks()
    associations, identity_rejections, identity_info = _select_dominant_identity(all_tracks)

    keep_face_paths = {a["face_path"] for a in associations}
    keep_body_paths = {a["body_path"] for a in associations}
    filtered_face = [c for c in quality_face if c["path"] in keep_face_paths]
    filtered_body = [c for c in quality_body if c["path"] in keep_body_paths]

    output_dir.mkdir(parents=True, exist_ok=True)
    validation = _validate_against_reference(associations, reference_feedback)
    feedback_data = {
        "timestamp": datetime.now().isoformat(),
        "pairing_mode": "automatic_multi_cue_v2",
        "cues_enabled": ctx.cues_enabled,
        "identity_selection": identity_info,
        "person_name": state.get("person_name", ""),
        "video_sources": state.get("video_paths", []),
        "total_frame_groups_shown": len(frame_groups),
        "candidate_pairs_count": len(associations) + len(identity_rejections),
        "confirmed_pairs_count": len(associations),
        "rejected_pairs_count": len(weak_rejections) + len(identity_rejections),
        "deleted_paths_count": 0,
        "confirmed_pairs": associations,
        "rejected_pairs": weak_rejections + identity_rejections,
        "deleted_paths": [],
        "validation_against_previous_feedback": validation,
    }
    feedback_path.write_text(json.dumps(feedback_data, indent=2), encoding="utf-8")

    videos_kept = sorted({a["video"] for a in associations})
    videos_total = len(set(state.get("video_paths", [])))
    print(
        f"[auto_pair] {len(frame_groups)} frame groups -> {len(all_tracks)} tracks "
        f"-> {len(associations)} confirmed pairs across {len(videos_kept)}/{videos_total} "
        f"video(s) (method={identity_info['method']})"
    )
    print(f"[auto_pair] cues enabled: {ctx.cues_enabled}")
    if validation.get("available"):
        print(
            f"[auto_pair] validation precision={validation['precision']} "
            f"recall={validation['recall']} vs existing feedback"
        )
    print(f"[auto_pair] feedback saved to {feedback_path}")

    return {
        "associations": associations,
        "frame_groups": frame_groups,
        "human_feedback_path": str(feedback_path.resolve()),
        "quality_body_crops": filtered_body,
        "quality_face_crops": filtered_face,
    }
