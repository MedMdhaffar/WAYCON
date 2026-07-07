"""Group face/body associations into per-person tracks.

Identity rule: face embeddings are the primary identity signal. Body-box
geometry is only a motion cue — it may propose a continuation, but a face
mismatch vetoes it. Conservative splitting is preferred over wrong merging:
one real person split across two person_ids is recoverable, two real people
merged into one person_id is not.

Every accepted/rejected continuation, merge, and split is recorded and written
to `<output_dir>/tracking_debug.json` (never into profile.json).
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from forensics.person_creation import config


_MAX_FRAME_GAP = 20
_MAX_MATCH_COST = 0.55
_MAX_CENTER_DIST = 0.45
_RELAXED_CENTER_DIST = 0.60
_MIN_RELAX_IOU = 0.15
_MAX_AREA_CHANGE = 0.85

# Legacy permissive merge threshold; only used when conservative mode is off.
_LEGACY_FACE_MERGE_SIMILARITY = 0.80

_REJECT_SENTINEL = 1_000_000.0
# Small assignment preference for the identity-consistent candidate when two
# detections both pass the geometry + face gates. Never affects acceptance.
_FACE_PREFERENCE_WEIGHT = 0.10
_MAX_LOGGED_REJECTIONS = 5


def _thresholds() -> dict:
    conservative = config.CONSERVATIVE_TRACKING
    return {
        "conservative_tracking": conservative,
        "face_continue_min_similarity": config.FACE_CONTINUE_MIN_SIMILARITY if conservative else None,
        "face_merge_min_similarity": (
            config.FACE_MERGE_MIN_SIMILARITY if conservative else _LEGACY_FACE_MERGE_SIMILARITY
        ),
        "min_face_crops_for_merge": config.MIN_FACE_CROPS_FOR_MERGE if conservative else 1,
        "min_merge_margin": config.MIN_MERGE_MARGIN if conservative else 0.0,
        "internal_face_min_similarity": config.INTERNAL_FACE_MIN_SIMILARITY,
        "internal_face_mean_similarity": config.INTERNAL_FACE_MEAN_SIMILARITY,
        "max_frame_gap": _MAX_FRAME_GAP,
        "max_match_cost": _MAX_MATCH_COST,
    }


def _bbox(a: dict) -> list[float]:
    return [float(v) for v in a["body_bbox"]]


def _bbox_dims(box: list[float]) -> tuple[float, float, float]:
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    return w, h, area


def _bbox_center(box: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _center_distance(a: list[float], b: list[float]) -> float:
    acx, acy = _bbox_center(a)
    bcx, bcy = _bbox_center(b)
    aw, ah, _area_a = _bbox_dims(a)
    bw, bh, _area_b = _bbox_dims(b)
    scale = max((aw + bw) / 2.0, (ah + bh) / 2.0, 1.0)
    return (((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5) / scale


def _area_change(a: list[float], b: list[float]) -> float:
    _aw, _ah, area_a = _bbox_dims(a)
    _bw, _bh, area_b = _bbox_dims(b)
    if area_a <= 0 or area_b <= 0:
        return 1.0
    return abs(area_a - area_b) / max(area_a, area_b)


def _track_metrics(track: dict) -> tuple[float, float]:
    moves = track.get("center_moves", [])
    ious = track.get("ious", [])
    avg_move = sum(moves) / len(moves) if moves else 0.0
    avg_iou = sum(ious) / len(ious) if ious else 0.0
    return avg_move, avg_iou


def _normalize_embedding(embedding: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        return None
    return embedding / norm


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _track_frame_keys(track: dict) -> set[tuple[str, int]]:
    return {
        (a.get("video", ""), int(a.get("frame_idx", 0)))
        for a in track.get("associations", [])
    }


def _track_face_paths(track: dict) -> list[str]:
    return list(dict.fromkeys(
        a.get("face_path")
        for a in track.get("associations", [])
        if a.get("face_path")
    ))


def _make_embed_fn(embedder):
    """Wrap the FaceNet embedder into a memoized `face_path -> unit vector`
    function so each face crop is read and embedded exactly once."""
    cache: dict[str, np.ndarray | None] = {}

    def embed_fn(face_path: str | None) -> np.ndarray | None:
        if not face_path:
            return None
        if face_path in cache:
            return cache[face_path]
        emb = None
        try:
            img = cv2.imread(str(Path(face_path).resolve()))
        except Exception:
            img = None
        if img is not None:
            try:
                emb = _normalize_embedding(np.asarray(embedder.embed(img), dtype=np.float32))
            except Exception as exc:
                print(f"[track_persons][warn] face embedding failed for {face_path}: {exc}")
        cache[face_path] = emb
        return emb

    return embed_fn


# --------------------------------------------------------------------------
# Track identity evidence
# --------------------------------------------------------------------------

def _track_add_face_embedding(track: dict, emb: np.ndarray | None) -> None:
    if emb is None:
        return
    track["face_embeddings"].append(emb)
    track["face_embedding_count"] = len(track["face_embeddings"])
    track["face_embedding_mean"] = _normalize_embedding(
        np.mean(np.asarray(track["face_embeddings"], dtype=np.float32), axis=0)
    )


def _refresh_track_face_state(track: dict, embed_fn) -> None:
    embeddings = [
        emb for emb in (embed_fn(path) for path in _track_face_paths(track))
        if emb is not None
    ]
    track["face_embeddings"] = embeddings
    track["face_embedding_count"] = len(embeddings)
    track["face_embedding_mean"] = (
        _normalize_embedding(np.mean(np.asarray(embeddings, dtype=np.float32), axis=0))
        if embeddings else None
    )


def _new_track(assoc: dict, track_id: int, face_emb: np.ndarray | None) -> dict:
    track = {
        "track_id": track_id,
        "video": assoc["video"],
        "first_frame_idx": int(assoc["frame_idx"]),
        "last_frame_idx": int(assoc["frame_idx"]),
        "first_center_x": _bbox_center(_bbox(assoc))[0],
        "last_body_bbox": _bbox(assoc),
        "associations": [dict(assoc)],
        "center_moves": [],
        "ious": [],
        "face_embeddings": [],
        "face_embedding_count": 0,
        "face_embedding_mean": None,
    }
    _track_add_face_embedding(track, face_emb)
    return track


def _append_assoc(track: dict, assoc: dict, metrics: dict, face_emb: np.ndarray | None) -> None:
    track["last_frame_idx"] = int(assoc["frame_idx"])
    track["last_body_bbox"] = _bbox(assoc)
    track["associations"].append(dict(assoc))
    track["center_moves"].append(float(metrics["center_dist"]))
    track["ious"].append(float(metrics["iou"]))
    _track_add_face_embedding(track, face_emb)


# --------------------------------------------------------------------------
# Continuation cost (geometry as motion cue, face as identity gate)
# --------------------------------------------------------------------------

def _candidate_cost(
    track: dict,
    assoc: dict,
    face_emb: np.ndarray | None,
) -> tuple[float | None, dict]:
    if track["video"] != assoc["video"]:
        return None, {"reason": "different_video"}

    frame_gap = int(assoc["frame_idx"]) - int(track["last_frame_idx"])
    if frame_gap <= 0:
        return None, {"reason": "non_forward_frame", "frame_gap": frame_gap}
    if frame_gap > _MAX_FRAME_GAP:
        return None, {"reason": "frame_gap_too_high", "frame_gap": frame_gap}

    # Identity gate first: a face mismatch vetoes the continuation no matter
    # how good the geometry looks.
    face_similarity = None
    track_mean = track.get("face_embedding_mean")
    if track_mean is not None and face_emb is not None:
        face_similarity = _cosine_similarity(track_mean, face_emb)
        if config.CONSERVATIVE_TRACKING and face_similarity < config.FACE_CONTINUE_MIN_SIMILARITY:
            return None, {
                "reason": "face_similarity_low",
                "frame_gap": frame_gap,
                "face_similarity": face_similarity,
            }

    prev_box = track["last_body_bbox"]
    box = _bbox(assoc)
    iou = _bbox_iou(prev_box, box)
    center_dist = _center_distance(prev_box, box)
    area_change = _area_change(prev_box, box)

    center_limit = _RELAXED_CENTER_DIST if iou >= _MIN_RELAX_IOU else _MAX_CENTER_DIST
    if center_dist > center_limit:
        return None, {
            "reason": "center_distance_too_high",
            "frame_gap": frame_gap,
            "center_dist": center_dist,
            "iou": iou,
            "face_similarity": face_similarity,
        }
    if area_change > _MAX_AREA_CHANGE:
        return None, {
            "reason": "area_change_too_high",
            "frame_gap": frame_gap,
            "area_change": area_change,
            "iou": iou,
            "face_similarity": face_similarity,
        }

    temporal = frame_gap / max(_MAX_FRAME_GAP, 1)
    cost = (
        (0.50 * min(1.0, center_dist))
        + (0.25 * (1.0 - iou))
        + (0.15 * min(1.0, area_change))
        + (0.10 * temporal)
    )
    if cost > _MAX_MATCH_COST:
        return None, {
            "reason": "geometry_cost_too_high",
            "cost": cost,
            "frame_gap": frame_gap,
            "center_dist": center_dist,
            "iou": iou,
            "area_change": area_change,
            "face_similarity": face_similarity,
        }

    return cost, {
        "cost": cost,
        "frame_gap": frame_gap,
        "center_dist": center_dist,
        "iou": iou,
        "area_change": area_change,
        "face_similarity": face_similarity,
    }


def _round_metrics(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        out[key] = round(value, 4) if isinstance(value, float) else value
    return out


# --------------------------------------------------------------------------
# Face-based merging of fragmented tracks (conservative)
# --------------------------------------------------------------------------

def _can_merge_tracks(a: dict, b: dict) -> bool:
    return not (_track_frame_keys(a) & _track_frame_keys(b))


def _merge_track_into(base: dict, other: dict) -> None:
    base["associations"].extend(dict(a) for a in other.get("associations", []))
    base["associations"].sort(
        key=lambda a: (a.get("video", ""), int(a.get("frame_idx", 0)), a.get("body_path", ""))
    )
    base["center_moves"].extend(other.get("center_moves", []))
    base["ious"].extend(other.get("ious", []))
    base["first_frame_idx"] = min(int(base["first_frame_idx"]), int(other["first_frame_idx"]))
    base["last_frame_idx"] = max(int(base["last_frame_idx"]), int(other["last_frame_idx"]))

    first_assoc = min(base["associations"], key=lambda a: int(a.get("frame_idx", 0)))
    last_assoc = max(base["associations"], key=lambda a: int(a.get("frame_idx", 0)))
    base["first_center_x"] = _bbox_center(_bbox(first_assoc))[0]
    base["last_body_bbox"] = _bbox(last_assoc)


def _merge_tracks_by_face_embedding(tracks: list[dict], embed_fn, debug: dict) -> list[dict]:
    thresholds = _thresholds()
    merge_threshold = thresholds["face_merge_min_similarity"]
    min_face_crops = thresholds["min_face_crops_for_merge"]
    merge_margin = thresholds["min_merge_margin"]

    print(f"[track_persons] initial_track_count={len(tracks)}")
    debug["initial_tracks_count"] = len(tracks)

    working = list(tracks)
    for track in working:
        _refresh_track_face_state(track, embed_fn)

    merge_decisions: dict[tuple[int, int], dict] = {}
    accepted_merges: list[dict] = []

    while len(working) >= 2:
        embeddings = {
            idx: track["face_embedding_mean"]
            for idx, track in enumerate(working)
            if track["face_embedding_mean"] is not None
        }
        print(f"[track_persons] tracks_with_valid_face_embeddings={len(embeddings)}")

        # All pairwise similarities among tracks with face evidence — also the
        # pool used for the ambiguity check.
        emb_items = sorted(embeddings.items())
        similarities: dict[tuple[int, int], float] = {}
        for pos, (i, emb_i) in enumerate(emb_items):
            for j, emb_j in emb_items[pos + 1:]:
                similarities[(i, j)] = _cosine_similarity(emb_i, emb_j)

        best_pair: tuple[int, int] | None = None
        best_similarity = -1.0

        for (i, j), similarity in similarities.items():
            track_i, track_j = working[i], working[j]
            pair_key = (
                min(track_i["track_id"], track_j["track_id"]),
                max(track_i["track_id"], track_j["track_id"]),
            )
            print(
                f"[track_persons] face_similarity "
                f"track_{track_i['track_id']} track_{track_j['track_id']}={similarity:.4f}"
            )

            def record(reason: str, accepted: bool = False) -> None:
                merge_decisions[pair_key] = {
                    "track_a": pair_key[0],
                    "track_b": pair_key[1],
                    "accepted": accepted,
                    "reason": reason,
                    "similarity": round(similarity, 4),
                }

            if (
                track_i["face_embedding_count"] < min_face_crops
                or track_j["face_embedding_count"] < min_face_crops
            ):
                record("insufficient_face_evidence")
                continue
            if similarity < merge_threshold:
                record("face_similarity_low")
                continue
            if not _can_merge_tracks(track_i, track_j):
                print(
                    f"[track_persons][warn] not merging track_{track_i['track_id']} "
                    f"and track_{track_j['track_id']}: duplicate frame observation"
                )
                record("duplicate_frame_observation")
                continue

            # Ambiguity: if a third track is nearly as similar to either side,
            # the identity evidence does not clearly single out this pair.
            competitor = max(
                (
                    sim
                    for (a, b), sim in similarities.items()
                    if (a, b) != (i, j) and (a in (i, j) or b in (i, j))
                ),
                default=None,
            )
            if competitor is not None and (similarity - competitor) < merge_margin:
                print(
                    f"[track_persons][warn] not merging track_{track_i['track_id']} "
                    f"and track_{track_j['track_id']}: ambiguous_face_match "
                    f"(best={similarity:.4f}, competitor={competitor:.4f})"
                )
                record("ambiguous_face_match")
                continue

            if similarity > best_similarity:
                best_pair = (i, j)
                best_similarity = similarity

        if best_pair is None:
            break

        i, j = best_pair
        base_idx, other_idx = (i, j) if working[i]["track_id"] < working[j]["track_id"] else (j, i)
        base = working[base_idx]
        other = working[other_idx]
        print(
            f"[track_persons] merged track_{base['track_id']} + "
            f"track_{other['track_id']} face_similarity={best_similarity:.4f}"
        )
        pair_key = (base["track_id"], other["track_id"])
        merge_decisions.pop(pair_key, None)
        accepted_merges.append({
            "track_a": base["track_id"],
            "track_b": other["track_id"],
            "accepted": True,
            "reason": "accepted",
            "similarity": round(best_similarity, 4),
        })
        _merge_track_into(base, other)
        _refresh_track_face_state(base, embed_fn)
        del working[other_idx]

    debug["merge_decisions"] = accepted_merges + sorted(
        merge_decisions.values(), key=lambda d: (d["track_a"], d["track_b"])
    )
    if accepted_merges:
        merged_text = ", ".join(
            f"({d['track_a']},{d['track_b']},{d['similarity']:.4f})" for d in accepted_merges
        )
        print(f"[track_persons] merged_track_pairs={merged_text}")
    else:
        print("[track_persons] merged_track_pairs=[]")
    print(f"[track_persons] track_count_after_merge={len(working)}")
    return working


# --------------------------------------------------------------------------
# Mixed-track detection and splitting
# --------------------------------------------------------------------------

def _internal_face_similarities(embeddings: list[np.ndarray]) -> list[float]:
    sims = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(_cosine_similarity(embeddings[i], embeddings[j]))
    return sims


def _split_track_by_face(track: dict, embed_fn) -> list[list[dict]]:
    """Break a mixed track into segments at points where the face identity
    changes sharply. Associations without a usable face embedding stay with
    the current segment."""
    assocs = sorted(
        track["associations"],
        key=lambda a: (a.get("video", ""), int(a.get("frame_idx", 0)), a.get("body_path", "")),
    )
    segments: list[list[dict]] = []
    current: list[dict] = []
    current_embs: list[np.ndarray] = []

    for assoc in assocs:
        emb = embed_fn(assoc.get("face_path"))
        if current and emb is not None and current_embs:
            seg_mean = _normalize_embedding(
                np.mean(np.asarray(current_embs, dtype=np.float32), axis=0)
            )
            if seg_mean is not None and _cosine_similarity(seg_mean, emb) < config.FACE_CONTINUE_MIN_SIMILARITY:
                segments.append(current)
                current = []
                current_embs = []
        current.append(assoc)
        if emb is not None:
            current_embs.append(emb)

    if current:
        segments.append(current)
    return segments


def _track_from_segment(segment: list[dict], track_id: int, embed_fn) -> dict:
    track = _new_track(segment[0], track_id, embed_fn(segment[0].get("face_path")))
    for assoc in segment[1:]:
        prev_box = track["last_body_bbox"]
        box = _bbox(assoc)
        metrics = {
            "center_dist": _center_distance(prev_box, box),
            "iou": _bbox_iou(prev_box, box),
        }
        _append_assoc(track, assoc, metrics, embed_fn(assoc.get("face_path")))
    return track


def _split_mixed_tracks(
    tracks: list[dict],
    embed_fn,
    debug: dict,
    next_track_id: int,
) -> list[dict]:
    if not config.CONSERVATIVE_TRACKING:
        return tracks

    out: list[dict] = []
    for track in tracks:
        _refresh_track_face_state(track, embed_fn)
        embeddings = track["face_embeddings"]
        if len(embeddings) < 2:
            out.append(track)
            continue

        sims = _internal_face_similarities(embeddings)
        min_sim = min(sims)
        mean_sim = sum(sims) / len(sims)
        if (
            min_sim >= config.INTERNAL_FACE_MIN_SIMILARITY
            and mean_sim >= config.INTERNAL_FACE_MEAN_SIMILARITY
        ):
            out.append(track)
            continue

        warning = (
            f"track_{track['track_id']} has inconsistent face identities "
            f"(min_sim={min_sim:.4f}, mean_sim={mean_sim:.4f}) — probably mixed"
        )
        print(f"[track_persons][warn] {warning}")
        debug["warnings"].append(warning)

        segments = _split_track_by_face(track, embed_fn)
        if len(segments) < 2:
            debug["split_decisions"].append({
                "track_id": track["track_id"],
                "split": False,
                "reason": "no_clear_split_point",
                "internal_min_similarity": round(min_sim, 4),
                "internal_mean_similarity": round(mean_sim, 4),
            })
            out.append(track)
            continue

        new_ids = []
        for segment in segments:
            new_track = _track_from_segment(segment, next_track_id, embed_fn)
            new_ids.append(next_track_id)
            next_track_id += 1
            out.append(new_track)

        debug["split_decisions"].append({
            "track_id": track["track_id"],
            "split": True,
            "reason": "internal_face_similarity_low",
            "internal_min_similarity": round(min_sim, 4),
            "internal_mean_similarity": round(mean_sim, 4),
            "segment_sizes": [len(s) for s in segments],
            "new_track_ids": new_ids,
        })
        print(
            f"[track_persons] split track_{track['track_id']} into "
            f"{len(segments)} tracks: {new_ids}"
        )

    return out


# --------------------------------------------------------------------------
# Core tracking (kept free of model loading so tests can inject an embed_fn)
# --------------------------------------------------------------------------

def _run_tracking(associations: list[dict], embed_fn, debug: dict) -> list[dict]:
    debug.setdefault("thresholds", _thresholds())
    debug.setdefault("input_associations_count", len(associations))
    debug.setdefault("track_decisions", [])
    debug.setdefault("merge_decisions", [])
    debug.setdefault("split_decisions", [])
    debug.setdefault("warnings", [])

    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for assoc in associations:
        grouped[(assoc.get("video", ""), int(assoc.get("frame_idx", 0)))].append(assoc)

    tracks: list[dict] = []
    next_track_id = 1

    for (video, frame_idx), detections in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        active = [
            track
            for track in tracks
            if track["video"] == video and 0 < frame_idx - int(track["last_frame_idx"]) <= _MAX_FRAME_GAP
        ]
        det_embs = [embed_fn(det.get("face_path")) for det in detections]

        assigned_dets: set[int] = set()
        rejections_by_det: dict[int, list[dict]] = defaultdict(list)

        if active and detections:
            candidate_costs: dict[tuple[int, int], tuple[float, dict]] = {}
            cost_matrix = []
            for track in active:
                row = []
                for det_idx, det in enumerate(detections):
                    cost, metrics = _candidate_cost(track, det, det_embs[det_idx])
                    if cost is None:
                        row.append(_REJECT_SENTINEL)
                        rejections_by_det[det_idx].append({
                            "track_id": track["track_id"],
                            **_round_metrics(metrics),
                        })
                    else:
                        # Prefer the identity-consistent candidate among gate
                        # survivors; acceptance still uses the raw geometry cost.
                        sim = metrics.get("face_similarity")
                        row.append(cost - (_FACE_PREFERENCE_WEIGHT * sim if sim is not None else 0.0))
                        candidate_costs[(track["track_id"], det_idx)] = (cost, metrics)
                cost_matrix.append(row)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for row, col in zip(row_ind, col_ind):
                if cost_matrix[row][col] >= _REJECT_SENTINEL:
                    continue
                track = active[row]
                det = detections[col]
                _accepted_cost, metrics = candidate_costs[(track["track_id"], col)]
                _append_assoc(track, det, metrics, det_embs[col])
                assigned_dets.add(col)
                debug["track_decisions"].append({
                    "frame_idx": frame_idx,
                    "video": Path(video).name if video else video,
                    "track_id": track["track_id"],
                    "candidate_body_path": det.get("body_path"),
                    "accepted": True,
                    "reason": "accepted",
                    **_round_metrics(metrics),
                })

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue
            debug["track_decisions"].append({
                "frame_idx": frame_idx,
                "video": Path(video).name if video else video,
                "track_id": next_track_id,
                "candidate_body_path": det.get("body_path"),
                "accepted": False,
                "reason": "new_track",
                "rejected_candidates": rejections_by_det.get(det_idx, [])[:_MAX_LOGGED_REJECTIONS],
            })
            tracks.append(_new_track(det, next_track_id, det_embs[det_idx]))
            next_track_id += 1

    tracks = _merge_tracks_by_face_embedding(tracks, embed_fn, debug)
    tracks = _split_mixed_tracks(tracks, embed_fn, debug, next_track_id)
    debug["final_tracks_count"] = len(tracks)
    print(f"[track_persons] final_track_count={len(tracks)}")
    return tracks


# --------------------------------------------------------------------------
# Public output + invariants
# --------------------------------------------------------------------------

def _public_track(track: dict, idx: int) -> dict:
    associations = sorted(track["associations"], key=lambda a: (a["video"], a["frame_idx"]))
    face_paths = list(dict.fromkeys(a.get("face_path") for a in associations if a.get("face_path")))
    body_paths = list(dict.fromkeys(a.get("body_path") for a in associations if a.get("body_path")))
    frames = [int(a["frame_idx"]) for a in associations]
    person_id = f"person_{idx}"

    for assoc in associations:
        assoc["person_id"] = person_id

    avg_center_movement, avg_iou = _track_metrics(track)
    return {
        "person_id": person_id,
        "associations": associations,
        "face_paths": face_paths,
        "body_paths": body_paths,
        "frame_range": [min(frames), max(frames)] if frames else [0, 0],
        "num_observations": len(associations),
        "avg_center_movement": round(avg_center_movement, 4),
        "avg_iou": round(avg_iou, 4),
        "face_embedding_count": int(track.get("face_embedding_count", 0)),
    }


def _warn_track_invariants(person_tracks: list[dict], tracks: list[dict], debug: dict) -> None:
    seen_assoc: dict[str, str] = {}
    for track in person_tracks:
        frame_counts = Counter(
            (a.get("video"), a.get("frame_idx"))
            for a in track.get("associations", [])
        )
        duplicates = [key for key, count in frame_counts.items() if count > 1]
        if duplicates:
            warning = (
                f"{track['person_id']} has multiple observations "
                f"in the same frame: {duplicates[:5]}"
            )
            print(f"[track_persons][warn] {warning}")
            debug["warnings"].append(warning)

        for assoc in track.get("associations", []):
            assoc_key = assoc.get("body_path") or f"{assoc.get('video')}:{assoc.get('frame_idx')}:{assoc.get('body_bbox')}"
            owner = seen_assoc.get(assoc_key)
            if owner and owner != track["person_id"]:
                warning = (
                    f"association appears in multiple tracks: "
                    f"{assoc_key} in {owner} and {track['person_id']}"
                )
                print(f"[track_persons][warn] {warning}")
                debug["warnings"].append(warning)
            seen_assoc[assoc_key] = track["person_id"]

    # Face identity consistency: after merging/splitting, every remaining
    # track should hold a single identity.
    for public, internal in zip(person_tracks, tracks):
        embeddings = internal.get("face_embeddings") or []
        if len(embeddings) < 2:
            continue
        sims = _internal_face_similarities(embeddings)
        min_sim = min(sims)
        mean_sim = sum(sims) / len(sims)
        if (
            min_sim < config.INTERNAL_FACE_MIN_SIMILARITY
            or mean_sim < config.INTERNAL_FACE_MEAN_SIMILARITY
        ):
            warning = (
                f"{public['person_id']} still has low internal face similarity "
                f"(min={min_sim:.4f}, mean={mean_sim:.4f}) — review its crops"
            )
            print(f"[track_persons][warn] {warning}")
            debug["warnings"].append(warning)


def _write_debug(debug: dict, output_dir: str | None) -> None:
    if not output_dir:
        return
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        debug_path = out_dir / "tracking_debug.json"
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(debug, f, indent=2)
        print(f"[track_persons] tracking debug saved -> {debug_path}")
    except OSError as exc:
        print(f"[track_persons][warn] failed to write tracking_debug.json: {exc}")


def track_persons(state: dict) -> dict:
    associations = sorted(
        list(state.get("associations") or []),
        key=lambda a: (a.get("video", ""), int(a.get("frame_idx", 0)), a.get("body_path", "")),
    )

    debug: dict = {
        "input_associations_count": len(associations),
        "thresholds": _thresholds(),
        "track_decisions": [],
        "merge_decisions": [],
        "split_decisions": [],
        "warnings": [],
    }

    from forensics.person_creation.models.face_embedder import get_face_embedder, release_face_embedder
    from forensics.person_creation.utils.memory import cleanup_memory, log_memory, clarify_oom

    embedder = get_face_embedder()
    log_memory("before loading face_embedder (track_persons)")
    try:
        embedder.load(device=config.FACE_DEVICE)
        log_memory("after loading face_embedder")
        embed_fn = _make_embed_fn(embedder)
        tracks = _run_tracking(associations, embed_fn, debug)
    except Exception as exc:
        raise clarify_oom(exc, "track_persons/face_embedder") from exc
    finally:
        release_face_embedder()
        cleanup_memory("track_persons/face_embedder")

    tracks.sort(key=lambda t: (t["first_frame_idx"], t["first_center_x"], t["track_id"]))
    person_tracks = [_public_track(track, idx + 1) for idx, track in enumerate(tracks)]
    _warn_track_invariants(person_tracks, tracks, debug)
    _write_debug(debug, state.get("output_dir"))

    tracked_associations = [
        assoc
        for track in person_tracks
        for assoc in track["associations"]
    ]

    print(
        f"[track_persons] associations={len(associations)} -> "
        f"tracks={len(person_tracks)}"
    )
    for track in person_tracks:
        print(
            f"[track_persons] {track['person_id']} observations={track['num_observations']} "
            f"frames={track['frame_range']} bodies={len(track['body_paths'])} "
            f"faces={len(track['face_paths'])} avg_center_movement={track['avg_center_movement']:.4f} "
            f"avg_iou={track['avg_iou']:.4f}"
        )

    return {
        "person_tracks": person_tracks,
        "associations": tracked_associations,
    }
