from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


_MAX_FRAME_GAP = 20
_MAX_MATCH_COST = 0.55
_MAX_CENTER_DIST = 0.45
_RELAXED_CENTER_DIST = 0.60
_MIN_RELAX_IOU = 0.15
_MAX_AREA_CHANGE = 0.85
_FACE_MERGE_SIMILARITY = 0.80


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


def _track_frame_keys(track: dict) -> set[tuple[str, int]]:
    return {
        (a.get("video", ""), int(a.get("frame_idx", 0)))
        for a in track.get("associations", [])
    }


def _can_merge_tracks(a: dict, b: dict) -> bool:
    return not (_track_frame_keys(a) & _track_frame_keys(b))


def _track_face_paths(track: dict) -> list[str]:
    return list(dict.fromkeys(
        a.get("face_path")
        for a in track.get("associations", [])
        if a.get("face_path")
    ))


def _mean_face_embedding(track: dict, embedder) -> np.ndarray | None:
    embeddings = []
    for face_path in _track_face_paths(track):
        try:
            img = cv2.imread(str(Path(face_path).resolve()))
        except Exception:
            img = None
        if img is None:
            continue

        try:
            emb = _normalize_embedding(np.asarray(embedder.embed(img), dtype=np.float32))
        except Exception as exc:
            print(f"[track_persons][warn] face embedding failed for {face_path}: {exc}")
            continue
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None

    return _normalize_embedding(np.mean(embeddings, axis=0))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


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


def _merge_tracks_by_face_embedding(tracks: list[dict], embedder) -> list[dict]:
    print(f"[track_persons] initial_track_count={len(tracks)}")
    if len(tracks) < 2:
        valid_embeddings = 0
        if tracks:
            valid_embeddings = sum(
                1 for track in tracks
                if _mean_face_embedding(track, embedder) is not None
            )
        print(f"[track_persons] tracks_with_valid_face_embeddings={valid_embeddings}")
        print("[track_persons] merged_track_pairs=[]")
        print(f"[track_persons] final_track_count={len(tracks)}")
        return tracks

    working = list(tracks)
    merged_pairs: list[tuple[int, int, float]] = []

    while True:
        embeddings: dict[int, np.ndarray] = {}
        for idx, track in enumerate(working):
            emb = _mean_face_embedding(track, embedder)
            if emb is not None:
                embeddings[idx] = emb

        print(f"[track_persons] tracks_with_valid_face_embeddings={len(embeddings)}")
        best_pair: tuple[int, int] | None = None
        best_similarity = -1.0
        emb_items = sorted(embeddings.items())

        for pos, (i, emb_i) in enumerate(emb_items):
            for j, emb_j in emb_items[pos + 1:]:
                similarity = _cosine_similarity(emb_i, emb_j)
                print(
                    f"[track_persons] face_similarity "
                    f"track_{working[i]['track_id']} track_{working[j]['track_id']}={similarity:.4f}"
                )
                if similarity < _FACE_MERGE_SIMILARITY:
                    continue
                if not _can_merge_tracks(working[i], working[j]):
                    print(
                        f"[track_persons][warn] not merging track_{working[i]['track_id']} "
                        f"and track_{working[j]['track_id']}: duplicate frame observation"
                    )
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
        merged_pairs.append((base["track_id"], other["track_id"], best_similarity))
        _merge_track_into(base, other)
        del working[other_idx]

    if merged_pairs:
        merged_text = ", ".join(f"({a},{b},{sim:.4f})" for a, b, sim in merged_pairs)
        print(f"[track_persons] merged_track_pairs={merged_text}")
    else:
        print("[track_persons] merged_track_pairs=[]")
    print(f"[track_persons] final_track_count={len(working)}")
    return working


def _candidate_cost(track: dict, assoc: dict) -> tuple[float, dict] | tuple[None, dict]:
    if track["video"] != assoc["video"]:
        return None, {"reason": "different_video"}

    frame_gap = int(assoc["frame_idx"]) - int(track["last_frame_idx"])
    if frame_gap <= 0:
        return None, {"reason": "non_forward_frame", "frame_gap": frame_gap}
    if frame_gap > _MAX_FRAME_GAP:
        return None, {"reason": "frame_gap", "frame_gap": frame_gap}

    prev_box = track["last_body_bbox"]
    box = _bbox(assoc)
    iou = _bbox_iou(prev_box, box)
    center_dist = _center_distance(prev_box, box)
    area_change = _area_change(prev_box, box)

    center_limit = _RELAXED_CENTER_DIST if iou >= _MIN_RELAX_IOU else _MAX_CENTER_DIST
    if center_dist > center_limit:
        return None, {
            "reason": "center_distance",
            "frame_gap": frame_gap,
            "center_dist": center_dist,
            "iou": iou,
        }
    if area_change > _MAX_AREA_CHANGE:
        return None, {
            "reason": "area_change",
            "frame_gap": frame_gap,
            "area_change": area_change,
            "iou": iou,
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
            "reason": "cost",
            "cost": cost,
            "frame_gap": frame_gap,
            "center_dist": center_dist,
            "iou": iou,
            "area_change": area_change,
        }

    return cost, {
        "frame_gap": frame_gap,
        "center_dist": center_dist,
        "iou": iou,
        "area_change": area_change,
    }


def _new_track(assoc: dict, track_id: int) -> dict:
    return {
        "track_id": track_id,
        "video": assoc["video"],
        "first_frame_idx": int(assoc["frame_idx"]),
        "last_frame_idx": int(assoc["frame_idx"]),
        "first_center_x": _bbox_center(_bbox(assoc))[0],
        "last_body_bbox": _bbox(assoc),
        "associations": [dict(assoc)],
        "center_moves": [],
        "ious": [],
    }


def _append_assoc(track: dict, assoc: dict, metrics: dict) -> None:
    track["last_frame_idx"] = int(assoc["frame_idx"])
    track["last_body_bbox"] = _bbox(assoc)
    track["associations"].append(dict(assoc))
    track["center_moves"].append(float(metrics["center_dist"]))
    track["ious"].append(float(metrics["iou"]))


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
    }


def _warn_track_invariants(person_tracks: list[dict]) -> None:
    seen_assoc: dict[str, str] = {}
    for track in person_tracks:
        frame_counts = Counter(
            (a.get("video"), a.get("frame_idx"))
            for a in track.get("associations", [])
        )
        duplicates = [key for key, count in frame_counts.items() if count > 1]
        if duplicates:
            print(
                f"[track_persons][warn] {track['person_id']} has multiple observations "
                f"in the same frame: {duplicates[:5]}"
            )

        for assoc in track.get("associations", []):
            assoc_key = assoc.get("body_path") or f"{assoc.get('video')}:{assoc.get('frame_idx')}:{assoc.get('body_bbox')}"
            owner = seen_assoc.get(assoc_key)
            if owner and owner != track["person_id"]:
                print(
                    f"[track_persons][warn] association appears in multiple tracks: "
                    f"{assoc_key} in {owner} and {track['person_id']}"
                )
            seen_assoc[assoc_key] = track["person_id"]


def track_persons(state: dict) -> dict:
    associations = sorted(
        list(state.get("associations") or []),
        key=lambda a: (a.get("video", ""), int(a.get("frame_idx", 0)), a.get("body_path", "")),
    )
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

        candidate_costs: dict[tuple[int, int], tuple[float, dict]] = {}
        if active and detections:
            cost_matrix = []
            for track in active:
                row = []
                for det in detections:
                    cost, metrics = _candidate_cost(track, det)
                    if cost is None:
                        row.append(1_000_000.0)
                    else:
                        row.append(cost)
                        candidate_costs[(track["track_id"], id(det))] = (cost, metrics)
                cost_matrix.append(row)

            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            assigned_tracks: set[int] = set()
            assigned_dets: set[int] = set()
            for row, col in zip(row_ind, col_ind):
                cost = cost_matrix[row][col]
                if cost > _MAX_MATCH_COST:
                    continue
                track = active[row]
                det = detections[col]
                _accepted_cost, metrics = candidate_costs[(track["track_id"], id(det))]
                _append_assoc(track, det, metrics)
                assigned_tracks.add(track["track_id"])
                assigned_dets.add(col)

            if len(assigned_tracks) != len(set(assigned_tracks)):
                print(f"[track_persons][warn] duplicate track assignment in frame {frame_idx}")
        else:
            assigned_dets = set()

        for det_idx, det in enumerate(detections):
            if det_idx in assigned_dets:
                continue
            tracks.append(_new_track(det, next_track_id))
            next_track_id += 1

    from forensics.person_creation.models.face_embedder import get_face_embedder, release_face_embedder
    from forensics.person_creation.utils.memory import cleanup_memory, log_memory, clarify_oom
    from forensics.person_creation import config

    embedder = get_face_embedder()
    log_memory("before loading face_embedder (track_persons)")
    try:
        embedder.load(device=config.FACE_DEVICE)
        log_memory("after loading face_embedder")
        tracks = _merge_tracks_by_face_embedding(tracks, embedder)
    except Exception as exc:
        raise clarify_oom(exc, "track_persons/face_embedder") from exc
    finally:
        release_face_embedder()
        cleanup_memory("track_persons/face_embedder")

    tracks.sort(key=lambda t: (t["first_frame_idx"], t["first_center_x"], t["track_id"]))
    person_tracks = [_public_track(track, idx + 1) for idx, track in enumerate(tracks)]
    _warn_track_invariants(person_tracks)

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
