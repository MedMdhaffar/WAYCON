import cv2
import numpy as np
from pathlib import Path


def _normalize(embedding: np.ndarray) -> np.ndarray | None:
    norm = float(np.linalg.norm(embedding))
    if norm <= 0:
        return None
    return embedding / norm


def _face_paths_by_person(state: dict) -> dict[str, list[str]]:
    by_person: dict[str, list[str]] = {}

    for track in state.get("person_tracks") or []:
        person_id = track.get("person_id")
        if not person_id:
            continue
        paths = track.get("face_paths") or [
            assoc.get("face_path")
            for assoc in track.get("associations", [])
            if assoc.get("face_path")
        ]
        by_person[person_id] = list(dict.fromkeys(p for p in paths if p))

    if by_person:
        return by_person

    for assoc in state.get("associations") or []:
        person_id = assoc.get("person_id")
        face_path = assoc.get("face_path")
        if person_id and face_path:
            by_person.setdefault(person_id, []).append(face_path)

    return {
        person_id: list(dict.fromkeys(paths))
        for person_id, paths in by_person.items()
    }


def _embed_paths(face_paths: list[str], embedder, person_id: str) -> list[list[float]]:
    embeddings = []
    for path in face_paths:
        if not Path(path).exists():
            print(f"[embed_faces_per_person][warn] {person_id}: missing face crop skipped: {path}")
            continue
        try:
            img = cv2.imread(str(Path(path).resolve()))
        except Exception as exc:
            print(f"[embed_faces_per_person][warn] {person_id}: failed to read {path}: {exc}")
            continue
        if img is None:
            print(f"[embed_faces_per_person][warn] {person_id}: unreadable face crop skipped: {path}")
            continue
        try:
            emb = _normalize(np.asarray(embedder.embed(img), dtype=np.float32))
        except Exception as exc:
            print(f"[embed_faces_per_person][warn] {person_id}: embedding failed for {path}: {exc}")
            continue
        if emb is not None:
            embeddings.append(emb.astype(float).tolist())
    return embeddings


def embed_faces_per_person(state: dict) -> dict:
    """Compute one FaceNet identity embedding per tracked person."""

    from forensics.person_creation.models.face_embedder import get_face_embedder, release_face_embedder
    from forensics.person_creation.utils.memory import cleanup_memory, log_memory, clarify_oom
    from forensics.person_creation import config

    paths_by_person = _face_paths_by_person(state)
    if not paths_by_person:
        print("[embed_faces_per_person] no per-person face crops to embed")
        return {
            "face_embeddings_by_person": {},
            "face_embedding_by_person": {},
            "face_embeddings": [],
            "mean_face_embedding": [],
        }

    embedder = get_face_embedder()
    log_memory("before loading face_embedder")
    try:
        embedder.load(device=config.FACE_DEVICE)
        log_memory("after loading face_embedder")

        embeddings_by_person: dict[str, list[list[float]]] = {}
        mean_by_person: dict[str, list[float]] = {}

        for person_id, face_paths in paths_by_person.items():
            embeddings = _embed_paths(face_paths, embedder, person_id)
            embeddings_by_person[person_id] = embeddings
            if not embeddings:
                print(f"[embed_faces_per_person] {person_id}: no readable face crops")
                continue

            mean_embedding = _normalize(np.mean(np.asarray(embeddings, dtype=np.float32), axis=0))
            if mean_embedding is None:
                print(f"[embed_faces_per_person] {person_id}: mean embedding has zero norm")
                continue
            mean_by_person[person_id] = mean_embedding.astype(float).tolist()
            print(
                f"[embed_faces_per_person] {person_id} embedded {len(embeddings)} "
                f"face crops, norm={float(np.linalg.norm(mean_embedding)):.4f}"
            )
    except Exception as exc:
        raise clarify_oom(exc, "embed_faces_per_person") from exc
    finally:
        release_face_embedder()
        cleanup_memory("embed_faces_per_person")

    first_person = next(iter(mean_by_person), None)
    return {
        "face_embeddings_by_person": embeddings_by_person,
        "face_embedding_by_person": mean_by_person,
        # Legacy state fields only; build_multi_profile does not write these to profile.json.
        "face_embeddings": embeddings_by_person.get(first_person, []) if first_person else [],
        "mean_face_embedding": mean_by_person.get(first_person, []) if first_person else [],
    }


def embed_faces(state: dict) -> dict:
    return embed_faces_per_person(state)
