"""Identity-guard regression test for the person tracker.

Simulates the frontend bug where one "Person" card contained a woman and two
men: three different simulated identities appear sequentially near the same
image position, within the tracker's frame gap, so geometry alone would chain
them into a single track. Their (fake) face embeddings differ, so the
conservative tracker must keep them separate.

FaceNet is not loaded — a deterministic fake `embed_fn` maps each face path's
identity prefix (A_/B_/C_) to a fixed random unit vector with small per-crop
noise, mimicking same-person similarity ~0.99 and cross-person similarity
~0.0.

Run from the repository root:

    python -m forensics.person_creation.tools.test_tracking_identity_guard
"""

import sys
import zlib

import numpy as np

from forensics.person_creation.nodes.track_persons import _run_tracking


_EMBED_DIM = 512
_BBOX = [100.0, 100.0, 200.0, 400.0]  # same spot for everyone


def _make_embed_fn():
    rng = np.random.default_rng(42)
    identity_vectors = {}
    for name in ("A", "B", "C"):
        vec = rng.normal(size=_EMBED_DIM).astype(np.float32)
        identity_vectors[name] = vec / np.linalg.norm(vec)

    cache: dict[str, np.ndarray] = {}

    def embed_fn(face_path):
        if not face_path:
            return None
        if face_path in cache:
            return cache[face_path]
        identity = face_path.split("_", 1)[0]
        base = identity_vectors[identity]
        # Small per-crop noise: scale 0.005 over 512 dims keeps same-identity
        # cosine similarity ~0.99 while cross-identity stays near 0.
        noise = np.random.default_rng(zlib.crc32(face_path.encode())).normal(
            scale=0.005, size=_EMBED_DIM
        ).astype(np.float32)
        vec = base + noise
        vec = vec / np.linalg.norm(vec)
        cache[face_path] = vec
        return vec

    return embed_fn


def _assoc(identity: str, frame_idx: int) -> dict:
    return {
        "face_path": f"{identity}_f{frame_idx:06d}_face.jpg",
        "body_path": f"{identity}_f{frame_idx:06d}_body.jpg",
        "frame_idx": frame_idx,
        "video": "simulated.mp4",
        "video_name": "simulated.mp4",
        "body_bbox": list(_BBOX),
        "face_bbox": [130.0, 110.0, 170.0, 160.0],
        "body_sharpness": 100.0,
    }


def _track_identities(track: dict) -> set[str]:
    return {
        a["face_path"].split("_", 1)[0]
        for a in track["associations"]
        if a.get("face_path")
    }


def _check(condition: bool, message: str, failures: list[str]) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    if not condition:
        failures.append(message)


def main() -> int:
    failures: list[str] = []

    # --- Scenario 1: three different people take turns at the same spot ----
    associations = [
        _assoc("A", 10), _assoc("A", 15), _assoc("A", 20),
        _assoc("B", 25), _assoc("B", 30),
        _assoc("C", 35), _assoc("C", 40),
    ]
    debug: dict = {}
    tracks = _run_tracking(associations, _make_embed_fn(), debug)

    _check(
        len(tracks) == 3,
        f"three different identities produce three tracks (got {len(tracks)})",
        failures,
    )
    for track in tracks:
        identities = _track_identities(track)
        _check(
            len(identities) == 1,
            f"track_{track['track_id']} holds a single identity (got {sorted(identities)})",
            failures,
        )
    total_assocs = sum(len(t["associations"]) for t in tracks)
    _check(
        total_assocs == len(associations),
        f"no association was lost or duplicated ({total_assocs}/{len(associations)})",
        failures,
    )
    rejection_reasons = {
        rej["reason"]
        for decision in debug.get("track_decisions", [])
        for rej in decision.get("rejected_candidates", [])
    }
    _check(
        "face_similarity_low" in rejection_reasons,
        f"identity handoffs were rejected for face_similarity_low (got {sorted(rejection_reasons)})",
        failures,
    )

    # --- Scenario 2 (positive control): one person across the same frames --
    same_person = [_assoc("A", f) for f in (10, 15, 20, 25, 30, 35, 40)]
    debug2: dict = {}
    tracks2 = _run_tracking(same_person, _make_embed_fn(), debug2)
    _check(
        len(tracks2) == 1,
        f"a single consistent identity stays one track (got {len(tracks2)})",
        failures,
    )

    # --- Scenario 3: fragmented same person is merged, different is not ----
    # A appears, disappears far past the frame gap, reappears: geometry can't
    # link the fragments, the face merge must — while B stays separate.
    fragmented = [
        _assoc("A", 10), _assoc("A", 15), _assoc("A", 20),
        _assoc("B", 25), _assoc("B", 30),
        _assoc("A", 100), _assoc("A", 105), _assoc("A", 110),
    ]
    debug3: dict = {}
    tracks3 = _run_tracking(fragmented, _make_embed_fn(), debug3)
    _check(
        len(tracks3) == 2,
        f"fragmented same-person tracks merge, different person stays out (got {len(tracks3)})",
        failures,
    )
    for track in tracks3:
        identities = _track_identities(track)
        _check(
            len(identities) == 1,
            f"track_{track['track_id']} holds a single identity (got {sorted(identities)})",
            failures,
        )

    print()
    if failures:
        print(f"{len(failures)} check(s) FAILED")
        return 1
    print("All identity-guard checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
