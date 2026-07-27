from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from forensics.person_creation.nodes.identity_config import load_identity_config


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


def _dbscan_cosine(unit_vecs: np.ndarray, eps: float, min_samples: int) -> list[int]:
    """Standard DBSCAN over a cosine-distance matrix.

    Returns a per-row cluster label; -1 marks noise. Implemented on scipy
    (already a module dependency) so identity clustering carries no extra
    third-party requirement.
    """
    n = len(unit_vecs)
    if n == 1:
        return [0] if min_samples <= 1 else [-1]

    dist = squareform(pdist(unit_vecs, metric="cosine"))
    neighbors = [set(np.nonzero(row <= eps)[0].tolist()) for row in dist]

    labels = [-1] * n
    visited = [False] * n
    cluster_id = -1

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        if len(neighbors[i]) < min_samples:
            continue  # not a core point (may still be claimed as a border point)

        cluster_id += 1
        labels[i] = cluster_id
        seeds = list(neighbors[i] - {i})
        idx = 0
        while idx < len(seeds):
            j = seeds[idx]
            idx += 1
            if not visited[j]:
                visited[j] = True
                if len(neighbors[j]) >= min_samples:  # j is core: expand through it
                    seeds.extend(k for k in neighbors[j] if k not in seeds)
            if labels[j] == -1:  # unclaimed point becomes a border of this cluster
                labels[j] = cluster_id

    return labels


def _confidence(unit_vecs: np.ndarray) -> float:
    """Mean pairwise cosine similarity within a cluster.

    Vectors are already L2-normalized, so cosine similarity is a plain dot
    product. A singleton cluster has no pairs, so it scores a perfect 1.0.
    """
    n = len(unit_vecs)
    if n < 2:
        return 1.0
    sims = unit_vecs @ unit_vecs.T
    off_diagonal = sims[np.triu_indices(n, k=1)]
    return round(float(np.mean(off_diagonal)), 4)


def _robust_representative(
    unit_vecs: np.ndarray,
    records: list[dict],
) -> tuple[np.ndarray, dict]:
    """Choose the observed medoid; one poor vector cannot replace the group."""
    if len(unit_vecs) == 1:
        index = 0
        pairwise = np.asarray([], dtype=np.float64)
    else:
        similarities = unit_vecs @ unit_vecs.T
        mean_similarity = (similarities.sum(axis=1) - 1.0) / (len(unit_vecs) - 1)
        index = max(
            range(len(records)),
            key=lambda item: (
                float(mean_similarity[item]),
                float(records[item].get("sharpness") or 0.0),
                str(records[item].get("crop_path") or ""),
            ),
        )
        pairwise = similarities[np.triu_indices(len(unit_vecs), k=1)]
    return unit_vecs[index], {
        "representative_strategy": "normalized_medoid",
        "representative_quality": (
            records[index].get("face_quality") or {}
        ).get("quality_class", "standard"),
        "intra_cluster_similarity_min": (
            round(float(pairwise.min()), 4) if pairwise.size else 1.0
        ),
        "intra_cluster_similarity_median": (
            round(float(np.median(pairwise)), 4) if pairwise.size else 1.0
        ),
        "intra_cluster_similarity_max": (
            round(float(pairwise.max()), 4) if pairwise.size else 1.0
        ),
        "outlier_count": 0,
    }


def cluster_identities(state: dict) -> dict:
    """Cluster every face embedding into identities with DBSCAN.

    Input state:  `all_face_embeddings` (one record per quality face crop,
                  each carrying its raw FaceNet `embedding`),
                  `identity_clustering_config` (eps / min_samples).
    Output state: `identity_clusters` (one entry per DBSCAN cluster with its
                  face records, mean representative embedding, face count and
                  intra-cluster confidence) and `unresolved_faces` (DBSCAN
                  noise points, label -1).
    """
    cfg = load_identity_config(state)
    records = [r for r in state.get("all_face_embeddings", []) if r.get("embedding") is not None]

    if not records:
        print("[cluster_identities] no face embeddings - no identities formed")
        return {"identity_clusters": [], "unresolved_faces": []}

    embeddings = _l2_normalize(np.asarray([r["embedding"] for r in records], dtype=np.float64))

    # Keep the configured DBSCAN density rule intact even for short live runs.
    # A singleton is noise when min_samples=3; publication policy, not a hidden
    # min_samples clamp, decides whether a strong known-person match is visible.
    min_samples = int(cfg["min_samples"])
    labels = _dbscan_cosine(embeddings, float(cfg["eps"]), min_samples)

    min_faces = int(cfg["min_cluster_face_count"])
    clusters: list[dict] = []
    unresolved: list[dict] = []

    for label in sorted(set(labels)):
        members = [i for i, lbl in enumerate(labels) if lbl == label]
        if label == -1:
            unresolved.extend(records[i] for i in members)
            continue

        member_vecs = embeddings[members]
        member_records = [records[i] for i in members]
        representative, consistency = _robust_representative(
            member_vecs,
            member_records,
        )
        distinct_frames = {
            (record.get("video"), record.get("frame_idx"))
            for record in member_records
        }
        clusters.append({
            "cluster_id": int(label),
            "face_records": [records[i] for i in members],
            "representative_embedding": representative.tolist(),
            "face_count": len(members),
            "confidence": _confidence(member_vecs),
            "low_confidence": len(members) < min_faces,
            "distinct_evidence_count": len({
                str(record.get("crop_path") or "") for record in member_records
            }),
            "distinct_frame_count": len(distinct_frames),
            "temporal_frame_min": min(
                (int(record["frame_idx"]) for record in member_records
                 if record.get("frame_idx") is not None),
                default=None,
            ),
            "temporal_frame_max": max(
                (int(record["frame_idx"]) for record in member_records
                 if record.get("frame_idx") is not None),
                default=None,
            ),
            **consistency,
        })

    for cluster in clusters:
        representative = np.asarray(
            cluster["representative_embedding"],
            dtype=np.float64,
        )
        other_similarities = [
            float(representative @ np.asarray(
                other["representative_embedding"],
                dtype=np.float64,
            ))
            for other in clusters
            if other is not cluster
        ]
        nearest = max(other_similarities) if other_similarities else None
        cluster["nearest_cluster_similarity"] = (
            round(nearest, 4) if nearest is not None else None
        )
        cluster["nearest_cluster_separation_margin"] = (
            round(
                float(cluster["intra_cluster_similarity_median"]) - nearest,
                4,
            )
            if nearest is not None else None
        )

    print(
        f"[cluster_identities] {len(clusters)} identity cluster(s) from "
        f"{len(records)} faces; {len(unresolved)} unresolved (noise)"
    )
    return {"identity_clusters": clusters, "unresolved_faces": unresolved}
