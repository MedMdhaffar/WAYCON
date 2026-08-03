"""Copy retained crops from staging to cluster storage and publish one remap."""

from __future__ import annotations

from pathlib import Path

from forensics.media_paths import get_media_root, resolve_media_path
from forensics.person_creation.media_lifecycle import (
    cleanup_relocated_sources,
    copy_media_for_handoff,
    rewrite_media_references,
)


_PROMOTION_FIELDS = (
    "body_crops",
    "face_crops",
    "associations",
    "identity_clusters",
    "cluster_assignments",
    "all_face_embeddings",
    "failed_face_embeddings",
    "quality_face_crops",
    "quality_body_crops",
    "frame_groups",
    "unresolved_faces",
    "unattached_bodies",
    "rolling_analysis",
    "stream_stats",
)


def promote_crops(state: dict) -> dict:
    """Copy confirmed evidence, verify destinations, then publish rewritten state."""
    output_dir = Path(state["output_dir"])
    media_root = get_media_root()
    associations = list(state.get("associations") or [])
    clusters = list(state.get("identity_clusters") or [])
    remap: dict[str, str] = {}
    cleanup_pairs: list[tuple[str, str]] = []

    def promote(raw_path: str, destination_dir: Path) -> str:
        if raw_path in remap:
            return remap[raw_path]
        destination = destination_dir / Path(raw_path).name
        canonical, cleanup = copy_media_for_handoff(
            raw_path,
            destination,
            media_root=media_root,
        )
        remap[raw_path] = str(resolve_media_path(
            canonical,
            media_root=media_root,
            allow_legacy_absolute=False,
            require_exists=True,
            image_only=True,
        ))
        if cleanup is not None and cleanup not in cleanup_pairs:
            cleanup_pairs.append(cleanup)
        return remap[raw_path]

    for cluster in clusters:
        cid = int(cluster["cluster_id"])
        face_dir = output_dir / f"cluster_{cid}" / "face_crops"
        for record in cluster.get("face_records", []):
            if record.get("crop_path"):
                promote(str(record["crop_path"]), face_dir)

    for association in associations:
        cid = association.get("cluster_id")
        cluster_dir = output_dir if cid is None else output_dir / f"cluster_{int(cid)}"
        if association.get("face_path"):
            promote(str(association["face_path"]), cluster_dir / "face_crops")
        if association.get("body_path"):
            promote(str(association["body_path"]), cluster_dir / "body_crops")

    retained_old = set(remap)

    def retained_crops(field: str) -> list[dict]:
        return [
            rewrite_media_references(item, remap)
            for item in (state.get(field) or [])
            if item.get("path") in retained_old
        ]

    update = {
        field: rewrite_media_references(state.get(field), remap)
        for field in _PROMOTION_FIELDS
        if field in state
    }
    update["quality_face_crops"] = retained_crops("quality_face_crops")
    update["quality_body_crops"] = retained_crops("quality_body_crops")
    update["_media_path_remap"] = remap
    update["_media_cleanup_pairs"] = cleanup_pairs
    update["media_lifecycle_version"] = int(state.get("media_lifecycle_version", 0)) + 1
    print(
        f"[promote_crops] verified {len(cleanup_pairs)} retained crop handoffs; "
        "staging sources remain until the rewritten state is published"
    )
    return update


def cleanup_promoted_crops(state: dict) -> dict:
    """Remove promoted staging copies after the promotion update was published."""
    pairs = list(state.get("_media_cleanup_pairs") or [])
    removed = cleanup_relocated_sources(pairs, media_root=get_media_root())
    print(f"[cleanup_promoted_crops] removed {len(removed)} verified staging copies")
    return {
        "_media_cleanup_pairs": [],
        "media_cleanup_warning": "",
    }
