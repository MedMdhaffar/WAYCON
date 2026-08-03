from threading import Event
from typing import Any, Callable
from typing_extensions import NotRequired, TypedDict


class PersonCreationState(TypedDict):
    person_name: str
    video_paths: list[str]
    input_type: NotRequired[str]
    source_type: NotRequired[str]
    camera_uri: NotRequired[str]
    source_uri_masked: NotRequired[str]
    camera_id: NotRequired[str]
    duration_seconds: NotRequired[int]
    live_stream_config: NotRequired[dict]
    stream_stats: NotRequired[dict]
    stream_report_path: NotRequired[str]
    _status_callback: NotRequired[Callable[..., Any]]
    _stop_event: NotRequired[Event]
    output_dir: str
    process_every_n: int
    identity_clustering_config: dict
    reid_config: dict
    reid_available: bool
    reid_unavailable_reason: str

    body_crops: list[dict]
    face_crops: list[dict]
    # each dict: {path, frame_idx, video, bbox:[x1,y1,x2,y2], sharpness}

    quality_body_crops: list[dict]
    quality_face_crops: list[dict]
    total_quality_body_crops: int
    total_quality_face_crops: int
    face_rejection_counts: NotRequired[dict[str, int]]

    face_embeddings: list[list[float]]
    mean_face_embedding: list[float]

    # DBSCAN identity clustering: raw per-face embeddings in, clusters out.
    all_face_embeddings: list[dict]
    failed_face_embeddings: list[dict]
    identity_clusters: list[dict]
    unresolved_faces: list[dict]  # DBSCAN noise faces (label -1)

    frame_groups: list[dict]     # [{frame_idx, video, video_name, faces:[...], bodies:[...]}]
    associations: list[dict]     # confirmed face/body pairs from auto_pair
    cluster_assignments: dict[int, list[dict]]
    unattached_bodies: list[dict]
    human_feedback_path: str     # absolute path to pairing_feedback.json
    best_body_crops: list[str]   # top-5 body crop paths, temporally spread
    per_cluster_best_body_crops: dict[int, list[str]]
    reid_embeddings: dict[int, list[float] | None]
    reid_crop_counts: dict[int, int]
    reid_reasons: dict[int, str]

    clothing_raw: str
    clothing_structured: dict  # {top, bottom, shoes, full}
    per_cluster_clothing: dict[int, dict]
    clothing_diagnostics: list[dict]

    profile: dict
    per_cluster_profiles: dict[int, dict]
    live_identity_decisions: NotRequired[list[dict]]
    _canonical_live_state: NotRequired[bool]
    _canonical_live_identities: NotRequired[list[dict]]
    canonical_live_report_path: NotRequired[str]
    live_finalization_timings: NotRequired[dict]
    media_lifecycle_version: NotRequired[int]
    media_cleanup_warning: NotRequired[str]
    _media_path_remap: NotRequired[dict[str, str]]
    _media_cleanup_pairs: NotRequired[list[tuple[str, str]]]
    _media_finalized_root: NotRequired[str]
