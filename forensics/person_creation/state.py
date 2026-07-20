import operator
from typing import Annotated, Any
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

    # Presence-gated segment ingestion (see presence_segmentation.SegmentBatch /
    # single_segment_capture.py). Ingestion happens outside the graph; a run of the
    # graph consumes one already-captured segment via these fields instead of opening
    # a camera or video path itself. `segment_frames` is the one place raw in-memory
    # frame tensors live in state -- nodes/process_live_stream.py clears it once
    # consumed so the rest of the run stays references + compact metadata.
    segment_id: NotRequired[str]
    segment_seq_num: NotRequired[int]
    segment_start_ts: NotRequired[str]
    segment_end_ts: NotRequired[str]
    segment_incomplete: NotRequired[bool]
    segment_frames: NotRequired[list[Any]]
    segment_frame_timestamps: NotRequired[list[str]]

    # Tag written onto clothing_jobs rows in finalize.py; see
    # global_memory.config.CLOTHING_PIPELINE_VERSION for the default.
    pipeline_version: NotRequired[str]

    # Deliberately NOT in state: a live-progress callback. It isn't
    # JSON/pickle-serializable, which is why a LangGraph checkpointer
    # (MemorySaver) was previously dropped -- see status_reporting.py, which
    # threads it through a ContextVar instead so state stays fully
    # serializable (a prerequisite for any future segment-recovery
    # checkpointing).

    output_dir: str
    process_every_n: int
    identity_clustering_config: dict
    reid_config: dict
    reid_available: bool
    reid_unavailable_reason: str

    body_crops: Annotated[list[dict], operator.add]
    face_crops: Annotated[list[dict], operator.add]
    # each dict: {path, frame_idx, video, bbox:[x1,y1,x2,y2], sharpness}

    quality_body_crops: list[dict]
    quality_face_crops: list[dict]
    total_quality_body_crops: int
    total_quality_face_crops: int

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

    profile: dict
    per_cluster_profiles: dict[int, dict]
