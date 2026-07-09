import operator
from typing import Annotated
from typing_extensions import TypedDict


class PersonCreationState(TypedDict):
    person_name: str
    video_paths: list[str]
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
