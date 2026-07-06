import operator
from typing import Annotated
from typing_extensions import TypedDict


class PersonCreationState(TypedDict):
    # Input/config
    person_name: str
    video_paths: list[str]
    output_dir: str
    process_every_n: int

    # Raw detections
    body_crops: Annotated[list[dict], operator.add]
    face_crops: Annotated[list[dict], operator.add]
    # each dict: {path, frame_idx, video, bbox:[x1,y1,x2,y2], sharpness}

    # Quality detections
    quality_body_crops: list[dict]
    quality_face_crops: list[dict]

    # Legacy single-person face fields kept only for compatibility.
    face_embeddings: list[list[float]]
    mean_face_embedding: list[float]

    # Association/tracking
    frame_groups: list[dict]     # [{frame_idx, video, video_name, faces:[...], bodies:[...]}]
    associations: list[dict]     # confirmed pairs from human_in_the_loop
    person_tracks: list[dict]
    human_feedback_path: str     # absolute path to pairing_feedback.json

    # Per-person crop selections
    best_body_crops: list[str]   # legacy first-person crop paths
    best_body_crops_by_person: dict[str, list[str]]

    # Per-person face identity
    face_embeddings_by_person: dict[str, list[list[float]]]
    face_embedding_by_person: dict[str, list[float]]

    # Per-person appearance
    clothing_raw: str  # legacy first-person clothing
    clothing_structured: dict  # legacy first-person clothing
    clothing_raw_by_person: dict[str, str]
    clothing_by_person: dict[str, dict]

    # Legacy first-person appearance fields plus per-person fields.
    reid: dict
    reid_by_person: dict[str, dict]

    color_signals: dict
    color_signals_debug: dict
    color_signals_by_person: dict[str, dict]
    color_signals_debug_by_person: dict[str, dict]

    review_feedback: dict
    approved: bool

    profile: dict
