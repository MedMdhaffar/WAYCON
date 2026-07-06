import operator
from typing import Annotated
from typing_extensions import TypedDict


class PersonCreationState(TypedDict):
    person_name: str
    video_paths: list[str]
    output_dir: str
    process_every_n: int

    body_crops: Annotated[list[dict], operator.add]
    face_crops: Annotated[list[dict], operator.add]
    # each dict: {path, frame_idx, video, bbox:[x1,y1,x2,y2], sharpness}

    quality_body_crops: list[dict]
    quality_face_crops: list[dict]

    face_embeddings: list[list[float]]
    mean_face_embedding: list[float]

    frame_groups: list[dict]     # [{frame_idx, video, video_name, faces:[...], bodies:[...]}]
    associations: list[dict]     # confirmed pairs from human_in_the_loop
    person_tracks: list[dict]
    human_feedback_path: str     # absolute path to pairing_feedback.json
    best_body_crops: list[str]   # top-5 body crop paths, temporally spread
    best_body_crops_by_person: dict[str, list[str]]

    clothing_raw: str
    clothing_structured: dict  # {top, bottom, shoes, full}
    clothing_raw_by_person: dict[str, str]
    clothing_by_person: dict[str, dict]

    review_feedback: dict
    approved: bool

    profile: dict
