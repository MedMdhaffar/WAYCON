from functools import wraps

from langgraph.graph import StateGraph, START, END

from forensics.person_creation.state import PersonCreationState
from forensics.person_creation.nodes.load_models import load_models
from forensics.person_creation.nodes.process_video import process_video
from forensics.person_creation.nodes.filter_quality import filter_quality
from forensics.person_creation.nodes.embed_all_faces import embed_all_faces
from forensics.person_creation.nodes.cluster_identities import cluster_identities
from forensics.person_creation.nodes.assign_bodies_to_clusters import assign_bodies_to_clusters
from forensics.person_creation.nodes.promote_crops import promote_crops
from forensics.person_creation.nodes.select_best import select_best
from forensics.person_creation.nodes.compute_reid import compute_reid
from forensics.person_creation.nodes.describe_clothing import describe_clothing
from forensics.person_creation.nodes.build_profile import build_profile
from forensics.person_creation.nodes.finalize import finalize
from forensics.person_creation.utils.profiling import profile_measure


def _profiled_node(name, node):
    """Time a registered graph node without changing its public callable."""
    @wraps(node)
    def wrapped(state):
        metadata = {
            "video_count": len(state.get("video_paths", [])),
            "body_crop_count_in": len(state.get("body_crops", [])),
            "face_crop_count_in": len(state.get("face_crops", [])),
        }
        with profile_measure(f"node.{name}", metadata=metadata):
            result = node(state)
            if isinstance(result, dict):
                for key in (
                    "body_crops", "face_crops", "quality_body_crops",
                    "quality_face_crops", "all_face_embeddings",
                    "failed_face_embeddings", "identity_clusters",
                    "unresolved_faces", "associations", "tracks",
                    "best_body_crops", "per_cluster_profiles",
                ):
                    value = result.get(key)
                    if isinstance(value, (list, dict)):
                        metadata[f"{key}_count_out"] = len(value)
            return result
    return wrapped


def build_graph():
    builder = StateGraph(PersonCreationState)

    builder.add_node("load_models", _profiled_node("load_models", load_models))
    builder.add_node("process_video", _profiled_node("process_video", process_video))
    builder.add_node("filter_quality", _profiled_node("filter_quality", filter_quality))
    builder.add_node("embed_all_faces", _profiled_node("embed_all_faces", embed_all_faces))
    builder.add_node("cluster_identities", _profiled_node("cluster_identities", cluster_identities))
    builder.add_node(
        "assign_bodies_to_clusters",
        _profiled_node("assign_bodies_to_clusters", assign_bodies_to_clusters),
    )
    builder.add_node("promote_crops", _profiled_node("promote_crops", promote_crops))
    builder.add_node("select_best", _profiled_node("select_best", select_best))
    builder.add_node("compute_reid", _profiled_node("compute_reid", compute_reid))
    builder.add_node("describe_clothing", _profiled_node("describe_clothing", describe_clothing))
    builder.add_node("build_profile", _profiled_node("build_profile", build_profile))
    builder.add_node("finalize", _profiled_node("finalize", finalize))

    builder.add_edge(START,                 "load_models")
    builder.add_edge("load_models",         "process_video")
    builder.add_edge("process_video",       "filter_quality")
    builder.add_edge("filter_quality",      "embed_all_faces")
    builder.add_edge("embed_all_faces",     "cluster_identities")
    builder.add_edge("cluster_identities",  "assign_bodies_to_clusters")
    builder.add_edge("assign_bodies_to_clusters", "promote_crops")
    builder.add_edge("promote_crops",       "select_best")
    builder.add_edge("select_best",         "compute_reid")
    builder.add_edge("compute_reid",        "describe_clothing")
    builder.add_edge("describe_clothing",   "build_profile")
    builder.add_edge("build_profile",       "finalize")
    builder.add_edge("finalize",            END)

    return builder.compile()
