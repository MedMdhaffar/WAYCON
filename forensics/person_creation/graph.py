from langgraph.graph import StateGraph, START, END

from forensics.person_creation.state import PersonCreationState
from forensics.person_creation.nodes.process_video import process_video
from forensics.person_creation.nodes.process_live_stream import process_live_stream
from forensics.person_creation.nodes.filter_quality import filter_quality
from forensics.person_creation.nodes.embed_all_faces import embed_all_faces
from forensics.person_creation.nodes.cluster_identities import cluster_identities
from forensics.person_creation.nodes.assign_bodies_to_clusters import assign_bodies_to_clusters
from forensics.person_creation.nodes.promote_crops import promote_crops
from forensics.person_creation.nodes.select_best import select_best
from forensics.person_creation.nodes.compute_reid import compute_reid
from forensics.person_creation.nodes.build_profile import build_profile
from forensics.person_creation.nodes.finalize import finalize

# load_models is intentionally NOT a node here -- see load_models.py's docstring.
# Models load once, before ingestion starts, called directly by the process
# entrypoint (service.py); this graph assumes models are already resident by the
# time it runs.


def route_ingestion(state: PersonCreationState) -> str:
    return "process_live_stream" if state.get("input_type") == "camera_uri" else "process_video"


def build_graph():
    builder = StateGraph(PersonCreationState)

    builder.add_node("process_video",       process_video)
    builder.add_node("process_live_stream", process_live_stream)
    builder.add_node("filter_quality",      filter_quality)
    builder.add_node("embed_all_faces",     embed_all_faces)
    builder.add_node("cluster_identities",  cluster_identities)
    builder.add_node("assign_bodies_to_clusters", assign_bodies_to_clusters)
    builder.add_node("promote_crops",       promote_crops)
    builder.add_node("select_best",         select_best)
    builder.add_node("compute_reid",        compute_reid)
    builder.add_node("build_profile",       build_profile)
    builder.add_node("finalize",            finalize)

    builder.add_conditional_edges(
        START,
        route_ingestion,
        {
            "process_video": "process_video",
            "process_live_stream": "process_live_stream",
        },
    )
    builder.add_edge("process_video",       "filter_quality")
    builder.add_edge("process_live_stream", "filter_quality")
    builder.add_edge("filter_quality",      "embed_all_faces")
    builder.add_edge("embed_all_faces",     "cluster_identities")
    builder.add_edge("cluster_identities",  "assign_bodies_to_clusters")
    builder.add_edge("assign_bodies_to_clusters", "promote_crops")
    builder.add_edge("promote_crops",       "select_best")
    builder.add_edge("select_best",         "compute_reid")
    # describe_clothing removed from the synchronous graph -- clothing description now
    # runs as an async job enqueued by finalize() (see clothing_jobs in
    # global_memory/schema.sql). finalize is the terminal node.
    builder.add_edge("compute_reid",        "build_profile")
    builder.add_edge("build_profile",       "finalize")
    builder.add_edge("finalize",            END)

    return builder.compile()
