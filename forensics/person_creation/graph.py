from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

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
from forensics.person_creation.nodes.register_global_memory import register_global_memory


def build_graph():
    builder = StateGraph(PersonCreationState)

    builder.add_node("load_models",         load_models)
    builder.add_node("process_video",       process_video)
    builder.add_node("filter_quality",      filter_quality)
    builder.add_node("embed_all_faces",     embed_all_faces)
    builder.add_node("cluster_identities",  cluster_identities)
    builder.add_node("assign_bodies_to_clusters", assign_bodies_to_clusters)
    builder.add_node("promote_crops",       promote_crops)
    builder.add_node("select_best",         select_best)
    builder.add_node("compute_reid",        compute_reid)
    builder.add_node("describe_clothing",   describe_clothing)
    builder.add_node("build_profile",       build_profile)
    builder.add_node("finalize",            finalize)
    builder.add_node("register_global_memory", register_global_memory)

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
    builder.add_edge("finalize",            "register_global_memory")
    builder.add_edge("register_global_memory", END)

    return builder.compile(checkpointer=MemorySaver())


"""
the checkpointer do the following:
preserve :
current state
previous state
execution progress
conversation history
resume point
"""
