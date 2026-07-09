from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from forensics.person_creation.state import PersonCreationState
from forensics.person_creation.nodes.prepare_runtime import prepare_runtime
from forensics.person_creation.nodes.process_video import process_video
from forensics.person_creation.nodes.filter_quality import filter_quality
<<<<<<< HEAD
from forensics.person_creation.nodes.embed_faces import embed_faces_per_person
from forensics.person_creation.nodes.auto_associate import auto_associate
from forensics.person_creation.nodes.track_persons import track_persons
from forensics.person_creation.nodes.promote_crops import promote_crops
from forensics.person_creation.nodes.select_best_per_person import select_best_per_person
from forensics.person_creation.nodes.describe_clothing_per_person import describe_clothing_per_person
from forensics.person_creation.nodes.extract_reid import extract_reid
from forensics.person_creation.nodes.extract_colors import extract_colors
from forensics.person_creation.nodes.build_multi_profile import build_multi_profile
=======
from forensics.person_creation.nodes.embed_all_faces import embed_all_faces
from forensics.person_creation.nodes.cluster_identities import cluster_identities
from forensics.person_creation.nodes.assign_bodies_to_clusters import assign_bodies_to_clusters
from forensics.person_creation.nodes.promote_crops import promote_crops
from forensics.person_creation.nodes.select_best import select_best
from forensics.person_creation.nodes.compute_reid import compute_reid
from forensics.person_creation.nodes.describe_clothing import describe_clothing
from forensics.person_creation.nodes.build_profile import build_profile
>>>>>>> Khalifa_branch
from forensics.person_creation.nodes.finalize import finalize
from forensics.person_creation.nodes.human_in_the_loop import human_in_the_loop


def build_graph():
    builder = StateGraph(PersonCreationState)

    builder.add_node("prepare_runtime",     prepare_runtime)
    builder.add_node("process_video",       process_video)
    builder.add_node("filter_quality",      filter_quality)
<<<<<<< HEAD
    # builder.add_node("human_in_the_loop",   human_in_the_loop)
    builder.add_node("auto_associate",      auto_associate)
    builder.add_node("track_persons",       track_persons)
    builder.add_node("promote_crops",       promote_crops)
    builder.add_node("embed_faces_per_person", embed_faces_per_person)
    builder.add_node("select_best_per_person", select_best_per_person)
    builder.add_node("describe_clothing_per_person", describe_clothing_per_person)
    builder.add_node("extract_reid",       extract_reid)
    builder.add_node("extract_colors",     extract_colors)
    builder.add_node("build_multi_profile", build_multi_profile)
=======
    builder.add_node("embed_all_faces",     embed_all_faces)
    builder.add_node("cluster_identities",  cluster_identities)
    builder.add_node("assign_bodies_to_clusters", assign_bodies_to_clusters)
    builder.add_node("promote_crops",       promote_crops)
    builder.add_node("select_best",         select_best)
    builder.add_node("compute_reid",        compute_reid)
    builder.add_node("describe_clothing",   describe_clothing)
    builder.add_node("build_profile",       build_profile)
>>>>>>> Khalifa_branch
    builder.add_node("finalize",            finalize)

    builder.add_edge(START,                 "prepare_runtime")
    builder.add_edge("prepare_runtime",     "process_video")
    builder.add_edge("process_video",       "filter_quality")
<<<<<<< HEAD
    # builder.add_edge("filter_quality",      "human_in_the_loop")
    builder.add_edge("filter_quality",      "auto_associate")
    builder.add_edge("auto_associate",      "track_persons")
    builder.add_edge("track_persons",       "promote_crops")
    builder.add_edge("promote_crops",       "embed_faces_per_person")
    builder.add_edge("embed_faces_per_person", "select_best_per_person")
    builder.add_edge("select_best_per_person", "describe_clothing_per_person")
    builder.add_edge("describe_clothing_per_person", "extract_reid")
    builder.add_edge("extract_reid",        "extract_colors")
    builder.add_edge("extract_colors",      "build_multi_profile")
    builder.add_edge("build_multi_profile", "finalize")
=======
    builder.add_edge("filter_quality",      "embed_all_faces")
    builder.add_edge("embed_all_faces",     "cluster_identities")
    builder.add_edge("cluster_identities",  "assign_bodies_to_clusters")
    builder.add_edge("assign_bodies_to_clusters", "promote_crops")
    builder.add_edge("promote_crops",       "select_best")
    builder.add_edge("select_best",         "compute_reid")
    builder.add_edge("compute_reid",        "describe_clothing")
    builder.add_edge("describe_clothing",   "build_profile")
    builder.add_edge("build_profile",       "finalize")
>>>>>>> Khalifa_branch
    builder.add_edge("finalize",            END)

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
