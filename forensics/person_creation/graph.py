from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from forensics.person_creation.state import PersonCreationState
from forensics.person_creation.nodes.load_models import load_models
from forensics.person_creation.nodes.process_video import process_video
from forensics.person_creation.nodes.filter_quality import filter_quality
from forensics.person_creation.nodes.embed_faces import embed_faces
from forensics.person_creation.nodes.auto_associate import auto_associate
from forensics.person_creation.nodes.track_persons import track_persons
from forensics.person_creation.nodes.promote_crops import promote_crops
from forensics.person_creation.nodes.select_best_per_person import select_best_per_person
from forensics.person_creation.nodes.describe_clothing_per_person import describe_clothing_per_person
from forensics.person_creation.nodes.build_multi_profile import build_multi_profile
from forensics.person_creation.nodes.finalize import finalize
from forensics.person_creation.nodes.human_in_the_loop import human_in_the_loop


def build_graph():
    builder = StateGraph(PersonCreationState)

    builder.add_node("load_models",         load_models)
    builder.add_node("process_video",       process_video)
    builder.add_node("filter_quality",      filter_quality)
    # builder.add_node("human_in_the_loop",   human_in_the_loop)
    builder.add_node("auto_associate",      auto_associate)
    builder.add_node("track_persons",       track_persons)
    builder.add_node("promote_crops",       promote_crops)
    builder.add_node("embed_faces",         embed_faces)
    builder.add_node("select_best_per_person", select_best_per_person)
    builder.add_node("describe_clothing_per_person", describe_clothing_per_person)
    builder.add_node("build_multi_profile", build_multi_profile)
    builder.add_node("finalize",            finalize)

    builder.add_edge(START,                 "load_models")
    builder.add_edge("load_models",         "process_video")
    builder.add_edge("process_video",       "filter_quality")
    # builder.add_edge("filter_quality",      "human_in_the_loop")
    builder.add_edge("filter_quality",      "auto_associate")
    builder.add_edge("auto_associate",      "track_persons")
    builder.add_edge("track_persons",       "promote_crops")
    builder.add_edge("promote_crops",       "embed_faces")
    builder.add_edge("embed_faces",         "select_best_per_person")
    builder.add_edge("select_best_per_person", "describe_clothing_per_person")
    builder.add_edge("describe_clothing_per_person", "build_multi_profile")
    builder.add_edge("build_multi_profile", "finalize")
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
