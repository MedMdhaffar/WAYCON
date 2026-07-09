import argparse
from forensics.person_creation.graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="Create a person profile for forensics DB")
    parser.add_argument("--name", required=True, help="Person name (e.g. Malek)")
    parser.add_argument("--videos", nargs="+", required=True, help="Paths to video clips")
    parser.add_argument("--output", required=True, help="Output directory (e.g. forensics/person_db/malek)")
    parser.add_argument("--every", type=int, default=5, help="Process every N frames (default: 5)")
    parser.add_argument("--reid-model", default="osnet_x0_25", help="Body ReID model name (default: osnet_x0_25)")
    parser.add_argument("--reid-weights", default="market1501", help="Body ReID weights label (default: market1501)")
    args = parser.parse_args()

    graph = build_graph()

    initial_state = {
        "person_name": args.name,
        "video_paths": args.videos,
        "output_dir": args.output,
        "process_every_n": args.every,
        "reid_config": {
            "model": args.reid_model,
            "weights": args.reid_weights,
            "input_size": [256, 128],
            "embedding_dim": 512,
        },
        "body_crops": [],
        "face_crops": [],
    }

    print(f"\n=== Person Creation: {args.name} ===")
    print(f"Videos : {args.videos}")
    print(f"Output : {args.output}")
    print(f"Every N: {args.every}\n")

    # Fully automatic — no human interrupts. Accumulate each node's partial
    # update the same way the Flask service does, so the final dict below
    # is the complete end-of-run state without needing a checkpointer.
    state: dict = {}
    for event in graph.stream(initial_state, stream_mode="updates"):
        for node, update in event.items():
            print(f"  [{node}] done")
            state.update(update)

    profiles = state.get("per_cluster_profiles", {})
    profile = state.get("profile", {})
    if profiles:
        print(f"\nProfiles saved under: {args.output}")
        for cid, item in profiles.items():
            print(f"  cluster_{cid}: {item.get('face_crop_count', 0)} face crops, appearance={item.get('appearance', {})}")
        return
    if profile:
        print(f"\nProfile saved: {args.output}/profile.json")
        print(f"  Face crops : {profile.get('face_crop_count', 0)}")
        print(f"  Appearance : {profile.get('appearance', {})}")
        feedback_path = state.get("human_feedback_path", "")
        if feedback_path:
            print(f"  Feedback   : {feedback_path}")


if __name__ == "__main__":
    main()
