import argparse
import json
import uuid
from langgraph.types import Command
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
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

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

    def _stream(input_val):
        """Stream graph events and return when interrupted or done."""
        for event in graph.stream(input_val, config, stream_mode="updates"):
            for node, update in event.items():
                if node == "__interrupt__":
                    return update[0].value
                print(f"  [{node}] done")
        return None

    # --- Run until the profile-review interrupt (pairing is fully automatic) ---
    interrupt_data = _stream(initial_state)

    if interrupt_data and "profile_preview" in interrupt_data:
        print("\n" + "=" * 60)
        print("REVIEW REQUIRED")
        print(json.dumps(interrupt_data.get("profile_preview", {}), indent=2))
        print("=" * 60)
        answer = input("\nType 'approve' to save, or describe corrections: ").strip()
        resume_value = {"approved": True, "corrections": None if answer.lower() == "approve" else answer}
        _stream(Command(resume=resume_value))

    snapshot = graph.get_state(config)
    profiles = snapshot.values.get("per_cluster_profiles", {})
    profile = snapshot.values.get("profile", {})
    if profiles:
        print(f"\nProfiles saved under: {args.output}")
        for cid, item in profiles.items():
            print(f"  cluster_{cid}: {item.get('face_crop_count', 0)} face crops, appearance={item.get('appearance', {})}")
        return
    if profile:
        print(f"\nProfile saved: {args.output}/profile.json")
        print(f"  Face crops : {profile.get('face_crop_count', 0)}")
        print(f"  Appearance : {profile.get('appearance', {})}")
        feedback_path = snapshot.values.get("human_feedback_path", "")
        if feedback_path:
            print(f"  Feedback   : {feedback_path}")


if __name__ == "__main__":
    main()
