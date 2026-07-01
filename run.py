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
    args = parser.parse_args()

    graph = build_graph()
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "person_name": args.name,
        "video_paths": args.videos,
        "output_dir": args.output,
        "process_every_n": args.every,
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

    # --- Run until first interrupt (human pairing) ---
    interrupt_data = _stream(initial_state)

    if interrupt_data and "frame_groups" in interrupt_data:
        print("\n" + "=" * 60)
        print("PAIRING REQUIRED")
        print(f"{interrupt_data['total_frames']} frame groups with face+body detections.")
        print("Use the UI at http://localhost:5175 to pair faces to bodies.")
        print("After confirming pairs in the UI, press Enter here to continue.")
        input("Press Enter once you have confirmed pairs in the UI...")
        # In CLI mode, pairs were submitted via UI — resume with empty payload
        # (the UI already POSTed to /api/person/confirm-pairs which set resume_value)
        # For pure CLI usage, we skip pairing and resume with empty pairs
        resume_pairing = {"human_pairs": [], "deleted_paths": []}
        interrupt_data = _stream(Command(resume=resume_pairing))

    # --- Second interrupt (profile review) ---
    if interrupt_data and "profile_preview" in interrupt_data:
        print("\n" + "=" * 60)
        print("REVIEW REQUIRED")
        print(json.dumps(interrupt_data.get("profile_preview", {}), indent=2))
        print("=" * 60)
        answer = input("\nType 'approve' to save, or describe corrections: ").strip()
        resume_value = {"approved": True, "corrections": None if answer.lower() == "approve" else answer}
        _stream(Command(resume=resume_value))

    snapshot = graph.get_state(config)
    profile = snapshot.values.get("profile", {})
    if profile:
        print(f"\nProfile saved: {args.output}/profile.json")
        print(f"  Face crops : {profile.get('face_crop_count', 0)}")
        print(f"  Appearance : {profile.get('appearance', {})}")
        feedback_path = snapshot.values.get("human_feedback_path", "")
        if feedback_path:
            print(f"  Feedback   : {feedback_path}")


if __name__ == "__main__":
    main()
