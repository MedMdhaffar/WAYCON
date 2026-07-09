"""Manual recognition test against global memory.

Usage:
    python -m forensics.global_memory.test_recognition --show-log
    python -m forensics.global_memory.test_recognition --video path/to/video.mp4
    python -m forensics.global_memory.test_recognition --video path/to/video.mp4 --register
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from forensics.global_memory import GlobalMemory
from forensics.global_memory.config import SIMILARITY_THRESHOLD


def extract_face_embeddings_from_video(
    video_path: str,
    every_n: int = 5,
    min_face_size: int = 60,
) -> list[np.ndarray]:
    import cv2
    import torch
    from facenet_pytorch import InceptionResnetV1, MTCNN
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mtcnn = MTCNN(keep_all=True, device=device, min_face_size=min_face_size)
    embedder = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    cap = cv2.VideoCapture(str(video_path))
    embeddings: list[np.ndarray] = []
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1
            if frame_idx % every_n != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                faces = mtcnn(Image.fromarray(rgb))
            except Exception:
                continue
            if faces is None:
                continue
            if faces.ndim == 3:
                faces = faces.unsqueeze(0)

            with torch.no_grad():
                batch = faces.to(device)
                embs = embedder(batch)
                embs = torch.nn.functional.normalize(embs, p=2, dim=1)
                embeddings.extend(e.cpu().numpy() for e in embs)
    finally:
        cap.release()

    print(f"Extracted {len(embeddings)} face embeddings from {video_path}")
    return embeddings


def cluster_embeddings(
    embeddings: list[np.ndarray],
    similarity_threshold: float = 0.70,
) -> list[np.ndarray]:
    if not embeddings:
        return []

    clusters: list[list[np.ndarray]] = []
    for emb in embeddings:
        matched = False
        for cluster in clusters:
            mean = np.mean(cluster, axis=0)
            mean = mean / np.linalg.norm(mean)
            if float(np.dot(mean, emb)) >= similarity_threshold:
                cluster.append(emb)
                matched = True
                break
        if not matched:
            clusters.append([emb])

    result: list[np.ndarray] = []
    for cluster in clusters:
        mean = np.mean(cluster, axis=0)
        mean = mean / np.linalg.norm(mean)
        result.append(mean)
    print(f"Clustered into {len(result)} distinct person(s)")
    return result


def show_log(person_id: str | None = None, limit: int = 50) -> None:
    gm = GlobalMemory()
    try:
        rows = gm.get_recognition_history(person_id=person_id, limit=limit)
    finally:
        gm.close()

    print("\nRECOGNITION LOG")
    print(f"Total entries: {len(rows)}")
    if not rows:
        print("(empty - no enrollments recorded yet)")
        return

    for row in rows:
        tag = "NEW" if row["event_type"] == "new_enrollment" else "MATCH"
        print(f"[{row['ts']}] {tag}")
        print(f"person_id  : {row['person_id']}")
        print(f"event      : {row['event_type']}")
        if row["similarity"] is not None:
            print(f"similarity : {row['similarity']:.4f}")
        before = row["embedding_count_before"]
        after = row["embedding_count_after"]
        if before is None:
            print(f"crop count : {after} (first enrollment)")
        else:
            print(f"crop count : {before} -> {after} (+{after - before})")
        print(f"source     : {row['video_sources']}")
        print()


def run_recognition_test(
    video_path: str,
    threshold: float | None = None,
    every_n: int = 5,
    do_register: bool = False,
) -> None:
    threshold = SIMILARITY_THRESHOLD if threshold is None else float(threshold)
    video_path = str(Path(video_path))

    print("\n" + "=" * 70)
    print("RECOGNITION TEST")
    print(f"Video    : {video_path}")
    print(f"Threshold: {threshold}")
    print(f"Register : {do_register}")
    print("=" * 70)

    embeddings = extract_face_embeddings_from_video(video_path, every_n=every_n)
    clusters = cluster_embeddings(embeddings, similarity_threshold=0.70)
    if not clusters:
        print("No face clusters found.")
        return

    gm = GlobalMemory()
    try:
        log_before = {entry["id"] for entry in gm.get_recognition_history(limit=200)}
        enrolled = gm.list_all()
        print(f"\nGlobal memory currently has {len(enrolled)} enrolled person(s):")
        for person in enrolled:
            app = person.get("latest_appearance")
            clothing = f"{app['top']}, {app['bottom']}" if app else "no appearance"
            print(f"  {person['person_id']} | {person['name']} | {clothing}")

        matched_count = 0
        new_count = 0
        for i, cluster_emb in enumerate(clusters):
            print(f"\n--- Cluster {i} ---")
            results = gm.query_by_face(cluster_emb.tolist(), top_k=3, threshold=threshold)
            if results:
                matched_count += 1
                best = results[0]
                print("  MATCH FOUND")
                print(f"  person_id  : {best['person_id']}")
                print(f"  name       : {best['name']}")
                print(f"  similarity : {best['similarity']:.4f}")
                app = best.get("appearance")
                if app:
                    print(f"  clothing   : {app.get('top')} | {app.get('bottom')}")
                    print(f"  is_stale   : {app.get('is_stale')}")

                if do_register:
                    minimal_profile = {
                        "id": best["person_id"],
                        "name": best["name"],
                        "face_embedding": cluster_emb.tolist(),
                        "face_crops": [None],
                        "video_sources": [video_path],
                        "appearance": {
                            "date": datetime.now().strftime("%Y-%m-%d"),
                            "top": None,
                            "bottom": None,
                            "shoes": None,
                            "full": None,
                        },
                        "appearance_signals": {"color": {}},
                        "best_body_crops": [],
                    }
                    assigned_id = gm.register(minimal_profile)
                    print(f"  Registered -> {assigned_id} (log updated)")
            else:
                new_count += 1
                print(f"  NO MATCH - unknown person below threshold {threshold}")
                if do_register:
                    print("  Skipping registration for unknown person.")

        if do_register:
            print("\n" + "-" * 70)
            print("RECOGNITION LOG - new entries written this run:")
            for entry in gm.get_recognition_history(limit=200):
                if entry["id"] not in log_before:
                    tag = "NEW" if entry["event_type"] == "new_enrollment" else "MATCH"
                    sim = f" sim={entry['similarity']:.4f}" if entry["similarity"] is not None else ""
                    print(f"  {tag} {entry['person_id']} [{entry['ts']}]{sim}")

        print("\n" + "=" * 70)
        print("SUMMARY")
        print(f"  Clusters detected : {len(clusters)}")
        print(f"  Matched existing  : {matched_count}")
        print(f"  New unknown       : {new_count}")
        print("=" * 70)
    finally:
        gm.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=None, help="Path to test video")
    parser.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    parser.add_argument("--every_n", type=int, default=5)
    parser.add_argument("--show-log", action="store_true")
    parser.add_argument("--person", default=None, help="Filter --show-log by person_id")
    parser.add_argument("--register", action="store_true", help="Write recognized matches to the log")
    args = parser.parse_args()

    if args.show_log:
        show_log(person_id=args.person)
    elif args.video:
        run_recognition_test(
            video_path=args.video,
            threshold=args.threshold,
            every_n=args.every_n,
            do_register=args.register,
        )
    else:
        parser.print_help()
