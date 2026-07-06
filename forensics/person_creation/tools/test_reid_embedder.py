import argparse

import numpy as np

from forensics.person_creation.nodes.extract_reid import _extract_for_paths
from forensics.person_creation.models.reid_embedder import REID_MODEL_NAME, get_reid_embedder


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate OSNet_x1_0 ReID embeddings.")
    parser.add_argument("--crops", nargs="+", required=True, help="Body crop image paths")
    parser.add_argument("--device", default=None, help="Optional torch device, e.g. cuda or cpu")
    args = parser.parse_args()

    if args.device:
        get_reid_embedder().load(device=args.device)

    reid = _extract_for_paths(args.crops)
    embedding = reid.get("embedding")
    norm = float(np.linalg.norm(np.asarray(embedding, dtype=np.float32))) if embedding else 0.0

    print(f"model: {REID_MODEL_NAME}")
    print(f"valid_crops: {len(reid.get('source_crops') or [])}")
    print(f"embedding_dim: {reid.get('embedding_dim')}")
    print(f"l2_norm: {norm:.6f}")
    if reid.get("error"):
        print(f"error: {reid['error']}")
        return 1
    if not np.isclose(norm, 1.0, atol=1e-3):
        print("error: final embedding is not L2-normalized")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
