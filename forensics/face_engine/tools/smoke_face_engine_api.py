"""Small live API smoke test; this script never starts or loads Face Engine models."""

from __future__ import annotations

import argparse
import sys

import numpy as np

from forensics.face_engine.client import FaceEngineClient


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Face Engine base URL (defaults to FACE_ENGINE_URL)")
    args = parser.parse_args()

    client = FaceEngineClient(base_url=args.url)
    try:
        health = client.health()
        print(f"health: OK {health}")
        if not health.get("models_loaded"):
            print("FAIL: Face Engine is reachable but its models are not loaded", file=sys.stderr)
            return 1

        # A valid, low-variance image exercises encoding and the multipart contract.
        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        faces = client.detect(image)
        print(f"detect: OK ({len(faces)} face(s))")

        # Embedding requires a real face crop. Reuse a detected crop when available;
        # a no-face result is still a successful detect API smoke test.
        if not faces:
            print("embed: SKIPPED (dummy image contains no detected face)")
            return 0
        x1, y1, x2, y2 = (int(v) for v in faces[0]["bbox"])
        crop = image[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
        embedding = client.embed(crop)
        print(f"embed: OK (shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f})")
        return 0
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
