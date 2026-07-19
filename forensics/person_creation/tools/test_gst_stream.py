"""Manual/interactive test for GstFrameBuffer (forensics/person_creation/gst_stream.py).

Exercises against a real RTSP source (a camera, or a local test server -- see the
runbook printed by --help / the accompanying instructions) without touching the
rest of the person_creation pipeline. Prints frame throughput and connection-state
transitions (CONNECTED / RECONNECTING) live, so a network interruption or a stopped
publisher can be observed causing a reconnect with exponential backoff.

Usage:
    python -m forensics.person_creation.tools.test_gst_stream rtsp://user:pass@host:554/path
    python -m forensics.person_creation.tools.test_gst_stream rtsp://localhost:8554/test \
        --duration 120 --codec h264

Exit code is 0 if at least one frame was received, 1 otherwise.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time

from forensics.person_creation.gst_stream import ConnectionState, GstFrameBuffer
from forensics.person_creation.live_stream import mask_camera_uri


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("uri", help="RTSP URI, e.g. rtsp://user:pass@10.0.0.104:554/Streaming/Channels/101")
    parser.add_argument("--duration", type=float, default=60.0, help="seconds to run (default 60, 0 = until Ctrl+C)")
    parser.add_argument("--codec", choices=["h264", "h265"], default="h264")
    parser.add_argument("--decoder", default=None, help="override decoder element(s), e.g. 'nvv4l2decoder ! nvvideoconvert'")
    parser.add_argument("--stall-timeout", type=float, default=5.0, help="seconds with no frame before declaring a stall")
    parser.add_argument("--buffer-size", type=int, default=10)
    parser.add_argument("--save-frame", metavar="PATH", default=None, help="write the first received frame as a JPEG to PATH")
    args = parser.parse_args()

    events: list[str] = []

    def on_disconnect(reason: str) -> None:
        msg = f"[{_now()}] DISCONNECTED reason={reason!r}"
        print(msg, flush=True)
        events.append(msg)

    def on_reconnected() -> None:
        msg = f"[{_now()}] RECONNECTED"
        print(msg, flush=True)
        events.append(msg)

    def on_state_change(state: ConnectionState) -> None:
        print(f"[{_now()}] state -> {state.value}", flush=True)

    print(f"Connecting to {mask_camera_uri(args.uri)} (codec={args.codec}, decoder={args.decoder or 'default'})")
    buffer = GstFrameBuffer(
        args.uri,
        max_size=args.buffer_size,
        codec=args.codec,
        decoder=args.decoder,
        stall_timeout_seconds=args.stall_timeout,
        on_disconnect=on_disconnect,
        on_reconnected=on_reconnected,
        on_state_change=on_state_change,
    )

    stop = {"flag": False}

    def _sigint(_sig, _frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _sigint)

    buffer.start()
    print("Pipeline started. Press Ctrl+C to stop early.\n")

    started = time.monotonic()
    saved_first_frame = False
    last_report = 0.0
    frames_seen = 0

    try:
        while not stop["flag"]:
            elapsed = time.monotonic() - started
            if args.duration > 0 and elapsed >= args.duration:
                break

            item = buffer.get(timeout=1.0)
            if item is not None:
                frames_seen += 1
                if args.save_frame and not saved_first_frame:
                    _save_jpeg(item.frame, args.save_frame)
                    print(f"[{_now()}] saved first frame -> {args.save_frame}")
                    saved_first_frame = True

            if elapsed - last_report >= 2.0:
                stats = buffer.stats(frames_seen)
                print(
                    f"[{_now()}] elapsed={elapsed:5.1f}s state={stats['connection_state']:12s} "
                    f"read={stats['frames_read']:5d} dropped={stats['frames_dropped']:4d} "
                    f"reconnects={stats['reconnect_count']}",
                    flush=True,
                )
                last_report = elapsed
    finally:
        buffer.stop()

    final_stats = buffer.stats(frames_seen)
    print("\n--- final stats ---")
    for key, value in final_stats.items():
        print(f"{key}: {value}")
    print(f"reconnect_events: {len(events)}")

    return 0 if final_stats["frames_read"] > 0 else 1


def _now() -> str:
    return time.strftime("%H:%M:%S")


def _save_jpeg(frame, path: str) -> None:
    import cv2

    cv2.imwrite(path, frame)


if __name__ == "__main__":
    sys.exit(main())
