import argparse
import sys
import time
from pathlib import Path

import cv2
import yaml

# Local YOLOX (pip install -e fails on some Python versions due to onnx-simplifier pin)
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT / "YOLOX"))

from core.pipeline import Pipeline


def _default_video_path() -> Path | None:
    for name in (
        "freepik_i-need-a-realistic-video-_2801472922.mp4",
        "sample.mp4",
        "input.mp4",
    ):
        p = _ROOT / name
        if p.is_file():
            return p
    for p in sorted(_ROOT.glob("*.mp4")):
        return p
    return None


def run_video_cli(video_path: Path, out_path: Path | None, max_frames: int | None) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}", file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    print("[main] Loading models (one-time)...")
    t0 = time.perf_counter()
    pipeline = Pipeline()
    print(f"[main] Pipeline ready in {time.perf_counter() - t0:.1f}s")

    frame_i = 0
    t_run = time.perf_counter()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames is not None and frame_i >= max_frames:
            break

        result = pipeline.process_frame(frame)
        vis = pipeline.draw(frame.copy(), result)

        if writer:
            writer.write(vis)

        frame_i += 1
        if frame_i % 30 == 0:
            n_p, n_o = len(result["persons"]), len(result["objects"])
            elapsed = time.perf_counter() - t_run
            print(f"[main] frame={frame_i} persons={n_p} objects={n_o} fps={frame_i / elapsed:.1f}")

    cap.release()
    if writer:
        writer.release()
        print(f"[main] Wrote {out_path} ({frame_i} frames)")
    else:
        print(f"[main] Done ({frame_i} frames, no --out file)")


def run_server() -> None:
    import uvicorn

    with open(_ROOT / "configs/pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)["api"]
    uvicorn.run("api.server:app", host=cfg["host"], port=cfg["port"], reload=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Re-ID pipeline: run on a video locally, or serve API only with --serve.",
        epilog='Paths with spaces must be quoted, e.g. --video "/path/Video-for validation/file.mp4"',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Input video path (default: first .mp4 in project root)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional annotated output video (.mp4)",
    )
    ap.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (debug)",
    )
    ap.add_argument(
        "--serve",
        action="store_true",
        help="Start web server: open http://127.0.0.1:8000/ for UI (models load on first WebSocket use)",
    )
    args = ap.parse_args()

    if args.serve:
        run_server()
        return

    video = args.video or _default_video_path()
    if video is None:
        print(
            "No input video. Pass --video path/to.mp4 or place a .mp4 in the project root.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not video.is_file():
        print(f"Not a file: {video}", file=sys.stderr)
        sys.exit(1)

    run_video_cli(video.resolve(), args.out.resolve() if args.out else None, args.max_frames)


if __name__ == "__main__":
    main()
