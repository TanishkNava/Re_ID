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


def prompt_video_path() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    print("[main] Please select a video file from the dialog window...")
    file_path = filedialog.askopenfilename(
        title="Select Video File for Re-ID Tracker",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
    )
    if file_path:
        return Path(file_path)
    return None


def run_video_cli(
    video_path: Path,
    out_path: Path | None,
    max_frames: int | None,
    show: bool = True,
    loop: bool = False,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: cannot open video {video_path}", file=sys.stderr)
        sys.exit(1)

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if out_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps_src, (w, h))

    print(f"[main] Video: {video_path.name}  ({w}\u00d7{h} @ {fps_src:.0f} fps, {total} frames)")
    print("[main] Loading models (one-time)...")
    t0 = time.perf_counter()
    pipeline = Pipeline()
    print(f"[main] Pipeline ready in {time.perf_counter() - t0:.1f}s")
    if loop:
        print("[main] Loop mode ON - press Q/ESC to quit.")

    if show:
        cv2.namedWindow("Re-ID Tracker", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Re-ID Tracker", min(w, 1280), min(h, 720))

    loop_n   = 0
    quit_flag = False

    while not quit_flag:
        # ── Reset for this loop iteration ─────────────────────────────
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        pipeline.memory.reset()   # fresh Re-ID gallery each loop
        frame_i  = 0
        t_run    = time.perf_counter()
        fps_disp = 0.0
        t_fps    = time.perf_counter()
        loop_n  += 1
        if loop_n > 1:
            print(f"[main] --- Loop {loop_n} ---")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_i >= max_frames:
                break

            result = pipeline.process_frame(frame)
            vis    = pipeline.draw(frame.copy(), result)

            # ── FPS overlay ───────────────────────────────────────────────
            now = time.perf_counter()
            if now - t_fps >= 0.5:
                fps_disp = frame_i / max(now - t_run, 1e-3)
                t_fps    = now

            n_p, n_o = len(result["persons"]), len(result["objects"])
            loop_tag = f"  Loop:{loop_n}" if loop else ""
            overlay  = (
                f"FPS: {fps_disp:.1f}  |  Persons: {n_p}  Objects: {n_o}"
                f"  |  Frame: {frame_i}/{total if total > 0 else '?'}{loop_tag}"
            )
            cv2.rectangle(vis, (0, 0), (len(overlay) * 9 + 10, 28), (0, 0, 0), -1)
            cv2.putText(vis, overlay, (6, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 230, 230), 2)

            if writer:
                writer.write(vis)

            if show:
                cv2.imshow("Re-ID Tracker", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:   # q or ESC to quit
                    print("[main] Quit by user.")
                    quit_flag = True
                    break
                elif key == ord(" "):              # Space to pause
                    print("[main] Paused — press any key to resume.")
                    cv2.waitKey(0)

            frame_i += 1
            if frame_i % 30 == 0:
                elapsed = time.perf_counter() - t_run
                print(f"[main] frame={frame_i}  persons={n_p}  objects={n_o}"
                      f"  pipeline-fps={frame_i / elapsed:.1f}")

        if not loop or quit_flag:
            break

    cap.release()
    if writer:
        writer.release()
        print(f"[main] Wrote {out_path} ({frame_i} frames)")
    else:
        print(f"[main] Done ({frame_i} frames, no --out file)")

    if show:
        cv2.destroyAllWindows()


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
        "--no-show",
        action="store_true",
        help="Disable the live preview window (useful for headless / server runs)",
    )
    ap.add_argument(
        "--loop",
        action="store_true",
        help="Loop the video indefinitely (gallery resets each loop). Press Q/ESC to quit.",
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

    video = args.video
    if video is None:
        video = prompt_video_path()
        
    if video is None:
        print(
            "No video selected. Exiting.",
            file=sys.stderr,
        )
        sys.exit(1)
    if not video.is_file():
        print(f"Not a file: {video}", file=sys.stderr)
        sys.exit(1)

    run_video_cli(
        video.resolve(),
        args.out.resolve() if args.out else None,
        args.max_frames,
        show=not args.no_show,
        loop=args.loop,
    )


if __name__ == "__main__":
    main()
