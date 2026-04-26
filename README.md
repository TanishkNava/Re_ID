# Warehouse Person Re-ID & Tracking System

Real-time person re-identification and multi-object tracking for warehouse CCTV footage.

**Stack:** YOLOX-s (detection) + ByteTrack (tracking) + OSNet (Re-ID) + CUDA

---

## Quick Start

```bash
# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-deps
pip install torchreid gdown pycocotools lapx scipy opencv-python PyYAML

# Run with live window
python main.py

# Loop continuously (press Q to quit)
python main.py --loop

# Save annotated output
python main.py --out output_annotated.mp4

# Web dashboard
python main.py --serve   # open http://localhost:8000
```

## Controls (Live Window)
| Key | Action |
|---|---|
| `Q` / `ESC` | Quit |
| `Space` | Pause / Resume |

## CLI Flags
| Flag | Description |
|---|---|
| `--video PATH` | Input video (default: first .mp4 in project root) |
| `--out PATH` | Save annotated output video |
| `--max-frames N` | Stop after N frames |
| `--no-show` | Headless mode (no window) |
| `--loop` | Replay video continuously |
| `--serve` | Start web dashboard |

## Config
Edit `configs/pipeline_config.yaml` to tune thresholds.

| Parameter | Default | Effect |
|---|---|---|
| `conf_threshold` | 0.35 | Detection sensitivity |
| `cosine_threshold` | 0.72 | Re-ID strictness |
| `track_buffer` | 90 | Frames before track is dropped |
| `ttl_seconds` | 300 | Gallery entry lifetime |

## How Re-ID Works
1. YOLOX detects all persons each frame
2. ByteTrack assigns short-term `track_id` (resets when person leaves)
3. OSNet extracts 512-d appearance embedding from each person crop
4. Gallery matches embedding → returns stable `person_N` ID
5. Frame-level uniqueness: same ID cannot appear on two people at once
6. When person re-appears, cosine match to gallery restores their original ID
