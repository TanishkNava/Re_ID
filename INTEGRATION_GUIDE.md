# YOLOX + BotSortReID + ByteTrack Integration Guide

This document explains exactly how to drop the new Person Re-ID tracker into your existing YOLOX codebase while preserving your normal ByteTrack behavior for forklifts, pallets, and other objects.

## 1. Directory Setup
Simply copy the entire `tracker/` folder that was just created into your existing YOLOX project root directory.

```text
your_yolox_workspace/
├── tracker/            <-- Drop the new folder here
│   ├── __init__.py
│   ├── bot_sort_reid.py
│   ├── kalman_filter.py
│   ├── reid_extractor.py
│   └── reid_gallery.py
├── yolox/              <-- Your existing YOLOX code
├── tools/demo.py       <-- Your existing inference script
└── ...
```

## 2. Dependencies
Ensure your environment has the required libraries for the person tracking module. If you are using the fallback `resnet18` (which does not require compilation), ensure you have `torchvision` installed:
```bash
pip install torchvision numpy scipy opencv-python
```
If you wish to use the dedicated OSNet model from TorchReID (much higher accuracy for person re-id), run:
```bash
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
```

## 3. Modifying Your Inference Script

Open your main inference script (typically `tools/demo.py` or wherever your `while True:` video loop is located). You need to initialize **both** trackers and route the detections based on `class_id`.

### A. Imports & Initialization
Add this alongside your normal `BYTETracker` imports:

```python
from tracker.bot_sort_reid import BotSortReID
# from yolox.tracker.byte_tracker import BYTETracker  <-- (Your existing import)

# Initialize both trackers globally or before the video loop runs
device = "cuda" if torch.cuda.is_available() else "cpu"

person_tracker = BotSortReID(device=device, reid_sim_thresh=0.65)
byte_tracker = BYTETracker(args) # Your existing ByteTrack init
```

### B. Routing Detections in the Loop
Inside your frame processing loop, after YOLOX outputs detections `[x1, y1, x2, y2, conf, cls_id]`, filter them and feed them to their respective trackers.

```python
# ---------------------------------------------------------
# Inside your video inference loop:
# `outputs` is your YOLOX result [N, 6] format
# `frame` is your raw BGR cv2 image
# ---------------------------------------------------------

if outputs is not None and len(outputs) > 0:
    
    # 1. Split Detections by Class
    # Class 0 is Person. Classes 1-6 are Forklift, Pallet, Truck, etc.
    person_mask = outputs[:, 5] == 0
    other_mask = outputs[:, 5] != 0
    
    person_dets = outputs[person_mask]
    other_dets = outputs[other_mask]

    # 2. Update Person Tracker (with Re-ID)
    # The person tracker explicitly takes the raw frame to extract crops for Re-ID!
    person_tracks = person_tracker.update(person_dets, frame)

    # 3. Update Existing ByteTrack (Non-Persons)
    # Important: Standard ByteTrack formats detections as [x1, y1, x2, y2, score]
    other_online_targets = []
    if len(other_dets) > 0:
        # Standard ByteTrack call (check your specific BYTETracker signature)
        other_online_targets = byte_tracker.update(other_dets[:, :5], [frame.shape[0], frame.shape[1]], img_size)

    # ---------------------------------------------------------
    # 4. Rendering / Output Aggregation
    # ---------------------------------------------------------
    
    # Render Persons (BotSortReID formats output nicely as dicts)
    for t in person_tracks:
        tid = t['track_id']
        x1, y1, x2, y2 = t['bbox']
        status = t['status'] # 'tracked', 'new', 'reidentified'
        
        # Color explicitly if reidentified
        color = (255, 0, 0) if status == "reidentified" else (0, 255, 0)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, f"Person {tid}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)

    # Render Other Logistics Objects (Forklifts/Pallets from BYTETracker)
    for t in other_online_targets:
        tlwh = t.tlwh # Top, Left, Width, Height
        tid = t.track_id
        # Draw bounding boxes for other objects...
        x1, y1, w, h = tlwh
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x1+w), int(y1+h)), (0, 0, 255), 2)
        cv2.putText(frame, f"Object {tid}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)
```

## 4. Resetting Trackers Formally
If you are processing multiple separate videos in a row (e.g. `video1.mp4`, then `video2.mp4`), make sure you wipe the gallery memory between distinct videos:

```python
person_tracker.reset()
```

## Summary of How it Works Under the Hood
1. **Filtering**: By explicitly indexing `outputs[:, 5] == 0`, we ensure only human features are passed to the heavier embedding neural network.
2. **Resource Management**: Forklifts and pallets continue routing natively through ByteTracks high-speed KF matrix matching operations without wasting massive GPU computation scaling visual embeddings for industrial machinery.
3. **No Collision**: Because the track ID incrementers in both scripts run independently, human IDs might organically overlap with forklift IDs mathematically. (i.e. you might have both Person #5 and Forklift #5). Prefixing the labels with `Person-` and `Object-` during the visual draw strictly separates the metrics locally.
