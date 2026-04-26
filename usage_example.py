import cv2
import numpy as np
import torch
import warnings
import os
import urllib.request
from ultralytics import YOLO

from tracker import BotSortReID
warnings.filterwarnings('ignore')

def download_sample_video(path):
    if not os.path.exists(path):
        url = "https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4"
        print(f"Downloading sample video from {url}...")
        urllib.request.urlretrieve(url, path)
        print("Download complete.")

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Initializing Re-ID Tracker on {device}...")
    tracker = BotSortReID(device=device, reid_sim_thresh=0.6) # Slight tweak for robustness

    print("Loading YOLOv8n (acting as simulated YOLOX layer)...")
    model = YOLO("yolov8n.pt") 

    video_source = "sample_video.mp4"
    download_sample_video(video_source)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open {video_source}.")
        return

    # Video Writer
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_path = "output_tracked.mp4"
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    unique_persons = set()
    reid_events = 0
    frame_id = 0

    print("Processing real sample video...")
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Send to mock detector (Filter person class=0)
        results = model(frame, classes=[0], verbose=False)
        boxes = results[0].boxes
        if len(boxes) > 0:
            detections = np.hstack([boxes.xyxy.cpu().numpy(), 
                                    boxes.conf.cpu().numpy()[:, None],
                                    boxes.cls.cpu().numpy()[:, None]])
        else:
            detections = np.empty((0, 6))

        # --- RE-ID TRACKER PIPELINE ---
        tracks = tracker.update(detections, frame)

        # --- VISUALIZATION OUTPUTS ---
        for t in tracks:
            tid = t['track_id']
            x1, y1, x2, y2 = map(int, t['bbox'])
            status = t['status']
            
            unique_persons.add(tid)
            if status == "reidentified":
                reid_events += 1
                color = (255, 0, 0)   # Blue
            elif status == "new":
                color = (0, 0, 255)   # Red
            else:
                color = (0, 255, 0)   # Green
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"ID: {tid} ({status})"
            cv2.putText(frame, label, (x1, max(0, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        out.write(frame)
        cv2.imshow("Tracking Viewport (Press ESC to close naturally)", frame)
        if cv2.waitKey(1) == 27:
            break
            
        frame_id += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n--- System Metrics ---")
    print(f"Total Unique Individuals Established: {len(unique_persons)}")
    print(f"Total Successful Re-Identifications: {reid_events}")
    print(f"Saved tracked output to: {os.path.abspath(out_path)}")

if __name__ == "__main__":
    main()
