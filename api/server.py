import cv2
import base64
import numpy as np
import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import yaml

from core.pipeline import Pipeline

_REPO_ROOT = Path(__file__).resolve().parent.parent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# Lazy init so `import api.server` / uvicorn startup does not load torch until first client.
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def encode_frame(frame: np.ndarray) -> str:
    """BGR frame → base64 JPEG string."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


@app.get("/")
async def root():
    """Browser UI (video + WebSocket client)."""
    html_path = _REPO_ROOT / "ui" / "index.html"
    if not html_path.is_file():
        return JSONResponse(
            {"error": "ui/index.html not found", "path": str(html_path)},
            status_code=404,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video for later WebSocket processing."""
    contents = await file.read()
    tmp_path = f"weights/_tmp_{file.filename}"
    with open(tmp_path, "wb") as f:
        f.write(contents)
    return {"status": "uploaded", "path": tmp_path, "filename": file.filename}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Real-time stream over WebSocket.
    Client sends: { "type": "frame", "data": "<base64 jpg>" }
                  { "type": "url", "url": "rtsp://..." }
                  { "type": "file", "path": "weights/_tmp_video.mp4" }
    Server sends: { "image": "<base64 jpg>", "persons": [...], "objects": [...] }
    """
    await websocket.accept()
    try:
        while True:
            msg = await websocket.receive_json()

            if msg.get("type") == "frame":
                img_data = base64.b64decode(msg["data"])
                np_arr   = np.frombuffer(img_data, np.uint8)
                frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    continue

                pl = get_pipeline()
                result      = pl.process_frame(frame)
                annotated   = pl.draw(frame.copy(), result)
                encoded     = encode_frame(annotated)

                await websocket.send_json({
                    "image"  : encoded,
                    "persons": result["persons"],
                    "objects": result["objects"],
                })

            elif msg.get("type") == "url":
                url = msg["url"]
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    await websocket.send_json({"error": f"Cannot open {url}"})
                    continue

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pl = get_pipeline()
                    result    = pl.process_frame(frame)
                    annotated = pl.draw(frame.copy(), result)
                    encoded   = encode_frame(annotated)

                    await websocket.send_json({
                        "image"  : encoded,
                        "persons": result["persons"],
                        "objects": result["objects"],
                    })
                    await asyncio.sleep(0.03)   # ~30fps
                cap.release()

            elif msg.get("type") == "file":
                path = msg["path"]
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    await websocket.send_json({"error": f"Cannot open {path}"})
                    continue

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    pl = get_pipeline()
                    result    = pl.process_frame(frame)
                    annotated = pl.draw(frame.copy(), result)
                    encoded   = encode_frame(annotated)

                    await websocket.send_json({
                        "image"  : encoded,
                        "persons": result["persons"],
                        "objects": result["objects"],
                    })
                    await asyncio.sleep(0.03)   # ~30fps
                cap.release()

    except WebSocketDisconnect:
        print("[WS] Client disconnected")


if __name__ == "__main__":
    with open("configs/pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)["api"]
    uvicorn.run("api.server:app", host=cfg["host"], port=cfg["port"], reload=False)