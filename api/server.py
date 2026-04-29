import asyncio
import base64
import cv2
import numpy as np
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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = Pipeline()
    return _pipeline


def encode_frame(frame: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(buf).decode("utf-8")


@app.get("/")
async def root():
    html_path = _REPO_ROOT / "ui" / "index.html"
    if not html_path.is_file():
        return JSONResponse(
            {"error": "ui/index.html not found", "path": str(html_path)},
            status_code=404,
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Save uploaded video under weights/ and return an absolute path for OpenCV."""
    contents = await file.read()
    safe_name = Path(file.filename or "video.bin").name
    dest_dir = _REPO_ROOT / "weights"
    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_dir / f"_upload_{safe_name}"
    tmp_path.write_bytes(contents)
    abs_path = str(tmp_path.resolve())
    return {"status": "uploaded", "path": abs_path, "filename": safe_name}


async def _drain_control_queue(
    queue: asyncio.Queue,
    *,
    paused: list[bool],
    stop: list[bool],
) -> None:
    """Process pending control / new-stream messages. `paused` and `stop` are length-1 lists for mutability."""
    while True:
        try:
            msg = queue.get_nowait()
        except asyncio.QueueEmpty:
            break
        if not isinstance(msg, dict):
            continue
        if msg.get("__disconnect__"):
            stop[0] = True
            await queue.put(msg)
            break
        t = msg.get("type")
        if t == "control":
            act = (msg.get("action") or "").lower()
            if act == "pause":
                paused[0] = True
            elif act in ("play", "resume"):
                paused[0] = False
            elif act == "stop":
                stop[0] = True
        elif t in ("file", "url"):
            stop[0] = True
            await queue.put(msg)


async def _stream_video_capture(
    websocket: WebSocket,
    queue: asyncio.Queue,
    cap: cv2.VideoCapture,
) -> None:
    pl = get_pipeline()
    paused = [False]
    stop = [False]
    try:
        while True:
            await _drain_control_queue(queue, paused=paused, stop=stop)
            if stop[0]:
                break
            while paused[0] and not stop[0]:
                await asyncio.sleep(0.04)
                await _drain_control_queue(queue, paused=paused, stop=stop)
            if stop[0]:
                break
            ret, frame = cap.read()
            if not ret:
                break
            result = pl.process_frame(frame)
            annotated = pl.draw(frame.copy(), result)
            encoded = encode_frame(annotated)
            await websocket.send_json(
                {
                    "image": encoded,
                    "persons": result["persons"],
                    "objects": result["objects"],
                }
            )
            await asyncio.sleep(0.03)
    finally:
        cap.release()
    await websocket.send_json({"event": "stream_end"})


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Client → server JSON:
      { "type": "frame", "data": "<base64 jpg>" }
      { "type": "url", "url": "..." }
      { "type": "file", "path": "/abs/path/video.mp4" }
      { "type": "control", "action": "pause" | "play" | "resume" | "stop" }
    Server → client:
      { "image", "persons", "objects" } per frame
      { "event": "stream_end" } when a file/url stream finishes
      { "error": "..." }
    """
    await websocket.accept()
    q: asyncio.Queue = asyncio.Queue()

    async def reader():
        try:
            while True:
                msg = await websocket.receive_json()
                await q.put(msg)
        except WebSocketDisconnect:
            await q.put({"__disconnect__": True})
        except Exception:
            await q.put({"__disconnect__": True})

    reader_task = asyncio.create_task(reader())

    async def handle_single_frame(msg: dict) -> None:
        img_data = base64.b64decode(msg["data"])
        np_arr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if frame is None:
            return
        pl = get_pipeline()
        result = pl.process_frame(frame)
        annotated = pl.draw(frame.copy(), result)
        encoded = encode_frame(annotated)
        await websocket.send_json(
            {
                "image": encoded,
                "persons": result["persons"],
                "objects": result["objects"],
            }
        )

    try:
        while True:
            msg = await q.get()
            if msg.get("__disconnect__"):
                break
            t = msg.get("type")
            if t == "control":
                continue
            if t == "frame":
                await handle_single_frame(msg)
            elif t == "url":
                url = msg.get("url") or ""
                cap = cv2.VideoCapture(url)
                if not cap.isOpened():
                    await websocket.send_json({"error": f"Cannot open {url}"})
                    continue
                await _stream_video_capture(websocket, q, cap)
            elif t == "file":
                path = msg.get("path") or ""
                cap = cv2.VideoCapture(path)
                if not cap.isOpened():
                    await websocket.send_json({"error": f"Cannot open {path}"})
                    continue
                await _stream_video_capture(websocket, q, cap)
    finally:
        reader_task.cancel()
        try:
            await reader_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    with open("configs/pipeline_config.yaml") as f:
        cfg = yaml.safe_load(f)["api"]
    uvicorn.run("api.server:app", host=cfg["host"], port=cfg["port"], reload=False)
