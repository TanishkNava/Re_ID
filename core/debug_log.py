import json
import time
from pathlib import Path
from typing import Any


_LOG_PATH = Path(__file__).resolve().parent.parent / ".cursor" / "debug-6f6a1e.log"
_SESSION_ID = "6f6a1e"


def _safe(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return repr(obj)


def log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict | None = None) -> None:
    payload = {
        "sessionId": _SESSION_ID,
        "runId": run_id,
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": {k: _safe(v) for k, v in (data or {}).items()},
        "timestamp": int(time.time() * 1000),
    }
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        # Never crash the app due to debug logging.
        pass

