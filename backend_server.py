"""
Guardian Drive™ v4.2 — FastAPI WebSocket Backend
Pushes full pipeline payload to dashboard every 1.2 seconds.
"""
import asyncio, json, time, subprocess
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Guardian Drive v4.2")
app.add_middleware(CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Connected dashboard clients
_clients: list[WebSocket] = []
# Last payload from main.py
_last_payload: dict = {}

@app.get("/health")
async def health():
    return {"status": "ok", "ready": True, "clients": len(_clients)}

@app.post("/push")
async def push(payload: dict):
    """main.py posts payload here every window."""
    global _last_payload
    _last_payload = payload
    dead = []
    for ws in _clients:
        try:
            await ws.send_text(json.dumps(payload, default=str))
        except:
            dead.append(ws)
    for ws in dead:
        _clients.remove(ws)
    return {"ok": True, "clients": len(_clients)}

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _clients.append(ws)
    try:
        # Send last known payload immediately on connect
        if _last_payload:
            await ws.send_text(json.dumps(_last_payload, default=str))
        while True:
            # Keep alive — dashboard sends nothing, just receives
            await asyncio.sleep(30)
            await ws.send_text(json.dumps({"ping": True}))
    except WebSocketDisconnect:
        pass
    except:
        pass
    finally:
        if ws in _clients:
            _clients.remove(ws)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
