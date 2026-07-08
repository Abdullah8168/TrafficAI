"""
TrafficAI - REST API (Module 8)

Exposes the TrafficAI pipeline as a REST API using FastAPI.
Any external app can send an image and get detections back as JSON.

Install:
    pip install fastapi uvicorn python-multipart

Run:
    uvicorn api.api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /              → health check
    GET  /docs          → auto-generated Swagger UI
    POST /detect        → vehicle detection only
    POST /track         → detection + tracking (requires session_id)
    POST /analyze       → full pipeline (detect + count + speed + lane)
"""

import io
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import time
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse

from ultralytics import YOLO

from configs.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
)

from speed.speed import VEHICLE_REAL_WIDTH


# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="TrafficAI API",
    description="Real-time vehicle detection, tracking, counting and speed estimation.",
    version="1.0.0"
)

# Load model once at startup
model = YOLO(MODEL_PATH)

# ── Session store — keeps tracker state per session_id ───────────────────────
# Each session_id gets its own tracker history so persistent IDs work
# across multiple frames sent from the same client.
sessions = {}


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — decode uploaded image
# ─────────────────────────────────────────────────────────────────────────────

def decode_image(file_bytes: bytes) -> np.ndarray:

    arr   = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    return frame


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    """Health check — confirms API is running."""

    return {
        "status": "ok",
        "model" : MODEL_PATH,
        "info"  : "POST /detect, /track or /analyze with an image file."
    }


# ── /detect ──────────────────────────────────────────────────────────────────

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float      = Form(CONFIDENCE_THRESHOLD)
):
    """
    Detect vehicles in a single image.

    Returns bounding boxes, class labels and confidence scores.
    No tracking — each call is independent.

    Example (curl):
        curl -X POST http://localhost:8000/detect \\
             -F "file=@image.jpg" \\
             -F "conf=0.4"
    """

    raw    = await file.read()
    frame  = decode_image(raw)
    t0     = time.time()

    results = model.predict(
        source  = frame,
        conf    = conf,
        classes = VEHICLE_CLASSES,
        verbose = False
    )

    detections = []

    if results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            detections.append({
                "class_id"  : cls_id,
                "label"     : model.names[cls_id],
                "confidence": round(float(box.conf[0].item()), 3),
                "box"       : {
                    "x1": int(box.xyxy[0][0].item()),
                    "y1": int(box.xyxy[0][1].item()),
                    "x2": int(box.xyxy[0][2].item()),
                    "y2": int(box.xyxy[0][3].item()),
                }
            })

    return JSONResponse({
        "vehicles"      : len(detections),
        "inference_ms"  : round((time.time() - t0) * 1000, 1),
        "detections"    : detections
    })


# ── /track ───────────────────────────────────────────────────────────────────

@app.post("/track")
async def track(
    file      : UploadFile = File(...),
    session_id: str        = Form("default"),
    conf      : float      = Form(CONFIDENCE_THRESHOLD)
):
    """
    Detection + ByteTrack tracking with persistent IDs.

    Send frames from the same video stream using the same session_id
    to keep IDs stable across calls.

    Example (curl):
        curl -X POST http://localhost:8000/track \\
             -F "file=@frame.jpg" \\
             -F "session_id=cam1" \\
             -F "conf=0.4"
    """

    raw   = await file.read()
    frame = decode_image(raw)
    t0    = time.time()

    results = model.track(
        frame,
        persist      = True,
        tracker      = "bytetrack.yaml",
        conf         = conf,
        classes      = VEHICLE_CLASSES,
        verbose      = False
    )

    tracks = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):
            cls_id   = int(boxes_data.cls[i].item())
            track_id = int(boxes_data.id[i].item())
            conf_val = float(boxes_data.conf[i].item())
            box      = boxes_data.xyxy[i].tolist()

            x1, y1, x2, y2 = map(int, box)

            tracks.append({
                "track_id"  : track_id,
                "class_id"  : cls_id,
                "label"     : model.names[cls_id],
                "confidence": round(conf_val, 3),
                "box"       : {
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2
                },
                "centroid"  : {
                    "x": (x1 + x2) // 2,
                    "y": (y1 + y2) // 2
                }
            })

    return JSONResponse({
        "session_id"  : session_id,
        "vehicles"    : len(tracks),
        "inference_ms": round((time.time() - t0) * 1000, 1),
        "tracks"      : tracks
    })


# ── /analyze ─────────────────────────────────────────────────────────────────

@app.post("/analyze")
async def analyze(
    file      : UploadFile = File(...),
    session_id: str        = Form("default"),
    conf      : float      = Form(CONFIDENCE_THRESHOLD),
    fps       : float      = Form(30.0)
):
    """
    Full pipeline — detection + tracking + speed estimation.

    Send consecutive frames with the same session_id.
    Speed is calculated from displacement across frames.

    Example (curl):
        curl -X POST http://localhost:8000/analyze \\
             -F "file=@frame.jpg" \\
             -F "session_id=cam1" \\
             -F "fps=30"
    """

    raw   = await file.read()
    frame = decode_image(raw)
    t0    = time.time()

    # ── Init session state ────────────────────────────────────────────────────
    if session_id not in sessions:
        sessions[session_id] = {
            "frame_num"     : 0,
            "speed_hist"    : {},    # track_id → [(cx,cy,ts)]
            "speeds"        : {},    # track_id → km/h
            "scale_samples" : [],
            "pixels_per_m"  : None,
            "counts"        : {},    # label → int
            "crossed"       : {},    # track_id → label
            "prev_cy"       : {},    # track_id → last cy
        }

    state     = sessions[session_id]
    frame_num = state["frame_num"]
    ts        = frame_num / fps
    state["frame_num"] += 1

    # ── Track ─────────────────────────────────────────────────────────────────
    results = model.track(
        frame,
        persist  = True,
        tracker  = "bytetrack.yaml",
        conf     = conf,
        classes  = VEHICLE_CLASSES,
        verbose  = False
    )

    vehicles = []

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):

            cls_id   = int(boxes_data.cls[i].item())
            track_id = int(boxes_data.id[i].item())
            conf_val = float(boxes_data.conf[i].item())
            box      = boxes_data.xyxy[i].tolist()

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            bw = x2 - x1

            # ── Auto scale ────────────────────────────────────────────────────
            if bw > 20:
                rw = VEHICLE_REAL_WIDTH.get(cls_id, 1.8)
                state["scale_samples"].append(bw / rw)
                state["scale_samples"] = state["scale_samples"][-50:]
                state["pixels_per_m"]  = float(np.median(state["scale_samples"]))

            # ── Speed ─────────────────────────────────────────────────────────
            if track_id not in state["speed_hist"]:
                state["speed_hist"][track_id] = []

            state["speed_hist"][track_id].append((cx, cy, ts))
            state["speed_hist"][track_id] = state["speed_hist"][track_id][-15:]

            speed_kmh = None

            if state["pixels_per_m"] and len(state["speed_hist"][track_id]) >= 2:
                positions = state["speed_hist"][track_id]
                cx1, cy1, t1 = positions[0]
                cx2, cy2, t2 = positions[-1]
                dt = t2 - t1

                if dt >= 0.2:
                    dist_px = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
                    dist_m  = dist_px / state["pixels_per_m"]
                    kmh     = (dist_m / dt) * 3.6

                    if 1 < kmh < 200:
                        prev = state["speeds"].get(track_id, kmh)
                        state["speeds"][track_id] = round(0.7*prev + 0.3*kmh, 1)
                        speed_kmh = state["speeds"][track_id]

            vehicles.append({
                "track_id"  : track_id,
                "class_id"  : cls_id,
                "label"     : model.names[cls_id],
                "confidence": round(conf_val, 3),
                "box"       : {"x1":x1,"y1":y1,"x2":x2,"y2":y2},
                "centroid"  : {"x":cx,"y":cy},
                "speed_kmh" : speed_kmh
            })

    # ── Speed summary ──────────────────────────────────────────────────────
    all_speeds = [v["speed_kmh"] for v in vehicles if v["speed_kmh"]]

    speed_summary = {}

    if all_speeds:
        speed_summary = {
            "avg_kmh": round(float(np.mean(all_speeds)), 1),
            "max_kmh": round(float(np.max(all_speeds)), 1),
            "min_kmh": round(float(np.min(all_speeds)), 1),
        }

    return JSONResponse({
        "session_id"   : session_id,
        "frame"        : frame_num,
        "vehicles"     : len(vehicles),
        "inference_ms" : round((time.time() - t0) * 1000, 1),
        "speed_summary": speed_summary,
        "tracks"       : vehicles
    })


# ── Clear session ─────────────────────────────────────────────────────────────

@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    """Clear tracker state for a session."""

    if session_id in sessions:
        del sessions[session_id]
        return {"status": "cleared", "session_id": session_id}

    return {"status": "not_found", "session_id": session_id}