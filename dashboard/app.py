"""
TrafficAI - Live Dashboard (Module 7)

Runs all modules together in a Streamlit dashboard.

Usage:
    streamlit run dashboard/app.py
"""

import sys
import os

# Use all CPU cores for ONNX Runtime and numpy operations
_cpu = str(os.cpu_count() or 4)
os.environ.setdefault("OMP_NUM_THREADS",        _cpu)
os.environ.setdefault("OPENBLAS_NUM_THREADS",   _cpu)
os.environ.setdefault("MKL_NUM_THREADS",        _cpu)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import base64
import cv2
import numpy as np
import streamlit as st
import threading
import queue
import time
from ultralytics import YOLO

from configs.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    OUTPUT_DIR
)

from counting.count import VehicleCounter
from speed.speed import SpeedEstimator
from lane.lane import LaneDetector
from ocr.ocr import ANPRDetector
from Density.density import DensityEstimator
from zone.zone import ZoneFilter


# ── Cache model so it loads ONCE and reuses on every run ──────────────────────
@st.cache_resource
def load_model():
    # Use ONNX if available (faster CPU inference), else fall back to .pt
    onnx_path = MODEL_PATH.replace(".pt", ".onnx")
    if os.path.exists(onnx_path):
        return YOLO(onnx_path)
    return YOLO(MODEL_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrafficAI Dashboard",
    page_icon="🚗",
    layout="wide"
)

# ── Session state defaults ────────────────────────────────────────────────────
for _k, _v in [("app_state", "idle"), ("zone_polygon", None),
               ("saved_video_path", None), ("first_frame", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

st.title("🚗 TrafficAI — Smart Traffic Monitoring")
st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Settings
# ─────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Settings")

source_type = st.sidebar.radio(
    "Input Source",
    ["Image", "Video File", "Webcam"]
)

video_path  = None
image_input = None   # holds numpy BGR array when source is Image

if source_type == "Image":
    uploaded_img = st.sidebar.file_uploader(
        "Upload Image", type=["jpg", "jpeg", "png", "bmp"]
    )
    if uploaded_img:
        file_bytes = np.frombuffer(uploaded_img.read(), np.uint8)
        image_input = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.sidebar.success(f"Loaded: {uploaded_img.name}")

elif source_type == "Video File":
    uploaded = st.sidebar.file_uploader(
        "Upload Video", type=["mp4", "avi", "mov"]
    )

    if uploaded:
        os.makedirs("data/videos", exist_ok=True)
        video_path = f"data/videos/{uploaded.name}"

        # Only write if not already saved — prevents rewriting mid-read on reruns
        if not os.path.exists(video_path) or os.path.getsize(video_path) != uploaded.size:
            with open(video_path, "wb") as f:
                f.write(uploaded.read())

        st.sidebar.success(f"Loaded: {uploaded.name}")

else:
    # IP Webcam (e.g. Android "IP Webcam" app)
    ip_cam_url = st.sidebar.text_input(
        "IP Camera URL",
        value="http://192.168.1.x:8080/video",
        help="Open IP Webcam app → note the IP shown → replace x with it"
    )
    video_path = ip_cam_url.strip() if ip_cam_url.strip() else None

    st.sidebar.caption(
        "💡 Common URLs:\n"
        "- IP Webcam (Android): `http://IP:8080/video`\n"
        "- DroidCam: `http://IP:4747/video`\n"
        "- RTSP: `rtsp://IP:8080/h264`"
    )

st.sidebar.markdown("---")
st.sidebar.subheader("Modules")

enable_counting = st.sidebar.checkbox("Vehicle Counting",    value=True,
                                      disabled=(source_type == "Image"))
enable_speed    = st.sidebar.checkbox("Speed Estimation",    value=True,
                                      disabled=(source_type == "Image"))
enable_lane     = st.sidebar.checkbox("Lane Detection",      value=True)
enable_anpr     = st.sidebar.checkbox("ANPR (Plates)",       value=False)

conf_threshold  = st.sidebar.slider(
    "Confidence", 0.1, 0.9, 0.20, 0.05
)

line_position   = st.sidebar.slider(
    "Counting Line Position", 0.1, 0.9, 0.5, 0.05
)

direction       = st.sidebar.selectbox(
    "Traffic Direction",
    ["down", "up"]
)

st.sidebar.markdown("---")

run_button  = st.sidebar.button("▶ Start", use_container_width=True)
stop_button = st.sidebar.button("⏹ Stop",  use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

col_video, col_stats = st.columns([2, 1])

with col_video:
    st.subheader("📹 Live Feed")
    video_placeholder = st.empty()

with col_stats:
    st.subheader("📊 Live Stats")

    # Traffic Density
    st.markdown("**Traffic Density**")
    density_placeholder = st.empty()

    st.markdown("---")

    # Counts
    st.markdown("**Vehicle Counts**")
    count_placeholder = st.empty()

    st.markdown("---")

    # Speeds
    st.markdown("**Speed (km/h)**")
    speed_placeholder = st.empty()

    st.markdown("---")

    # Violations
    st.markdown("**Lane Violations**")
    violation_placeholder = st.empty()

    st.markdown("---")

    # Plates
    st.markdown("**Plates Detected**")
    plate_placeholder = st.empty()

st.markdown("---")
status_placeholder = st.empty()


# ─────────────────────────────────────────────────────────────────────────────
# SIMPLE IoU TRACKER — replaces ByteTrack to avoid ONNX compatibility issues
# ─────────────────────────────────────────────────────────────────────────────

class _SimpleTracker:
    """
    Lightweight IoU-based tracker.
    Assigns consistent integer IDs across frames without needing ByteTrack.
    Works reliably with ONNX inference.
    """

    def __init__(self, iou_thresh=0.25, max_age=30):
        self._tracks   = {}   # id → {box, age, cls_id}
        self._next_id  = 1
        self._iou_thr  = iou_thresh
        self._max_age  = max_age

    def update(self, boxes, cls_ids):
        """
        boxes   : list of [x1,y1,x2,y2]
        cls_ids : list of int (same length)
        Returns : list of track IDs (same length as boxes)
        """
        # Age existing tracks; remove stale ones
        stale = [tid for tid, t in self._tracks.items()
                 if t["age"] >= self._max_age]
        for tid in stale:
            del self._tracks[tid]
        for tid in self._tracks:
            self._tracks[tid]["age"] += 1

        if not boxes:
            return []

        track_ids  = list(self._tracks.keys())
        track_boxes = [self._tracks[tid]["box"] for tid in track_ids]

        assigned   = {}   # box_idx → track_id
        used_trk   = set()

        if track_boxes:
            # Build IoU matrix
            iou_mat = np.zeros((len(boxes), len(track_boxes)), dtype=np.float32)
            for i, b in enumerate(boxes):
                for j, tb in enumerate(track_boxes):
                    iou_mat[i, j] = self._iou(b, tb)

            # Greedy best-match
            flat = np.argsort(iou_mat, axis=None)[::-1]
            for k in flat:
                i, j = divmod(int(k), len(track_boxes))
                if iou_mat[i, j] < self._iou_thr:
                    break
                if i in assigned or track_ids[j] in used_trk:
                    continue
                assigned[i]           = track_ids[j]
                used_trk.add(track_ids[j])

        result = []
        for i, (box, cls_id) in enumerate(zip(boxes, cls_ids)):
            if i in assigned:
                tid = assigned[i]
                self._tracks[tid] = {"box": box, "age": 0, "cls_id": cls_id}
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"box": box, "age": 0, "cls_id": cls_id}
            result.append(tid)

        return result

    @staticmethod
    def _iou(a, b):
        xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
        xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        if inter == 0:
            return 0.0
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / (area_a + area_b - inter)


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE CLASS — combines all active modules
# ─────────────────────────────────────────────────────────────────────────────

class TrafficPipeline:

    def __init__(self, conf, line_pos, direction,
                 counting, speed, lane, anpr, zone_polygon=None):

        self.model           = load_model()   # uses cached model — no reload

        # Reset only the tracker state, not the whole predictor
        # (resetting predictor breaks ONNX re-initialisation)
        try:
            if (hasattr(self.model, 'predictor') and
                    self.model.predictor is not None and
                    hasattr(self.model.predictor, 'trackers')):
                self.model.predictor.trackers = []
        except Exception:
            pass

        self.conf            = conf
        self.vehicle_classes = VEHICLE_CLASSES

        self.enable_counting = counting
        self.enable_speed    = speed
        self.enable_lane     = lane
        self.enable_anpr     = anpr

        # ── Module state ──────────────────────────────────────────────────────
        self.counts       = {}       # label → int
        self.crossed      = {}       # track_id → label
        self.prev_pos     = {}       # track_id → last cy
        self.track_origin = {}       # track_id → first cy when track appeared
        self._cross_log   = {}       # label → [(cx, frame_num)] recent crossings

        self.speeds     = {}       # track_id → km/h
        self.speed_hist = {}       # track_id → [(cx,cy,t)]
        self._scale_samples = []
        self._pixels_per_m  = None

        self.direction         = direction
        self.lane_detector     = LaneDetector(direction)

        self.plates     = {}       # track_id → plate text
        self._ocr_reader = None

        # OCR runs in its own thread so it never blocks processing
        self._ocr_queue  = queue.Queue(maxsize=20)
        self._ocr_stop   = threading.Event()
        self._ocr_thread = None

        if anpr:
            self._start_ocr_worker()

        # Counting line
        self.line_pos = line_pos

        # ── Traffic density ───────────────────────────────────────────────────
        self.density = DensityEstimator()

        # ── Detection zone filter ─────────────────────────────────────────────
        self.zone = ZoneFilter(zone_polygon)

        self.frame_num = 0

        # Simple IoU tracker — replaces ByteTrack to avoid ONNX compatibility issues
        self._tracker = _SimpleTracker()

        # Inference resolution — frames are pre-resized to this width before
        # being passed to model.predict(). Eliminates ultralytics letterboxing
        # a large frame down to 640, which costs 50-100ms on 1080p video.
        self._infer_width = 320   # matches ONNX export size (320×320)

    @property
    def violations(self):
        """Delegate to LaneDetector so pipeline.violations still works."""
        return self.lane_detector.violations

    # ── Process one frame ─────────────────────────────────────────────────────
    def process(self, frame, fps=30):

        h, w = frame.shape[:2]
        line_y = int(h * self.line_pos)
        ts = self.frame_num / fps
        self.frame_num += 1

        # Draw counting line
        if self.enable_counting:
            cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        # Lane detection — run every 10 frames, cache for violation checking
        # Lines are intentionally not drawn on the frame (invisible detection)
        if self.enable_lane:
            if self.frame_num % 10 == 0:
                self._cached_lane_lines = self.lane_detector.detect_lines(frame, h, w)

        # ── Pre-resize frame before inference ─────────────────────────────────
        # If frame is wider than _infer_width, shrink it first.
        # ultralytics letterboxes internally anyway — doing it here in one fast
        # cv2.resize() call saves 50-100ms on 1080p video vs letting ultralytics
        # do it on the full-resolution numpy array.
        if w > self._infer_width:
            scale     = self._infer_width / w
            infer_h   = int(h * scale)
            infer_frm = cv2.resize(frame, (self._infer_width, infer_h),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            infer_frm = frame   # already small enough

        # Run detection on every frame — no skipping, full accuracy
        try:
            results = self.model.predict(
                infer_frm,
                conf=self.conf,
                iou=0.5,
                max_det=50,
                verbose=False
            )
        except Exception as e:
            cv2.putText(frame, f"Detect error: {str(e)[:60]}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)
            return frame

        # Scale boxes back to original frame coordinates if we pre-resized
        scale_x = w / infer_frm.shape[1]
        scale_y = h / infer_frm.shape[0]

        det_boxes, det_cls, det_conf = [], [], []
        if results[0].boxes is not None and len(results[0].boxes):
            boxes_data = results[0].boxes
            for i in range(len(boxes_data)):
                cls_id = int(boxes_data.cls[i].item())
                if cls_id not in self.vehicle_classes:
                    continue
                x1, y1, x2, y2 = boxes_data.xyxy[i].tolist()
                # Scale back to original resolution for correct drawing
                x1, x2 = x1 * scale_x, x2 * scale_x
                y1, y2 = y1 * scale_y, y2 * scale_y
                det_boxes.append([x1, y1, x2, y2])
                det_cls.append(cls_id)
                det_conf.append(float(boxes_data.conf[i].item()))

        track_ids = self._tracker.update(det_boxes, det_cls)

        zone_boxes = []   # boxes of vehicles that pass the zone filter

        for idx, (box, cls_id, conf_val, track_id) in enumerate(
                zip(det_boxes, det_cls, det_conf, track_ids)):

            x1, y1, x2, y2 = map(int, box)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            label = self.model.names[cls_id]

            # ── Zone filter — skip vehicles outside the selected zone ──────────
            if not self.zone.is_inside(cx, cy):
                continue

            zone_boxes.append(box)

            # ── Counting ──────────────────────────────────────────────────────
            if self.enable_counting:
                # Record where this track first appeared
                if track_id not in self.track_origin:
                    self.track_origin[track_id] = cy

                if track_id in self.prev_pos and track_id not in self.crossed:
                    prev_y    = self.prev_pos[track_id]
                    origin_y  = self.track_origin[track_id]

                    # Only count if the vehicle STARTED clearly on the opposite
                    # side of the line (20 px margin). This prevents trucks/cars
                    # that appear right at the line — or whose track IDs reset
                    # near the line — from being counted spuriously.
                    margin       = 20
                    started_above = origin_y < line_y - margin
                    started_below = origin_y > line_y + margin

                    crossed_down = started_above and (prev_y < line_y) and (cy >= line_y)
                    crossed_up   = started_below and (prev_y > line_y) and (cy <= line_y)

                    if crossed_down or crossed_up:
                        # ── Spatial deduplication ─────────────────────────────
                        # If the same vehicle lost its track ID near the line and
                        # got a new one, it would cross again as a "new" vehicle.
                        # Block any crossing within 60px of a recent crossing of
                        # the same class within the last 45 frames.
                        cooldown   = 45   # frames
                        proximity  = 60   # pixels
                        now        = self.frame_num
                        recent     = self._cross_log.get(label, [])
                        recent     = [(x, t) for x, t in recent
                                      if now - t < cooldown]
                        duplicate  = any(abs(cx - x) < proximity
                                         for x, t in recent)

                        self.crossed[track_id] = label   # always mark crossed
                        if not duplicate:
                            self.counts[label] = self.counts.get(label, 0) + 1
                            recent.append((cx, now))

                        self._cross_log[label] = recent

                self.prev_pos[track_id] = cy

            # ── Speed ─────────────────────────────────────────────────────────
            speed_str = ""

            if self.enable_speed:
                bw = x2 - x1
                if bw > 20:
                    from speed.speed import VEHICLE_REAL_WIDTH
                    rw = VEHICLE_REAL_WIDTH.get(cls_id, 1.8)
                    self._scale_samples.append(bw / rw)
                    self._scale_samples = self._scale_samples[-50:]
                    self._pixels_per_m  = float(np.median(self._scale_samples))

                if track_id not in self.speed_hist:
                    self.speed_hist[track_id] = []

                self.speed_hist[track_id].append((cx, cy, ts))
                self.speed_hist[track_id] = self.speed_hist[track_id][-15:]

                kmh = self._calc_speed(self.speed_hist[track_id])

                if kmh is not None:
                    prev = self.speeds.get(track_id, kmh)
                    self.speeds[track_id] = 0.7 * prev + 0.3 * kmh

                if track_id in self.speeds:
                    speed_str = f"{self.speeds[track_id]:.1f} km/h"

            # ── Lane ──────────────────────────────────────────────────────────
            if self.enable_lane:
                cached = getattr(self, "_cached_lane_lines", [])
                self.lane_detector.update(track_id, cx, cy, cached, w)

            # ── ANPR ──────────────────────────────────────────────────────────
            # Submit ROI to the OCR worker thread; never block here
            if self.enable_anpr and self.frame_num % 10 == 0:
                vehicle_roi = frame[y1:y2, x1:x2]
                if vehicle_roi.size > 0:
                    try:
                        self._ocr_queue.put_nowait(
                            (track_id, vehicle_roi.copy())
                        )
                    except queue.Full:
                        pass

            # ── Draw box ──────────────────────────────────────────────────────
            violation   = self.lane_detector.get_violation(track_id)
            plate_text  = self.plates.get(track_id, "")

            color = (0, 0, 255) if violation else self._class_color(cls_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = f"{label} #{track_id}"
            if speed_str:
                label_text += f"  {speed_str}"

            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            )

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - th - 8)),
                (x1 + tw + 4, y1),
                color, -1
            )

            cv2.putText(
                frame, label_text,
                (x1 + 2, max(th, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 2
            )

            if violation:
                cv2.putText(
                    frame, violation,
                    (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255), 2
                )

            if plate_text:
                cv2.putText(
                    frame, f"Plate: {plate_text}",
                    (x1, y2 + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2
                )

        # ── Zone overlay ──────────────────────────────────────────────────────
        if self.zone.active:
            self.zone.draw(frame)

        # ── Traffic density — use zone boxes and zone area if zone is active ──
        zone_area = None
        if self.zone.active:
            zone_area = self.zone.polygon_area()
        self.density.update(zone_boxes, h, w, zone_area=zone_area)

        # Overlay density badge on frame (top-right corner)
        badge_text  = self.density.badge_text()
        badge_color = self.density.badge_color()
        (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        bx = w - bw - 14
        cv2.rectangle(frame, (bx - 4, 8), (bx + bw + 4, 8 + bh + 10),
                      badge_color, -1)
        cv2.putText(frame, badge_text, (bx, 8 + bh + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return frame

    # ── OCR worker (runs in its own thread) ──────────────────────────────────
    def _start_ocr_worker(self):

        def worker():
            while not self._ocr_stop.is_set():
                try:
                    track_id, roi = self._ocr_queue.get(timeout=0.5)
                    text = self._read_plate(roi)
                    if text and len(text) > len(self.plates.get(track_id, "")):
                        self.plates[track_id] = text
                    self._ocr_queue.task_done()
                except queue.Empty:
                    continue
                except Exception:
                    continue

        self._ocr_thread = threading.Thread(target=worker, daemon=True)
        self._ocr_thread.start()

    def stop_ocr_worker(self):
        self._ocr_stop.set()

    # ── Speed math ────────────────────────────────────────────────────────────
    def _calc_speed(self, positions):

        if self._pixels_per_m is None or len(positions) < 2:
            return None

        cx1, cy1, t1 = positions[0]
        cx2, cy2, t2 = positions[-1]
        dt = t2 - t1

        if dt < 0.2:
            return None

        dist_px  = np.sqrt((cx2-cx1)**2 + (cy2-cy1)**2)
        dist_m   = dist_px / self._pixels_per_m
        kmh      = (dist_m / dt) * 3.6

        return kmh if 1 < kmh < 200 else None

    # ── ANPR ──────────────────────────────────────────────────────────────────
    def _detect_plate_region(self, vehicle_roi):
        """
        Try to find a licence-plate-shaped rectangle inside the vehicle ROI.
        Returns the cropped plate image, or the bottom-third of the ROI as fallback.
        """
        h, w = vehicle_roi.shape[:2]

        gray  = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        cnts, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts    = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

        for c in cnts:
            peri   = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)

            if len(approx) == 4:
                bx, by, bw, bh = cv2.boundingRect(approx)
                aspect = bw / max(bh, 1)

                # Licence plates are roughly 2:1 to 5:1 wide-to-tall
                if 1.5 < aspect < 6.0 and bw > 40 and bh > 10:
                    return vehicle_roi[by:by+bh, bx:bx+bw]

        # Fallback: bottom third of vehicle (front/rear bumper area)
        return vehicle_roi[int(h * 0.6):, :]

    def _enhance_plate(self, plate_img):
        """Resize, CLAHE, Otsu threshold to help EasyOCR."""
        if plate_img.size == 0:
            return None

        h, w = plate_img.shape[:2]
        if h < 10 or w < 20:
            return None

        # Scale up if very small
        scale = max(1, 60 // h)
        plate_img = cv2.resize(plate_img, (w * scale, h * scale),
                               interpolation=cv2.INTER_CUBIC)

        gray  = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray  = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh

    def _read_plate(self, roi):

        import re

        try:
            if self._ocr_reader is None:
                import easyocr
                self._ocr_reader = easyocr.Reader(["en"], gpu=False,
                                                  verbose=False)

            plate_region = self._detect_plate_region(roi)
            enhanced     = self._enhance_plate(plate_region)

            if enhanced is None:
                return ""

            results = self._ocr_reader.readtext(enhanced, detail=1,
                                                paragraph=False)

            texts = [
                re.sub(r"[^A-Z0-9\-]", "", t.upper())
                for _, t, c in results if c > 0.25
            ]
            text = " ".join([t for t in texts if len(t) >= 3])

            return text

        except Exception:
            return ""

    def _class_color(self, cls_id):
        return {2:(0,255,0), 3:(0,165,255), 5:(255,0,0), 7:(0,0,255)}.get(
            cls_id, (200,200,200)
        )


# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────

def processing_thread(pipeline, cap, fps, frame_queue, stop_event, is_ip=False):
    """
    Runs in background thread.
    Processes and displays EVERY frame — no frame skipping.
    Sleep after each frame ensures display rate never exceeds target_fps,
    even on fast hardware where YOLO inference is quick.
    """
    target_fps    = min(fps, 10)          # cap display at 10 fps
    frame_interval = 1.0 / target_fps    # minimum seconds between frames

    while cap.isOpened() and not stop_event.is_set():

        _t_start = time.time()            # track how long this frame takes

        # For IP cameras: flush the internal buffer so we always get the
        # latest frame, not a stale one queued up from seconds ago.
        if is_ip:
            cap.grab()   # discard buffered frame
            cap.grab()
            cap.grab()

        success, frame = cap.read()

        if not success or frame is None:
            if is_ip:
                # Live stream hiccup — retry instead of stopping
                time.sleep(0.05)
                continue
            frame_queue.put(None)   # video file ended
            break

        processed = pipeline.process(frame, fps)

        # Resize to max 640px wide before encoding — reduces data further
        h_fr, w_fr = processed.shape[:2]
        if w_fr > 640:
            scale     = 640 / w_fr
            processed = cv2.resize(processed,
                                   (640, int(h_fr * scale)),
                                   interpolation=cv2.INTER_LINEAR)

        # Encode as JPEG quality 40 — ~20-40KB per frame vs ~1.5MB raw
        _, jpg_buf = cv2.imencode('.jpg', processed,
                                  [cv2.IMWRITE_JPEG_QUALITY, 40])
        jpg_bytes = jpg_buf.tobytes()

        item = (jpg_bytes, pipeline.counts.copy(),
                pipeline.speeds.copy(),
                pipeline.violations.copy(),
                pipeline.plates.copy(),
                pipeline.density.streamlit_text())

        # Blocking put — EVERY frame goes to display, no skipping.
        # Retries every 200 ms so stop_event can still exit cleanly.
        while not stop_event.is_set():
            try:
                frame_queue.put(item, timeout=0.2)
                break
            except queue.Full:
                continue

        # Sleep the remaining time so we never exceed target_fps.
        # On fast hardware this prevents racing ahead; on slow hardware
        # (YOLO > frame_interval) elapsed >= frame_interval so sleep is 0.
        elapsed  = time.time() - _t_start
        sleep_t  = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

    cap.release()


# ── Helper: run video capture loop ───────────────────────────────────────────

def _run_video(video_path, zone_polygon,
               conf_threshold, line_position, direction,
               enable_counting, enable_speed, enable_lane, enable_anpr,
               video_placeholder, density_placeholder, count_placeholder,
               speed_placeholder, violation_placeholder, plate_placeholder,
               status_placeholder, stop_button):
    """
    Opens video/stream, runs the pipeline, and streams results to placeholders.
    Extracted to a function so it can be called from both the legacy code path
    (no zone) and the new state-machine path (with zone).
    """
    pipeline = TrafficPipeline(
        conf         = conf_threshold,
        line_pos     = line_position,
        direction    = direction,
        counting     = enable_counting,
        speed        = enable_speed,
        lane         = enable_lane,
        anpr         = enable_anpr,
        zone_polygon = zone_polygon,
    )

    is_ip_cam = isinstance(video_path, str) and video_path.startswith(("http", "rtsp"))

    if is_ip_cam:
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        status_placeholder.error(
            "❌ Could not open video. Check the file or URL."
        )
        return

    ok, test_frame = cap.read()
    if not ok or test_frame is None:
        status_placeholder.error(
            "❌ Connected but no frames received. "
            "Try the RTSP URL: rtsp://YOUR_IP:8080/h264_ulaw.sdp"
        )
        cap.release()
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frame_queue = queue.Queue(maxsize=2)  # small buffer — blocks producer, keeps frames in order
    stop_event  = threading.Event()

    t = threading.Thread(
        target=processing_thread,
        args=(pipeline, cap, fps, frame_queue, stop_event, is_ip_cam),
        daemon=True
    )
    t.start()

    status_placeholder.info("🟢 Running... Press Stop to end.")

    fps_times     = []
    fps_display   = 0.0
    display_count = 0

    while True:
        try:
            result = frame_queue.get(timeout=0.5)
        except queue.Empty:
            # Processing thread is still running but hasn't produced a display
            # frame yet (it throttles to ≤10 fps).  Only exit if dead or stopped.
            if stop_button or not t.is_alive():
                break
            continue   # keep waiting

        if result is None:
            pipeline.stop_ocr_worker()
            status_placeholder.success("✅ Video finished.")
            st.session_state.app_state = "idle"
            break

        jpg_bytes, counts, speeds, violations, plates, density_text = result

        now = time.time()
        fps_times.append(now)
        fps_times = fps_times[-30:]
        if len(fps_times) >= 2:
            fps_display = (len(fps_times) - 1) / (fps_times[-1] - fps_times[0])

        display_count += 1
        # Display every frame using base64 HTML — browser renders JPEG natively
        b64 = base64.b64encode(jpg_bytes).decode()
        video_placeholder.markdown(
            f'<img src="data:image/jpeg;base64,{b64}" style="width:100%">',
            unsafe_allow_html=True
        )

        if display_count % 5 == 0:
            status_placeholder.info(f"🟢 Running — {fps_display:.1f} fps | Press Stop to end.")
            density_placeholder.markdown(density_text)

            if enable_counting and counts:
                total      = sum(counts.values())
                count_text = f"**Total: {total}**\n\n"
                for label, n in sorted(counts.items()):
                    count_text += f"- {label}: {n}\n"
                count_placeholder.markdown(count_text)

            if enable_speed and speeds:
                values     = list(speeds.values())
                speed_text = (
                    f"**Avg: {np.mean(values):.1f}**  |  "
                    f"Max: {max(values):.1f}  |  "
                    f"Min: {min(values):.1f}\n\n"
                )
                for tid, spd in sorted(speeds.items()):
                    speed_text += f"- #{tid}: {spd:.1f} km/h\n"
                speed_placeholder.markdown(speed_text)

            if enable_lane and violations:
                v_text = ""
                for tid, v in sorted(violations.items()):
                    v_text += f"- #{tid}: {v}\n"
                violation_placeholder.markdown(v_text)

            if enable_anpr and plates:
                p_text = ""
                for tid, plate in sorted(plates.items()):
                    if plate:
                        p_text += f"- #{tid}: {plate}\n"
                plate_placeholder.markdown(p_text)

        if stop_button:
            stop_event.set()
            pipeline.stop_ocr_worker()
            status_placeholder.warning("⏹ Stopped.")
            st.session_state.app_state = "idle"
            break

    stop_event.set()
    pipeline.stop_ocr_worker()


# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE — idle → zone_draw → processing
# ─────────────────────────────────────────────────────────────────────────────

# ── IDLE ─────────────────────────────────────────────────────────────────────
if st.session_state.app_state == "idle":

    if source_type == "Image":
        # Image mode — no zone feature, process immediately on Start
        if run_button and image_input is not None:

            model = load_model()
            try:
                if (hasattr(model, 'predictor') and model.predictor is not None and
                        hasattr(model.predictor, 'trackers')):
                    model.predictor.trackers = []
            except Exception:
                pass

            frame = image_input.copy()
            h, w  = frame.shape[:2]

            results = model.predict(
                frame,
                conf=conf_threshold,
                classes=VEHICLE_CLASSES,
                verbose=False
            )

            if enable_lane:
                _ld    = LaneDetector()
                _lines = _ld.detect_lines(frame, h, w)
                _ld.draw(frame, _lines)

            CLASS_COLORS = {2:(0,255,0), 3:(0,165,255), 5:(255,0,0), 7:(0,0,255)}
            detected     = {}
            plates_found = {}

            if results[0].boxes is not None and len(results[0].boxes):
                boxes_data = results[0].boxes
                ocr_reader = None
                if enable_anpr:
                    import easyocr, re as _re
                    ocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)

                for i in range(len(boxes_data)):
                    cls_id   = int(boxes_data.cls[i].item())
                    conf_val = float(boxes_data.conf[i].item())
                    box      = boxes_data.xyxy[i].tolist()
                    label    = model.names[cls_id]
                    x1, y1, x2, y2 = map(int, box)
                    color = CLASS_COLORS.get(cls_id, (200, 200, 200))
                    detected[label] = detected.get(label, 0) + 1
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    lbl_text = f"{label} {conf_val:.2f}"
                    (tw, th), _ = cv2.getTextSize(lbl_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                    cv2.rectangle(frame, (x1, max(0, y1-th-8)), (x1+tw+4, y1), color, -1)
                    cv2.putText(frame, lbl_text, (x1+2, max(th, y1-4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    if enable_anpr and ocr_reader:
                        roi = frame[y1:y2, x1:x2]
                        if roi.size > 0:
                            try:
                                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                                ocr_res  = ocr_reader.readtext(gray_roi, detail=1, paragraph=False)
                                texts = [_re.sub(r"[^A-Z0-9\-]", "", t.upper())
                                         for _, t, c in ocr_res if c > 0.25]
                                plate = " ".join([t for t in texts if len(t) >= 3])
                                if plate:
                                    plates_found[i] = plate
                                    cv2.putText(frame, f"Plate: {plate}",
                                                (x1, y2+20), cv2.FONT_HERSHEY_SIMPLEX,
                                                0.55, (0, 255, 255), 2)
                            except Exception:
                                pass

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(rgb, channels="RGB", use_column_width=True)

            if detected:
                total      = sum(detected.values())
                count_text = f"**Total: {total}**\n\n"
                for lbl, n in sorted(detected.items()):
                    count_text += f"- {lbl}: {n}\n"
                count_placeholder.markdown(count_text)
            else:
                count_placeholder.markdown("No vehicles detected.")

            speed_placeholder.markdown("*(not available for images)*")
            violation_placeholder.markdown("*(not available for images)*")

            if plates_found:
                p_text = ""
                for idx, plate in plates_found.items():
                    p_text += f"- Vehicle {idx+1}: {plate}\n"
                plate_placeholder.markdown(p_text)
            elif enable_anpr:
                plate_placeholder.markdown("No plates detected.")

            status_placeholder.success(
                f"✅ Done — {sum(detected.values())} vehicle(s) detected."
            )

        elif run_button and image_input is None:
            st.sidebar.error("Please upload an image first.")
        else:
            video_placeholder.info("Upload an image in the sidebar and press ▶ Start.")

    else:
        # Video / webcam mode
        if run_button:
            if video_path is None:
                st.sidebar.error("Please upload a video or enter a webcam URL first.")
            else:
                # Read first frame for zone selection
                _cap_tmp = cv2.VideoCapture(video_path)
                _ok, _first = _cap_tmp.read()
                _cap_tmp.release()
                if _ok and _first is not None:
                    st.session_state.first_frame      = _first
                    st.session_state.saved_video_path = video_path
                    st.session_state.app_state        = "zone_draw"
                    st.rerun()
                else:
                    status_placeholder.error("❌ Could not read video file.")
        else:
            video_placeholder.info(
                "Configure settings in the sidebar and press ▶ Start."
            )

# ── ZONE DRAW ────────────────────────────────────────────────────────────────
elif st.session_state.app_state == "zone_draw":

    try:
        from streamlit_drawable_canvas import st_canvas
        _canvas_ok = True
    except ImportError:
        _canvas_ok = False

    first_frame = st.session_state.first_frame
    fh, fw = first_frame.shape[:2]

    # Show instructions
    status_placeholder.info(
        "📍 **Mark your detection zone** — click exactly 4 corner points on the frame below. "
        "The points will be connected into a zone. Then click **Confirm Zone**."
    )

    col_canvas, col_ctrl = st.columns([3, 1])

    with col_canvas:
        if _canvas_ok:
            # Canvas display size (fit to ~700 px wide)
            canvas_w = min(fw, 700)
            canvas_h = int(fh * canvas_w / fw)

            import PIL.Image as PILImage
            bg_img = PILImage.fromarray(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
            bg_img = bg_img.resize((canvas_w, canvas_h))

            canvas_result = st_canvas(
                fill_color       = "rgba(0, 230, 0, 0.5)",
                stroke_width     = 2,
                stroke_color     = "#00e600",
                background_image = bg_img,
                update_streamlit = True,
                height           = canvas_h,
                width            = canvas_w,
                drawing_mode     = "point",
                point_display_radius = 6,
                key              = "zone_canvas",
            )

            # Show live point count so user knows when to stop clicking
            n_pts = 0
            if canvas_result is not None and canvas_result.json_data:
                n_pts = len([o for o in canvas_result.json_data.get("objects", [])
                             if o.get("type") == "circle"])
            if n_pts == 0:
                st.info("Click 4 corner points on the frame above.")
            elif n_pts < 4:
                st.warning(f"{n_pts}/4 points placed — click {4 - n_pts} more.")
            elif n_pts == 4:
                st.success("✅ 4 points placed — click **Confirm Zone**.")
            else:
                st.error(f"{n_pts} points placed — only 4 needed. Please cancel and redraw.")

            # ── Save canvas JSON to session state immediately ──────────────────
            if canvas_result is not None and canvas_result.json_data:
                circles = [o for o in canvas_result.json_data.get("objects", [])
                           if o.get("type") == "circle"]
                if circles:
                    st.session_state["_canvas_json"] = canvas_result.json_data
                    st.session_state["_canvas_w"]    = canvas_w
                    st.session_state["_canvas_h"]    = canvas_h

            # ── DEBUG: show raw canvas output so we can fix parsing ────────────
            with st.expander("🔍 Debug — Canvas JSON (remove after fixing)", expanded=False):
                if canvas_result is not None:
                    st.write("json_data:", canvas_result.json_data)
                st.write("Saved in session_state:", st.session_state.get("_canvas_json"))

        else:
            st.error(
                "⚠️ `streamlit-drawable-canvas` is not installed. "
                "Add it to your Dockerfile and rebuild, or click **Skip Zone**."
            )
            canvas_result = None
            canvas_w = fw
            canvas_h = fh

    with col_ctrl:
        st.markdown("### Zone Controls")
        confirm_btn = st.button("✅ Confirm Zone", use_container_width=True)
        skip_btn    = st.button("⏩ Skip Zone",    use_container_width=True)
        cancel_btn  = st.button("✖ Cancel",        use_container_width=True)

        st.markdown(
            "**How to mark zone:**\n"
            "1. Click the 4 corners of your zone\n"
            "2. Points are connected automatically\n"
            "3. Click **Confirm Zone** when done"
        )

    if confirm_btn:
        zone_poly = None
        saved_json = st.session_state.get("_canvas_json")
        saved_cw   = st.session_state.get("_canvas_w", canvas_w)
        saved_ch   = st.session_state.get("_canvas_h", canvas_h)

        if _canvas_ok and saved_json:
            zone_filter = ZoneFilter.from_canvas(
                saved_json, saved_cw, saved_ch, fw, fh
            )
            if zone_filter.active:
                zone_poly = zone_filter.polygon

        st.session_state.pop("_canvas_json", None)
        st.session_state.zone_polygon = zone_poly
        # Go to preview first so user can verify the zone before processing
        st.session_state.app_state = "zone_preview"
        st.rerun()

    if skip_btn:
        st.session_state.zone_polygon    = None
        st.session_state.app_state       = "processing"
        st.session_state._proc_can_start = True   # one-time token
        st.rerun()

    if cancel_btn:
        st.session_state.app_state    = "idle"
        st.session_state.zone_polygon = None
        st.rerun()

# ── ZONE PREVIEW ─────────────────────────────────────────────────────────────
elif st.session_state.app_state == "zone_preview":

    first_frame  = st.session_state.first_frame
    zone_polygon = st.session_state.zone_polygon

    if zone_polygon and len(zone_polygon) >= 3:
        # Draw the parsed zone on the first frame so user can verify it
        preview = first_frame.copy()
        ZoneFilter(zone_polygon).draw(preview)
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        video_placeholder.image(preview_rgb, channels="RGB", use_column_width=True)
        status_placeholder.success(
            f"✅ Zone parsed — {len(zone_polygon)} points. "
            "Does the green polygon match what you drew?"
        )
    else:
        # Polygon failed to parse — show original frame and warn
        preview_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(preview_rgb, channels="RGB", use_column_width=True)
        status_placeholder.warning(
            "⚠️ Zone could not be parsed — will process the full frame. "
            "Try redrawing: click more points and double-click to close."
        )

    col_ok, col_redo = st.columns(2)
    with col_ok:
        if st.button("▶ Start Processing", use_container_width=True):
            st.session_state.app_state      = "processing"
            st.session_state._proc_can_start = True   # one-time token
            st.rerun()
    with col_redo:
        if st.button("↩ Redraw Zone", use_container_width=True):
            st.session_state.zone_polygon = None
            st.session_state.app_state    = "zone_draw"
            st.rerun()

# ── PROCESSING ───────────────────────────────────────────────────────────────
elif st.session_state.app_state == "processing":

    # _proc_can_start is a one-time token set only when the user explicitly
    # clicks Start.  We consume it immediately (set to False) before calling
    # _run_video(), so any Streamlit rerun that arrives mid-processing sees
    # False and refuses to restart the video.
    if st.session_state.get("_proc_can_start", False):
        st.session_state._proc_can_start = False   # consume — no re-entry
        try:
            _run_video(
                video_path         = st.session_state.saved_video_path,
                zone_polygon       = st.session_state.zone_polygon,
                conf_threshold     = conf_threshold,
                line_position      = line_position,
                direction          = direction,
                enable_counting    = enable_counting,
                enable_speed       = enable_speed,
                enable_lane        = enable_lane,
                enable_anpr        = enable_anpr,
                video_placeholder  = video_placeholder,
                density_placeholder= density_placeholder,
                count_placeholder  = count_placeholder,
                speed_placeholder  = speed_placeholder,
                violation_placeholder = violation_placeholder,
                plate_placeholder  = plate_placeholder,
                status_placeholder = status_placeholder,
                stop_button        = stop_button,
            )
        finally:
            st.session_state.app_state = "idle"
    else:
        # Token already consumed — a stray rerun landed here.  Just reset.
        st.session_state.app_state = "idle"