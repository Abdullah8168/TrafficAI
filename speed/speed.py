"""
TrafficAI - Speed Estimation Module (Module 4)

No manual calibration required.

How it works:
    - A standard car is ~4.5m long and ~1.8m wide.
    - YOLO gives us the bounding box size in pixels.
    - We compute pixels-per-metre from detected car boxes.
    - Speed = pixel displacement / pixels-per-metre / time  →  km/h

Usage:
    python main.py --video data/videos/video1.mp4 --speed
    python main.py --webcam --speed
"""

import os
import cv2
import numpy as np
from ultralytics import YOLO

from configs.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    OUTPUT_DIR
)


# ── Colour per COCO class (BGR) ───────────────────────────────────────────────
CLASS_COLORS = {
    2: (0, 255, 0),       # Car        → Green
    3: (0, 165, 255),     # Motorcycle → Orange
    5: (255, 0, 0),       # Bus        → Blue
    7: (0, 0, 255),       # Truck      → Red
}

# Real-world vehicle dimensions (metres) used for auto scale
VEHICLE_REAL_WIDTH = {
    2: 1.8,    # Car
    3: 0.8,    # Motorcycle
    5: 2.5,    # Bus
    7: 2.5,    # Truck
}

VEHICLE_REAL_LENGTH = {
    2: 4.5,    # Car
    3: 2.0,    # Motorcycle
    5: 12.0,   # Bus
    7: 8.0,    # Truck
}


class SpeedEstimator:

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=CONFIDENCE_THRESHOLD
    ):

        print("[INFO] Loading YOLO Model for Speed Estimation...")

        self.model           = YOLO(model_path)
        self.conf            = conf
        self.vehicle_classes = VEHICLE_CLASSES

        # Accumulated pixel-per-metre estimates from detected boxes
        self._scale_samples  = []   # list of px/m values
        self._pixels_per_m   = None

        print("[INFO] Model Loaded Successfully!")
        print("[INFO] Auto-calibration active — no manual points needed.\n")

    ##########################################################
    # VIDEO
    ##########################################################

    def estimate_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "speed.mp4")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        history  = {}   # track_id → [(cx_px, cy_px, timestamp), ...]
        speeds   = {}   # track_id → smoothed speed km/h
        frame_num = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_num / fps
            frame_num += 1

            frame, speeds, history = self._process_frame(
                frame, timestamp, history, speeds
            )

            writer.write(frame)

        cap.release()
        writer.release()

        self._print_summary(speeds)

        print(f"[INFO] Speed video saved to {output_path}")

        return output_path

    ##########################################################
    # WEBCAM
    ##########################################################

    def estimate_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open webcam.")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        history   = {}
        speeds    = {}
        frame_num = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            timestamp = frame_num / fps
            frame_num += 1

            frame, speeds, history = self._process_frame(
                frame, timestamp, history, speeds
            )

            try:
                cv2.imshow("TrafficAI - Speed (Q to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception:
                pass

        cap.release()
        cv2.destroyAllWindows()

        self._print_summary(speeds)

    ##########################################################
    # CORE — process one frame
    ##########################################################

    def _process_frame(self, frame, timestamp, history, speeds):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=self.vehicle_classes,
            verbose=False
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            self._draw_scale_status(frame)
            return frame, speeds, history

        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):

            cls_id   = int(boxes_data.cls[i].item())

            if cls_id not in self.vehicle_classes:
                continue

            track_id = int(boxes_data.id[i].item())
            conf_val = float(boxes_data.conf[i].item())
            box      = boxes_data.xyxy[i].tolist()
            label    = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, box)
            box_w_px = x2 - x1
            box_h_px = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # ── Auto-calibrate scale from this box ────────────────────────────
            self._update_scale(cls_id, box_w_px, box_h_px)

            # ── Record centroid history ───────────────────────────────────────
            if track_id not in history:
                history[track_id] = []

            history[track_id].append((cx, cy, timestamp))
            history[track_id] = history[track_id][-15:]  # keep last 15 frames

            # ── Calculate speed ───────────────────────────────────────────────
            speed_kmh = self._calc_speed(history[track_id])

            if speed_kmh is not None:
                prev = speeds.get(track_id, speed_kmh)
                speeds[track_id] = 0.7 * prev + 0.3 * speed_kmh

            speed_str = f"{speeds[track_id]:.1f} km/h" \
                if track_id in speeds else "calibrating..."

            frame = self._draw_box(
                frame, box, track_id, cls_id, label, conf_val, speed_str
            )

        self._draw_scale_status(frame)

        return frame, speeds, history

    ##########################################################
    # AUTO SCALE ESTIMATION
    ##########################################################

    def _update_scale(self, cls_id, box_w_px, box_h_px):
        """
        Estimate pixels-per-metre from a detected bounding box.
        Uses vehicle width (more stable than height for traffic cameras).
        """

        real_w = VEHICLE_REAL_WIDTH.get(cls_id)

        if real_w is None or box_w_px < 20:
            return

        px_per_m = box_w_px / real_w
        self._scale_samples.append(px_per_m)

        # Keep last 50 samples, use median (robust to outliers)
        self._scale_samples = self._scale_samples[-50:]
        self._pixels_per_m  = float(np.median(self._scale_samples))

    ##########################################################
    # SPEED MATH
    ##########################################################

    def _calc_speed(self, positions):
        """
        Calculate speed (km/h) from centroid history in pixels.
        Converts pixel displacement to metres using auto-estimated scale.
        """

        if self._pixels_per_m is None or len(positions) < 2:
            return None

        # Use first and last position for displacement
        cx1, cy1, t1 = positions[0]
        cx2, cy2, t2 = positions[-1]

        dt = t2 - t1

        if dt < 0.2:
            return None

        # Pixel displacement
        dist_px = np.sqrt((cx2 - cx1) ** 2 + (cy2 - cy1) ** 2)

        # Convert to metres
        dist_m = dist_px / self._pixels_per_m

        speed_ms  = dist_m / dt
        speed_kmh = speed_ms * 3.6

        # Sanity cap
        if speed_kmh > 200 or speed_kmh < 1:
            return None

        return speed_kmh

    ##########################################################
    # DRAW HELPERS
    ##########################################################

    def _draw_box(self, image, box, track_id, cls_id,
                  label, conf_val, speed_str):

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        text = f"{label} #{track_id}  {speed_str}"

        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        cv2.rectangle(
            image,
            (x1, max(0, y1 - th - 8)),
            (x1 + tw + 4, y1),
            color, -1
        )

        cv2.putText(
            image, text,
            (x1 + 2, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2
        )

        return image

    def _draw_scale_status(self, frame):
        """Show auto-calibration status in top-left corner."""

        if self._pixels_per_m is None:
            msg   = "Auto-calibrating..."
            color = (0, 165, 255)
        else:
            msg   = f"Scale: {self._pixels_per_m:.1f} px/m"
            color = (0, 255, 0)

        cv2.putText(
            frame, msg,
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            color, 2
        )

    ##########################################################
    # SUMMARY
    ##########################################################

    def _print_summary(self, speeds):

        print("\n── Speed Summary ─────────────────────────────────")

        if not speeds:
            print("  No speed data recorded.")
        else:
            # Per-vehicle speeds
            print(f"  {'ID':<8} {'Speed (km/h)'}")
            print(f"  {'──':<8} {'────────────'}")

            for track_id, speed in sorted(speeds.items()):
                print(f"  #{track_id:<7} {speed:.1f} km/h")

            # Overall stats
            values = list(speeds.values())
            print(f"\n  Vehicles measured : {len(values)}")
            print(f"  Average speed     : {np.mean(values):.1f} km/h")
            print(f"  Max speed         : {np.max(values):.1f} km/h")
            print(f"  Min speed         : {np.min(values):.1f} km/h")

        print("──────────────────────────────────────────────────\n")