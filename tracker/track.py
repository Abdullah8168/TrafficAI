"""
TrafficAI - Vehicle Tracking Module (Module 2)

Wraps ByteTrack (built into Ultralytics) to assign
persistent IDs to every detected vehicle across frames.
"""

import os
import cv2
from ultralytics import YOLO

from configs.config import (
    MODEL_PATH,
    CONFIDENCE_THRESHOLD,
    VEHICLE_CLASSES,
    OUTPUT_DIR
)


# ── Colour per COCO class (BGR) ───────────────────────────────────────────────
CLASS_COLORS = {
    2: (0, 255, 0),      # Car        → Green
    3: (0, 165, 255),    # Motorcycle → Orange
    5: (255, 0, 0),      # Bus        → Blue
    7: (0, 0, 255),      # Truck      → Red
}


class VehicleTracker:

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=CONFIDENCE_THRESHOLD
    ):

        print("[INFO] Loading YOLO Model for Tracking...")

        self.model = YOLO(model_path)
        self.conf = conf
        self.vehicle_classes = VEHICLE_CLASSES

        print("[INFO] Model Loaded Successfully!")

    ##########################################################
    # VIDEO
    ##########################################################

    def track_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(
                f"Cannot open video: {video_path}"
            )

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_path = os.path.join(OUTPUT_DIR, "tracked.mp4")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        seen_ids = {}

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame, seen_ids = self._track_frame(frame, seen_ids)

            writer.write(frame)

        cap.release()
        writer.release()

        self._print_summary(seen_ids)

        print(f"[INFO] Tracked video saved to {output_path}")

        return output_path

    ##########################################################
    # WEBCAM
    ##########################################################

    def track_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open webcam.")

        seen_ids = {}

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame, seen_ids = self._track_frame(frame, seen_ids)

            try:
                cv2.imshow("TrafficAI - Tracker (Q to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except:
                pass

        cap.release()
        cv2.destroyAllWindows()

        self._print_summary(seen_ids)

    ##########################################################
    # CORE — track a single frame
    ##########################################################

    def _track_frame(self, frame, seen_ids):
        """
        Run ByteTrack on one frame.
        Returns the annotated frame and updated seen_ids dict.
        """

        results = self.model.track(
            frame,
            persist=True,                # keeps Kalman state between frames
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=self.vehicle_classes,
            verbose=False
        )

        if results[0].boxes is None:
            return frame, seen_ids

        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):

            cls_id = int(boxes_data.cls[i].item())

            if cls_id not in self.vehicle_classes:
                continue

            # track_id can be None on first frame
            if boxes_data.id is None:
                continue

            track_id = int(boxes_data.id[i].item())
            conf_val = float(boxes_data.conf[i].item())
            box      = boxes_data.xyxy[i].tolist()

            label = self.model.names[cls_id]

            seen_ids[track_id] = label

            frame = self._draw_box(
                frame, box, track_id, cls_id, label, conf_val
            )

        # Active count overlay
        active = len(boxes_data.id) if boxes_data.id is not None else 0

        cv2.putText(
            frame,
            f"Active: {active}  |  Total seen: {len(seen_ids)}",
            (12, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        return frame, seen_ids

    ##########################################################
    # DRAW BOX
    ##########################################################

    def _draw_box(self, image, box, track_id, cls_id, label, conf_val):

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # Bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Label: "Car #12  0.87"
        text = f"{label} #{track_id}  {conf_val:.2f}"

        (tw, th), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Filled background for readability
        cv2.rectangle(
            image,
            (x1, max(0, y1 - th - 8)),
            (x1 + tw + 4, y1),
            color,
            -1
        )

        cv2.putText(
            image,
            text,
            (x1 + 2, max(th, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        return image

    ##########################################################
    # SUMMARY
    ##########################################################

    def _print_summary(self, seen_ids):

        print("\n── Tracking Summary ──────────────────────────────")
        print(f"  Unique vehicles tracked: {len(seen_ids)}")

        class_counts = {}
        for label in seen_ids.values():
            class_counts[label] = class_counts.get(label, 0) + 1

        for label, count in sorted(class_counts.items()):
            print(f"    {label:<14}: {count}")

        print("──────────────────────────────────────────────────\n")