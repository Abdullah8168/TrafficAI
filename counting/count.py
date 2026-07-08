"""
TrafficAI - Vehicle Counting Module (Module 3)

Draws a virtual line across the frame.
Every tracked vehicle that crosses it is counted once per ID.
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


class VehicleCounter:

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=CONFIDENCE_THRESHOLD,
        line_position=0.5       # fraction of frame height (0.0 – 1.0)
    ):
        """
        Args:
            line_position: where to draw the counting line as a fraction
                           of the frame height. 0.5 = middle of frame.
        """

        print("[INFO] Loading YOLO Model for Counting...")

        self.model          = YOLO(model_path)
        self.conf           = conf
        self.vehicle_classes = VEHICLE_CLASSES
        self.line_position  = line_position

        print("[INFO] Model Loaded Successfully!")

    ##########################################################
    # VIDEO
    ##########################################################

    def count_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        # Counting line Y coordinate (pixels)
        line_y = int(height * self.line_position)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "counted.mp4")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        counts   = {}   # class_label → int
        crossed  = {}   # track_id    → class_label  (crossed IDs)
        prev_pos = {}   # track_id    → last centroid Y

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame, counts, crossed, prev_pos = self._count_frame(
                frame, line_y, counts, crossed, prev_pos
            )

            writer.write(frame)

        cap.release()
        writer.release()

        self._print_summary(counts)

        print(f"[INFO] Counted video saved to {output_path}")

        return output_path

    ##########################################################
    # WEBCAM
    ##########################################################

    def count_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open webcam.")

        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        line_y = int(height * self.line_position)

        counts   = {}
        crossed  = {}
        prev_pos = {}

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame, counts, crossed, prev_pos = self._count_frame(
                frame, line_y, counts, crossed, prev_pos
            )

            try:
                cv2.imshow("TrafficAI - Counter (Q to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except:
                pass

        cap.release()
        cv2.destroyAllWindows()

        self._print_summary(counts)

    ##########################################################
    # CORE — process one frame
    ##########################################################

    def _count_frame(self, frame, line_y, counts, crossed, prev_pos):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=self.vehicle_classes,
            verbose=False
        )

        # Draw the counting line
        h, w = frame.shape[:2]
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(
            frame, "COUNTING LINE",
            (10, line_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            self._draw_counts(frame, counts)
            return frame, counts, crossed, prev_pos

        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):

            cls_id = int(boxes_data.cls[i].item())

            if cls_id not in self.vehicle_classes:
                continue

            track_id = int(boxes_data.id[i].item())
            conf_val = float(boxes_data.conf[i].item())
            box      = boxes_data.xyxy[i].tolist()
            label    = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, box)
            centroid_y = (y1 + y2) // 2
            centroid_x = (x1 + x2) // 2

            # ── Crossing detection ────────────────────────────────────────────
            if track_id in prev_pos and track_id not in crossed:
                prev_y = prev_pos[track_id]

                # Vehicle moved from above line to below (or vice versa)
                crossed_line = (
                    (prev_y < line_y <= centroid_y) or
                    (prev_y > line_y >= centroid_y)
                )

                if crossed_line:
                    crossed[track_id] = label
                    counts[label] = counts.get(label, 0) + 1

                    # Flash green dot at crossing point
                    cv2.circle(frame, (centroid_x, line_y), 8, (0, 255, 0), -1)

            prev_pos[track_id] = centroid_y

            # ── Draw box ──────────────────────────────────────────────────────
            already_counted = track_id in crossed
            frame = self._draw_box(
                frame, box, track_id, cls_id,
                label, conf_val, already_counted
            )

        self._draw_counts(frame, counts)

        return frame, counts, crossed, prev_pos

    ##########################################################
    # DRAW BOX
    ##########################################################

    def _draw_box(self, image, box, track_id, cls_id,
                  label, conf_val, counted):

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        # Dashed/solid box: solid if counted, thinner if not
        thickness = 3 if counted else 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # Label
        text = f"{label} #{track_id}"
        if counted:
            text += " ✓"

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

    ##########################################################
    # DRAW COUNTS PANEL
    ##########################################################

    def _draw_counts(self, frame, counts):
        """Draws a live count panel in the top-right corner."""

        if not counts:
            return

        h, w = frame.shape[:2]
        panel_x = w - 200
        panel_y = 10
        line_h  = 30
        total   = sum(counts.values())

        # Background
        cv2.rectangle(
            frame,
            (panel_x - 10, panel_y),
            (w - 5, panel_y + line_h * (len(counts) + 1) + 5),
            (0, 0, 0), -1
        )

        # Total
        cv2.putText(
            frame, f"Total: {total}",
            (panel_x, panel_y + line_h),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 255, 255), 2
        )

        # Per class
        for idx, (label, count) in enumerate(sorted(counts.items()), start=2):
            color = CLASS_COLORS.get(
                next((k for k, v in {
                    2: "car", 3: "motorcycle",
                    5: "bus", 7: "truck"
                }.items() if v == label.lower()), None),
                (200, 200, 200)
            )
            cv2.putText(
                frame, f"{label}: {count}",
                (panel_x, panel_y + line_h * idx),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                color, 2
            )

    ##########################################################
    # SUMMARY
    ##########################################################

    def _print_summary(self, counts):

        print("\n── Counting Summary ──────────────────────────────")
        total = sum(counts.values())
        print(f"  Total vehicles counted: {total}")

        for label, count in sorted(counts.items()):
            print(f"    {label:<14}: {count}")

        print("──────────────────────────────────────────────────\n")