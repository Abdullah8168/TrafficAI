"""
TrafficAI - ANPR Module (Module 6)
Automatic Number Plate Recognition

Pipeline:
    Vehicle detected → Crop vehicle ROI → Detect plate → Crop plate
    → Enhance → OCR → Read plate number

Usage:
    python main.py --video data/videos/video1.mp4 --anpr
    python main.py --image data/images/image1.jpg --anpr
    python main.py --webcam --anpr

Install OCR engine (run once):
    pip install easyocr
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
    2: (0, 255, 0),
    3: (0, 165, 255),
    5: (255, 0, 0),
    7: (0, 0, 255),
}


class ANPRDetector:

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=CONFIDENCE_THRESHOLD
    ):

        print("[INFO] Loading YOLO Model for ANPR...")

        self.model           = YOLO(model_path)
        self.conf            = conf
        self.vehicle_classes = VEHICLE_CLASSES

        # track_id → best plate text read so far
        self.plate_results   = {}

        # Lazy-load EasyOCR (heavy import, only load when needed)
        self._reader         = None

        print("[INFO] Model Loaded Successfully!")
        print("[INFO] EasyOCR will load on first detection.\n")

    ##########################################################
    # OCR READER (lazy load)
    ##########################################################

    def _get_reader(self):

        if self._reader is None:
            import easyocr
            print("[INFO] Loading EasyOCR (first time may take a moment)...")
            self._reader = easyocr.Reader(["en"], gpu=False)
            print("[INFO] EasyOCR Ready!")

        return self._reader

    ##########################################################
    # IMAGE
    ##########################################################

    def detect_image(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(f"Cannot open image: {image_path}")

        frame = self._process_frame(image)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "anpr_result.jpg")

        cv2.imwrite(output_path, frame)

        self._print_summary()

        print(f"[INFO] Result saved to {output_path}")

        return output_path

    ##########################################################
    # VIDEO
    ##########################################################

    def detect_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(f"Cannot open video: {video_path}")

        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "anpr.mp4")

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        frame_num = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            # Run OCR every 5 frames (saves processing time)
            run_ocr = (frame_num % 5 == 0)
            frame   = self._process_frame(frame, run_ocr=run_ocr)

            writer.write(frame)
            frame_num += 1

        cap.release()
        writer.release()

        self._print_summary()

        print(f"[INFO] ANPR video saved to {output_path}")

        return output_path

    ##########################################################
    # WEBCAM
    ##########################################################

    def detect_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open webcam.")

        frame_num = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            run_ocr = (frame_num % 5 == 0)
            frame   = self._process_frame(frame, run_ocr=run_ocr)

            frame_num += 1

            try:
                cv2.imshow("TrafficAI - ANPR (Q to quit)", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except Exception:
                pass

        cap.release()
        cv2.destroyAllWindows()

        self._print_summary()

    ##########################################################
    # CORE — process one frame
    ##########################################################

    def _process_frame(self, frame, run_ocr=True):

        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            classes=self.vehicle_classes,
            verbose=False
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            return frame

        boxes_data = results[0].boxes

        for i in range(len(boxes_data)):

            cls_id   = int(boxes_data.cls[i].item())

            if cls_id not in self.vehicle_classes:
                continue

            track_id = int(boxes_data.id[i].item())
            box      = boxes_data.xyxy[i].tolist()
            label    = self.model.names[cls_id]

            x1, y1, x2, y2 = map(int, box)

            # ── Extract vehicle ROI ───────────────────────────────────────────
            vehicle_roi = frame[y1:y2, x1:x2]

            if vehicle_roi.size == 0:
                continue

            # ── Detect plate inside vehicle ROI ───────────────────────────────
            plate_box = self._detect_plate(vehicle_roi)

            plate_text = self.plate_results.get(track_id, "")

            if run_ocr and plate_box is not None:

                px1, py1, px2, py2 = plate_box
                plate_crop = vehicle_roi[py1:py2, px1:px2]

                if plate_crop.size > 0:
                    text = self._read_plate(plate_crop)

                    if text:
                        # Keep the longest clean read for this vehicle
                        existing = self.plate_results.get(track_id, "")
                        if len(text) > len(existing):
                            self.plate_results[track_id] = text
                            plate_text = text

                # Draw plate box on frame (absolute coordinates)
                abs_px1 = x1 + px1
                abs_py1 = y1 + py1
                abs_px2 = x1 + px2
                abs_py2 = y1 + py2

                cv2.rectangle(
                    frame,
                    (abs_px1, abs_py1),
                    (abs_px2, abs_py2),
                    (0, 255, 255), 2
                )

            # ── Draw vehicle box ──────────────────────────────────────────────
            frame = self._draw_box(
                frame, box, track_id, cls_id, label, plate_text
            )

        return frame

    ##########################################################
    # PLATE DETECTION (classical CV)
    ##########################################################

    def _detect_plate(self, vehicle_roi):
        """
        Finds the most likely license plate region inside a vehicle crop.
        Uses edge detection + contour filtering.
        Returns (x1, y1, x2, y2) or None.
        """

        h, w = vehicle_roi.shape[:2]

        gray  = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
        blur  = cv2.bilateralFilter(gray, 11, 17, 17)
        edges = cv2.Canny(blur, 30, 200)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]

        for contour in contours:

            perimeter = cv2.arcLength(contour, True)
            approx    = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

            if len(approx) == 4:

                bx, by, bw, bh = cv2.boundingRect(approx)

                # Plate aspect ratio: typically 2:1 to 5:1
                aspect = bw / bh if bh > 0 else 0

                if 2.0 <= aspect <= 6.0 and bw > 60 and bh > 15:

                    # Plate should be in the lower portion of the vehicle
                    if by > h * 0.3:
                        return (bx, by, bx + bw, by + bh)

        return None

    ##########################################################
    # OCR
    ##########################################################

    def _read_plate(self, plate_crop):
        """
        Enhance the plate crop and run EasyOCR on it.
        Returns cleaned plate text or empty string.
        """

        # ── Enhance ───────────────────────────────────────────────────────────
        enhanced = self._enhance_plate(plate_crop)

        # ── EasyOCR ──────────────────────────────────────────────────────────
        try:
            reader  = self._get_reader()
            results = reader.readtext(enhanced)

            texts = []

            for (_, text, confidence) in results:
                if confidence > 0.3:
                    texts.append(text)

            if not texts:
                return ""

            # Join multiple text regions and clean
            raw = " ".join(texts).upper()
            cleaned = self._clean_plate(raw)

            return cleaned

        except Exception as e:
            return ""

    def _enhance_plate(self, plate):
        """Preprocess plate image for better OCR accuracy."""

        # Resize to standard height
        h, w = plate.shape[:2]
        scale = 60 / h if h > 0 else 1
        resized = cv2.resize(
            plate,
            (int(w * scale), 60),
            interpolation=cv2.INTER_CUBIC
        )

        # Grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # CLAHE — improve contrast
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        # Threshold
        _, thresh = cv2.threshold(
            enhanced, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        return thresh

    def _clean_plate(self, text):
        """Remove unwanted characters from OCR output."""

        import re

        # Keep only alphanumeric and dash/space
        cleaned = re.sub(r"[^A-Z0-9\-\s]", "", text)
        cleaned = cleaned.strip()

        # Must have at least 4 characters to be a plate
        if len(cleaned) < 4:
            return ""

        return cleaned

    ##########################################################
    # DRAW BOX
    ##########################################################

    def _draw_box(self, image, box, track_id, cls_id, label, plate_text):

        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(cls_id, (200, 200, 200))

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Vehicle label + ID
        text = f"{label} #{track_id}"

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

        # Plate text below the box
        if plate_text:
            cv2.putText(
                image, f"Plate: {plate_text}",
                (x1, y2 + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 255, 255), 2
            )

        return image

    ##########################################################
    # SUMMARY
    ##########################################################

    def _print_summary(self):

        print("\n── ANPR Summary ──────────────────────────────────")

        if not self.plate_results:
            print("  No plates detected.")
        else:
            print(f"  {'ID':<8} {'Plate'}")
            print(f"  {'──':<8} {'─────'}")

            for track_id, plate in sorted(self.plate_results.items()):
                if plate:
                    print(f"  #{track_id:<7} {plate}")

        print("──────────────────────────────────────────────────\n")