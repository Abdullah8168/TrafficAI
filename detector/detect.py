"""
TrafficAI - Vehicle Detection Module
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


class VehicleDetector:

    def __init__(
        self,
        model_path=MODEL_PATH,
        conf=CONFIDENCE_THRESHOLD
    ):

        print("[INFO] Loading YOLO Model...")

        self.model = YOLO(model_path)
        self.conf = conf
        self.vehicle_classes = VEHICLE_CLASSES

        print("[INFO] Model Loaded Successfully!")

    ##########################################################
    # IMAGE
    ##########################################################

    def detect_image(self, image_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Cannot open image: {image_path}"
            )

        results = self.model.predict(
            source=image,
            conf=self.conf,
            verbose=False
        )

        annotated = self.draw_results(image, results)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_path = os.path.join(
            OUTPUT_DIR,
            "result.jpg"
        )

        cv2.imwrite(output_path, annotated)

        print(f"[INFO] Result saved to {output_path}")

        return output_path

    ##########################################################
    # VIDEO
    ##########################################################

    def detect_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise Exception(
                f"Cannot open video: {video_path}"
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        output_path = os.path.join(
            OUTPUT_DIR,
            "result.mp4"
        )

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height)
        )

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                verbose=False
            )

            annotated = self.draw_results(frame, results)

            writer.write(annotated)

        cap.release()
        writer.release()

        print(f"[INFO] Video saved to {output_path}")

        return output_path

    ##########################################################
    # WEBCAM
    ##########################################################

    def detect_webcam(self):

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise Exception("Cannot open webcam.")

        while True:

            success, frame = cap.read()

            if not success:
                break

            results = self.model.predict(
                source=frame,
                conf=self.conf,
                verbose=False
            )

            annotated = self.draw_results(frame, results)

            try:
                cv2.imshow("TrafficAI", annotated)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            except:
                pass

        cap.release()
        cv2.destroyAllWindows()

    ##########################################################
    # DRAW BOXES
    ##########################################################

    def draw_results(self, image, results):

        output = image.copy()

        for result in results:

            for box in result.boxes:

                cls = int(box.cls[0])

                if cls not in self.vehicle_classes:
                    continue

                confidence = float(box.conf[0])

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                label = (
                    f"{self.model.names[cls]} "
                    f"{confidence:.2f}"
                )

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    output,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        return output