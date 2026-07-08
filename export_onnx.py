"""
TrafficAI - ONNX Export (Module 9 - Part 1)

Converts YOLOv8 .pt model to ONNX format.
ONNX models are faster, portable, and can run without PyTorch.

Usage:
    python export_onnx.py
    python export_onnx.py --model yolov8n.pt --imgsz 640
"""

import argparse
from ultralytics import YOLO
import os


def export_model(model_path="yolov8n.pt", imgsz=640):

    print(f"[INFO] Loading model: {model_path}")

    model = YOLO(model_path)

    print(f"[INFO] Exporting to ONNX (imgsz={imgsz})...")

    export_path = model.export(
        format  = "onnx",
        imgsz   = imgsz,
        dynamic = False,    # fixed input shape — faster inference
        simplify= True,     # simplify ONNX graph
    )

    print(f"\n[INFO] Export complete!")
    print(f"[INFO] ONNX model saved at: {export_path}")
    print(f"[INFO] File size: {os.path.getsize(export_path) / 1e6:.1f} MB")

    return export_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Export YOLOv8 to ONNX")

    parser.add_argument("--model",  default="yolov8n.pt", help="Model path")
    parser.add_argument("--imgsz",  type=int, default=640, help="Image size")

    args = parser.parse_args()

    export_model(args.model, args.imgsz)