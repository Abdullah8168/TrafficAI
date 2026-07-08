"""
Configuration file for TrafficAI
"""

import os

# =====================================================
# Project Root
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# =====================================================
# Model
# =====================================================

MODEL_PATH = "yolov8n.pt"

# =====================================================
# Detection Settings
# =====================================================

CONFIDENCE_THRESHOLD = 0.40

# COCO Vehicle Classes
# 2 = Car
# 3 = Motorcycle
# 5 = Bus
# 7 = Truck
VEHICLE_CLASSES = [2, 3, 5, 7]

# =====================================================
# Directories
# =====================================================

IMAGE_DIR = os.path.join(BASE_DIR, "data", "images")

VIDEO_DIR = os.path.join(BASE_DIR, "data", "videos")

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")