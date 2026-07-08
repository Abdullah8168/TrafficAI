# ─────────────────────────────────────────────────────────────────────────────
# TrafficAI - Dockerfile (CPU-only PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

WORKDIR /app

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch first (much smaller than full PyTorch)
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir \
    ultralytics \
    opencv-python \
    fastapi \
    uvicorn \
    python-multipart \
    "streamlit==1.28.2" \
    plotly \
    easyocr \
    onnx \
    onnxruntime \
    pandas \
    Pillow \
    "streamlit-drawable-canvas==0.9.3"

# Copy project files
COPY . .

# Download YOLOv8 weights and export to ONNX for faster CPU inference
RUN python -c "\
from ultralytics import YOLO; \
model = YOLO('yolov8n.pt'); \
model.export(format='onnx', imgsz=320, dynamic=False, simplify=True); \
print('ONNX export complete: yolov8n.onnx')"

# Expose ports
EXPOSE 8000 8501

# Default command (overridden per-service in docker-compose.yml)
CMD ["uvicorn", "api.api:app", "--host", "0.0.0.0", "--port", "8000"]