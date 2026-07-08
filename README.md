# 🚗 TrafficAI — Smart Traffic Monitoring System

An AI-powered traffic monitoring system that performs real-time vehicle detection, counting, speed estimation, lane violation detection, license plate recognition, and traffic density analysis using computer vision and deep learning.

---

## 📌 Features

| Module | Description |
|--------|-------------|
| 🔍 **Vehicle Detection** | Detects cars, trucks, buses, and motorcycles using YOLOv8 |
| 🔢 **Vehicle Counting** | Counts vehicles crossing a configurable line with deduplication |
| 💨 **Speed Estimation** | Estimates speed in km/h using pixel-to-meter calibration |
| 🛣️ **Lane Detection** | Detects wrong-way driving and out-of-lane violations |
| 📷 **ANPR** | Automatic Number Plate Recognition using EasyOCR |
| 📊 **Traffic Density** | Classifies traffic as Low / Moderate / High / Jam |
| 🗺️ **Zone Filter** | Draw a custom detection zone on the video frame |
| 🎯 **Object Tracking** | IoU-based tracker for consistent vehicle IDs across frames |
| 🖥️ **Live Dashboard** | Real-time Streamlit dashboard with stats and video feed |

---

## 🛠️ Tech Stack

- **Detection:** YOLOv8 (Ultralytics) / ONNX Runtime
- **Tracking:** Custom IoU-based tracker
- **OCR:** EasyOCR
- **Backend API:** FastAPI
- **Dashboard:** Streamlit
- **Computer Vision:** OpenCV, NumPy
- **Containerization:** Docker, Docker Compose
- **Cloud Deployment:** AWS EC2 + ECR

---

## 📁 Project Structure

```
TrafficAI/
├── dashboard/
│   └── app.py              # Streamlit dashboard (main UI)
├── api/
│   └── api.py              # FastAPI backend
├── counting/
│   └── count.py            # Vehicle counting logic
├── speed/
│   └── speed.py            # Speed estimation
├── lane/
│   └── lane.py             # Lane detection & violation tracking
├── ocr/
│   └── ocr.py              # License plate recognition
├── Density/
│   └── density.py          # Traffic density estimator
├── zone/
│   └── zone.py             # Zone filter (polygon-based ROI)
├── detector/
│   └── detect.py           # Detection wrapper
├── tracker/
│   └── track.py            # Object tracker
├── configs/
│   └── config.py           # Global configuration
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── main.py
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Docker (for containerized deployment)
- YOLOv8 model file (`yolov8n.pt` or custom)

### 1. Clone the repository

```bash
git clone https://github.com/Abdullah8168/TrafficAI.git
cd TrafficAI
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add your YOLO model

Download YOLOv8 nano model:

```bash
python download_model.py
```

Or place your own `.pt` file in the `models/` directory and update `configs/config.py`.

### 4. Run the dashboard

```bash
streamlit run dashboard/app.py
```

### 5. Run the API (optional)

```bash
uvicorn api.api:app --host 0.0.0.0 --port 8000
```

---

## 🐳 Docker Deployment

### Build and run locally

```bash
docker build -t trafficai .
docker run -p 8501:8501 trafficai bash -c "streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
docker run -p 8000:8000 trafficai
```

---

## ☁️ AWS Deployment (EC2 + ECR)

### Push image to ECR

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com
docker tag trafficai:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/trafficai:latest
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/trafficai:latest
```

### Pull and run on EC2

```bash
docker pull <account_id>.dkr.ecr.us-east-1.amazonaws.com/trafficai:latest

# Run API
docker run -d -p 8000:8000 <account_id>.dkr.ecr.us-east-1.amazonaws.com/trafficai:latest

# Run Dashboard
docker run -d -p 8501:8501 <account_id>.dkr.ecr.us-east-1.amazonaws.com/trafficai:latest \
  bash -c "pip install lap --quiet && streamlit run dashboard/app.py --server.port 8501 --server.address 0.0.0.0"
```

Access at: `http://<EC2_PUBLIC_IP>:8501`

> **Note:** Make sure EC2 Security Group allows inbound traffic on ports `8501` and `8000`.

---

## ⚙️ Configuration

Edit `configs/config.py` to set:

```python
MODEL_PATH = "models/yolov8n.pt"
CONFIDENCE_THRESHOLD = 0.20
VEHICLE_CLASSES = [2, 3, 5, 7]   # car, motorcycle, bus, truck (COCO IDs)
```

---

## 📊 Dashboard Usage

1. Upload a video file or enter an IP camera URL
2. Select modules to enable (counting, speed, lane, ANPR)
3. Click **Start** — draw a detection zone on the first frame
4. Click **Confirm Zone** then **Start Processing**
5. Watch the live feed with real-time stats

---

## 🔧 Modules Overview

### Vehicle Counting
- Configurable counting line (slider)
- Spatial deduplication prevents double-counting
- Supports up/down traffic direction

### Speed Estimation
- Pixel-to-meter scale derived from vehicle width
- Smoothed with exponential moving average
- Displays speed in km/h on each vehicle box

### Lane Detection
- Hough transform detects lane boundaries
- Tracks vehicle trajectory over 20 frames
- Flags **WRONG WAY** and **OUTSIDE LANE** violations

### ANPR (License Plate Recognition)
- Runs in a background thread (non-blocking)
- Plate region detected via contour analysis
- EasyOCR reads alphanumeric text

### Traffic Density
- Counts vehicles inside the zone per frame
- Occupancy ratio determines density level
- Badge overlay (Low / Moderate / High / Jam) on video

---

## 📦 Requirements

See `requirements.txt`. Key packages:

```
ultralytics
opencv-python
streamlit
fastapi
uvicorn
easyocr
numpy
streamlit-drawable-canvas
```

---

## 👤 Author

**Abdullah** — [GitHub](https://github.com/Abdullah8168)

---
