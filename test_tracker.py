from tracker.track import VehicleTracker
import cv2

tracker = VehicleTracker()

cap = cv2.VideoCapture("data/videos/traffic.mp4")

ret, frame = cap.read()

if ret:

    results = tracker.track_frame(frame)

    print(results)

cap.release()