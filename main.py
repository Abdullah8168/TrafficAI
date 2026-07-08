"""
TrafficAI

Run Examples

Image detection:
    python main.py --image data/images/image1.jpg

Video detection:
    python main.py --video data/videos/video.mp4

Webcam detection:
    python main.py --webcam

Video tracking (Module 2):
    python main.py --video data/videos/video.mp4 --track

Video counting (Module 3):
    python main.py --video data/videos/video.mp4 --count

Video speed estimation (Module 4):
    python main.py --video data/videos/video.mp4 --speed

Video lane detection (Module 5):
    python main.py --video data/videos/video.mp4 --lane

Video ANPR (Module 6):
    python main.py --video data/videos/video.mp4 --anpr
    python main.py --image data/images/image1.jpg --anpr
"""

import argparse

from detector.detect import VehicleDetector
from tracker.track import VehicleTracker
from counting.count import VehicleCounter
from speed.speed import SpeedEstimator
from lane.lane import LaneDetector
from ocr.ocr import ANPRDetector


def main():

    parser = argparse.ArgumentParser(
        description="TrafficAI — Detection, Tracking, Counting, Speed, Lane & ANPR"
    )

    parser.add_argument("--image",     type=str,            help="Image path")
    parser.add_argument("--video",     type=str,            help="Video path")
    parser.add_argument("--webcam",    action="store_true", help="Use webcam")

    parser.add_argument("--track",     action="store_true",
                        help="Enable tracking (Module 2)")

    parser.add_argument("--count",     action="store_true",
                        help="Enable counting (Module 3)")

    parser.add_argument("--line",      type=float, default=0.5,
                        help="Counting line position 0.0–1.0 (default: 0.5)")

    parser.add_argument("--speed",     action="store_true",
                        help="Enable speed estimation (Module 4)")

    parser.add_argument("--lane",      action="store_true",
                        help="Enable lane detection (Module 5)")

    parser.add_argument("--direction", type=str, default="down",
                        choices=["down", "up"],
                        help="Expected traffic direction in frame (default: down)")

    parser.add_argument("--anpr",      action="store_true",
                        help="Enable license plate recognition (Module 6)")

    parser.add_argument("--conf",      type=float, default=None,
                        help="Confidence threshold")

    args = parser.parse_args()

    conf = args.conf if args.conf else 0.40

    # ── ANPR MODE (Module 6) ──────────────────────────────────────────────────
    if args.anpr:

        detector = ANPRDetector(conf=conf)

        if args.image:
            output = detector.detect_image(args.image)
            print(f"\nResult saved at:\n{output}")

        elif args.video:
            output = detector.detect_video(args.video)
            print(f"\nANPR video saved at:\n{output}")

        elif args.webcam:
            detector.detect_webcam()

        else:
            print("[ERROR] --anpr requires --image, --video or --webcam")
            parser.print_help()

    # ── LANE MODE (Module 5) ──────────────────────────────────────────────────
    elif args.lane:

        detector = LaneDetector(conf=conf, traffic_direction=args.direction)

        if args.video:
            output = detector.detect_video(args.video)
            print(f"\nLane video saved at:\n{output}")

        elif args.webcam:
            detector.detect_webcam()

        else:
            print("[ERROR] --lane requires --video or --webcam")
            parser.print_help()

    # ── SPEED MODE (Module 4) ─────────────────────────────────────────────────
    elif args.speed:

        estimator = SpeedEstimator(conf=conf)

        if args.video:
            output = estimator.estimate_video(args.video)
            print(f"\nSpeed video saved at:\n{output}")

        elif args.webcam:
            estimator.estimate_webcam()

        else:
            print("[ERROR] --speed requires --video or --webcam")
            parser.print_help()

    # ── COUNTING MODE (Module 3) ──────────────────────────────────────────────
    elif args.count:

        counter = VehicleCounter(conf=conf, line_position=args.line)

        if args.video:
            output = counter.count_video(args.video)
            print(f"\nCounted video saved at:\n{output}")

        elif args.webcam:
            counter.count_webcam()

        else:
            print("[ERROR] --count requires --video or --webcam")
            parser.print_help()

    # ── TRACKING MODE (Module 2) ──────────────────────────────────────────────
    elif args.track:

        tracker = VehicleTracker(conf=conf)

        if args.video:
            output = tracker.track_video(args.video)
            print(f"\nTracked video saved at:\n{output}")

        elif args.webcam:
            tracker.track_webcam()

        else:
            print("[ERROR] --track requires --video or --webcam")
            parser.print_help()

    # ── DETECTION MODE (Module 1) ─────────────────────────────────────────────
    else:

        detector = VehicleDetector(conf=conf)

        if args.image:
            output = detector.detect_image(args.image)
            print(f"\nImage saved at:\n{output}")

        elif args.video:
            output = detector.detect_video(args.video)
            print(f"\nVideo saved at:\n{output}")

        elif args.webcam:
            detector.detect_webcam()

        else:
            parser.print_help()


if __name__ == "__main__":
    main()