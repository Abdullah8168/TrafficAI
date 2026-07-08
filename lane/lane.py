"""
TrafficAI - Lane Detection Module (Module 5)

Provides LaneDetector for integration with the main dashboard pipeline.
app.py calls detect_lines() every N frames, draw() every frame,
and update() per vehicle to get violation results.

Violations detected:
  - WRONG WAY   : vehicle moving against expected traffic direction
  - OUTSIDE LANE: vehicle centre is outside every detected lane corridor
"""

import cv2
import numpy as np


class LaneDetector:
    """
    Lane-line detector and per-vehicle violation checker.

    Usage (inside TrafficPipeline):
    ─────────────────────────────────────────────────────
        # Init once
        self.lane_detector = LaneDetector(direction="down")

        # Every 10 frames — detect and cache lane lines
        self._cached_lane_lines = self.lane_detector.detect_lines(frame, h, w)

        # Every frame — draw cached lines
        self.lane_detector.draw(frame, self._cached_lane_lines)

        # Per vehicle in the detection loop
        violation = self.lane_detector.update(
            track_id, cx, cy, self._cached_lane_lines, w
        )
    """

    def __init__(self, direction="down"):
        """
        direction : "down" → vehicles travel top-to-bottom (default)
                    "up"   → vehicles travel bottom-to-top
        """
        self.direction    = direction
        self._cy_hist     = {}   # track_id → list[cy]          (wrong-way history)
        self._outside_cnt = {}   # track_id → int               (hysteresis counter)
        self.violations   = {}   # track_id → "WRONG WAY" | "OUTSIDE LANE"

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def detect_lines(self, frame, h, w):
        """
        Run Canny edge detection + probabilistic Hough on the bottom half of
        the frame.  Returns a list of (x1, y1, x2, y2) tuples.

        Designed to be called every N frames (e.g. every 10) and cached.
        """
        gray   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gray, (5, 5), 0)
        edges  = cv2.Canny(blur, 50, 150)

        # ROI: bottom half of frame only (lanes are usually below the horizon)
        mask   = np.zeros_like(edges)
        roi    = np.array([[(0, h), (0, h // 2), (w, h // 2), (w, h)]],
                          dtype=np.int32)
        cv2.fillPoly(mask, roi, 255)
        masked = cv2.bitwise_and(edges, mask)

        lines  = cv2.HoughLinesP(masked, 1, np.pi / 180, 50,
                                  minLineLength=80, maxLineGap=150)
        if lines is None:
            return []
        return [tuple(line[0]) for line in lines]

    def draw(self, frame, lines):
        """Draw detected lane lines on *frame* in-place (cyan)."""
        for x1, y1, x2, y2 in lines:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        return frame

    def update(self, track_id, cx, cy, lines, frame_w):
        """
        Check lane violations for one vehicle in the current frame.

        Parameters
        ----------
        track_id : int
        cx, cy   : vehicle centre (original-frame pixel coordinates)
        lines    : list[(x1,y1,x2,y2)] — cached from detect_lines()
        frame_w  : frame width in pixels

        Returns
        -------
        str | None  — current violation for this track_id, or None
        """
        # ── Update y-history for wrong-way check ─────────────────────────────
        hist = self._cy_hist.setdefault(track_id, [])
        hist.append(cy)
        if len(hist) > 20:
            hist.pop(0)

        # ── Wrong-way detection ───────────────────────────────────────────────
        if len(hist) >= 10:
            dy    = hist[-1] - hist[0]
            wrong = (
                (self.direction == "down" and dy < -30) or
                (self.direction == "up"   and dy >  30)
            )
            if wrong:
                self.violations[track_id] = "WRONG WAY"
                return "WRONG WAY"

        # ── Outside-lane detection (skipped if WRONG WAY already set) ─────────
        if self.violations.get(track_id) != "WRONG WAY":
            outside = self._is_outside_lane(cx, cy, lines, frame_w)
            if outside:
                cnt = self._outside_cnt.get(track_id, 0) + 1
                self._outside_cnt[track_id] = cnt
                # 8-frame hysteresis — ignore brief excursions (false positives)
                if cnt >= 8:
                    self.violations[track_id] = "OUTSIDE LANE"
            else:
                self._outside_cnt[track_id] = 0
                # Clear violation when vehicle moves back inside a lane
                if self.violations.get(track_id) == "OUTSIDE LANE":
                    del self.violations[track_id]

        return self.violations.get(track_id)

    def get_violation(self, track_id):
        """Return current violation string for track_id, or None."""
        return self.violations.get(track_id)

    # ─────────────────────────────────────────────────────────────────────────
    # Internal
    # ─────────────────────────────────────────────────────────────────────────

    def _is_outside_lane(self, cx, cy, lines, w):
        """
        Returns True if (cx, cy) lies outside every detected lane corridor.

        Algorithm:
          1. Extrapolate each non-horizontal Hough line to the vehicle's y.
          2. Sort the resulting x-intercepts left-to-right.
          3. Any adjacent pair separated by >= 8% of frame width is treated as
             a valid lane corridor.
          4. If cx falls inside any corridor -> inside lane -> return False.
             If cx falls in none -> outside lane -> return True.
        """
        if not lines:
            return False

        x_at_y = []
        for x1, y1, x2, y2 in lines:
            dy = y2 - y1
            if abs(dy) < 15:                   # skip near-horizontal lines
                continue
            x = x1 + (cy - y1) * (x2 - x1) / dy
            if -20 <= x <= w + 20:             # small margin beyond frame edges
                x_at_y.append(x)

        if len(x_at_y) < 2:
            return False                       # not enough lines -> no verdict

        x_at_y.sort()
        min_lane_w = w * 0.08                  # 8% of frame width minimum

        for i in range(len(x_at_y) - 1):
            if x_at_y[i + 1] - x_at_y[i] >= min_lane_w:
                if x_at_y[i] <= cx <= x_at_y[i + 1]:
                    return False               # inside this lane corridor

        return True                            # outside every corridor