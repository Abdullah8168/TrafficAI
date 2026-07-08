"""
TrafficAI - Detection Zone Filter (Module 9)

Allows the user to define a polygon region of interest (ROI) on the video frame.
Only vehicles whose centre point falls inside the polygon are processed.

Usage:
    from zone.zone import ZoneFilter

    zone = ZoneFilter(polygon=[[x1,y1],[x2,y2],[x3,y3]])
    if zone.is_inside(cx, cy):
        # process this detection
    zone.draw(frame)   # overlay the zone on the frame
"""

import cv2
import numpy as np


class ZoneFilter:
    """
    Polygon-based region-of-interest filter.

    Parameters
    ----------
    polygon : list of [x, y] | None
        Vertices of the detection zone in pixel coordinates.
        Pass None (or fewer than 3 points) for full-frame detection.
    """

    BORDER_COLOR  = (0, 230, 0)      # bright green border
    FILL_ALPHA    = 0.12             # overlay transparency

    def __init__(self, polygon=None):
        self.polygon = polygon if (polygon and len(polygon) >= 3) else None

    # ------------------------------------------------------------------
    @property
    def active(self):
        """True when a valid zone is set."""
        return self.polygon is not None

    # ------------------------------------------------------------------
    def polygon_area(self):
        """Return area of the zone polygon in pixels² (shoelace formula)."""
        if not self.active:
            return None
        n    = len(self.polygon)
        area = 0.0
        for i in range(n):
            j     = (i + 1) % n
            area += self.polygon[i][0] * self.polygon[j][1]
            area -= self.polygon[j][0] * self.polygon[i][1]
        return abs(area) / 2.0

    # ------------------------------------------------------------------
    def is_inside(self, cx, cy):
        """
        Return True if point (cx, cy) is inside the zone polygon.
        Always returns True when no zone is set (full-frame mode).
        """
        if not self.active:
            return True
        return self._ray_cast(float(cx), float(cy), self.polygon)

    # ------------------------------------------------------------------
    def draw(self, frame):
        """
        Draw the zone polygon on *frame* in-place.
        Green semi-transparent fill + solid border.
        """
        if not self.active:
            return frame

        pts = np.array(self.polygon, dtype=np.int32)

        # Semi-transparent fill
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], self.BORDER_COLOR)
        cv2.addWeighted(overlay, self.FILL_ALPHA, frame, 1 - self.FILL_ALPHA,
                        0, frame)

        # Solid border
        cv2.polylines(frame, [pts], isClosed=True,
                      color=self.BORDER_COLOR, thickness=2)

        # Label
        cx = int(np.mean([p[0] for p in self.polygon]))
        cy = int(np.mean([p[1] for p in self.polygon]))
        cv2.putText(frame, "ZONE", (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    self.BORDER_COLOR, 2, cv2.LINE_AA)

        return frame

    # ------------------------------------------------------------------
    @staticmethod
    def _ray_cast(x, y, poly):
        """Ray-casting algorithm for point-in-polygon test."""
        n      = len(poly)
        inside = False
        j      = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside

    # ------------------------------------------------------------------
    @classmethod
    def from_canvas(cls, json_data, canvas_w, canvas_h, frame_w, frame_h):
        """
        Build a ZoneFilter from 4 points clicked on streamlit-drawable-canvas
        (point / circle drawing mode).

        Each click creates a small circle object. We collect all circle centres,
        sort them clockwise around their centroid so the lines don't cross, then
        scale to frame pixel coordinates.
        """
        import math

        if not json_data or not json_data.get("objects"):
            return cls()

        pts_canvas = []
        for obj in json_data["objects"]:
            if obj.get("type") != "circle":
                continue

            radius = float(obj.get("radius", 0)) * float(obj.get("scaleX", 1))
            ox     = obj.get("originX", "left")
            oy     = obj.get("originY", "top")

            # fabric.js stores centre at (left, top) when originX/Y == 'center',
            # otherwise at (left + radius, top + radius).
            cx = float(obj.get("left", 0)) if ox == "center" else float(obj.get("left", 0)) + radius
            cy = float(obj.get("top",  0)) if oy == "center" else float(obj.get("top",  0)) + radius

            pts_canvas.append([cx, cy])

        if len(pts_canvas) < 3:
            return cls()

        # Sort points clockwise around centroid so polygon edges don't cross
        mx = sum(p[0] for p in pts_canvas) / len(pts_canvas)
        my = sum(p[1] for p in pts_canvas) / len(pts_canvas)
        pts_canvas.sort(key=lambda p: math.atan2(p[1] - my, p[0] - mx))

        # Scale to frame coordinates
        pts = [
            [p[0] * frame_w / canvas_w, p[1] * frame_h / canvas_h]
            for p in pts_canvas
        ]
        return cls(pts)