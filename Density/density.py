"""
TrafficAI - Traffic Density Estimator (Module 8)

Classifies traffic density for each frame based on:
  - Active vehicle count  (how many vehicles are visible)
  - Road occupancy %      (total vehicle box area / frame area * 100)

Density levels:
  LOW    -> count < 4  AND occupancy < 10 %
  MEDIUM -> count 4-7  OR  occupancy 10-25 %
  HIGH   -> count >= 8 OR  occupancy >= 25 %

Usage:
    from density.density import DensityEstimator

    estimator = DensityEstimator()
    level, count, occupancy = estimator.update(boxes, frame_h, frame_w)
"""


class DensityEstimator:
    """
    Estimates traffic density from vehicle bounding boxes in a single frame.

    Parameters
    ----------
    low_count    : max vehicles for LOW level  (default 4)
    high_count   : min vehicles for HIGH level (default 8)
    low_occ      : max occupancy % for LOW     (default 10.0)
    high_occ     : min occupancy % for HIGH    (default 25.0)
    """

    LEVELS = ("LOW", "MEDIUM", "HIGH")

    # Badge colours (BGR) for cv2 overlay
    COLORS = {
        "LOW":    (0, 200, 0),
        "MEDIUM": (0, 165, 255),
        "HIGH":   (0, 0, 220),
    }

    # Emoji for Streamlit display
    EMOJI = {
        "LOW":    "🟢",
        "MEDIUM": "🟡",
        "HIGH":   "🔴",
    }

    def __init__(self,
                 low_count=4,  high_count=8,
                 low_occ=10.0, high_occ=25.0):
        self.low_count  = low_count
        self.high_count = high_count
        self.low_occ    = low_occ
        self.high_occ   = high_occ

        # Last computed values (readable from outside)
        self.level     = "LOW"
        self.count     = 0
        self.occupancy = 0.0   # percent

    # ------------------------------------------------------------------
    def update(self, boxes, frame_h, frame_w, zone_area=None):
        """
        Compute density for the current frame.

        Parameters
        ----------
        boxes     : list of [x1, y1, x2, y2]  (vehicles inside zone only)
        frame_h   : frame height in pixels
        frame_w   : frame width  in pixels
        zone_area : area of the detection zone in pixels² (optional).
                    If provided, occupancy is calculated relative to the zone
                    area instead of the full frame area.

        Returns
        -------
        (level, count, occupancy)
            level     : str  "LOW" | "MEDIUM" | "HIGH"
            count     : int  number of vehicles in zone
            occupancy : float  % of zone (or frame) area covered by vehicles
        """
        ref_area = max(zone_area if zone_area is not None else frame_h * frame_w, 1)
        box_area = sum(
            max(0, x2 - x1) * max(0, y2 - y1)
            for x1, y1, x2, y2 in boxes
        )

        self.count     = len(boxes)
        self.occupancy = round(box_area / ref_area * 100, 1)

        n   = self.count
        occ = self.occupancy

        if n >= self.high_count or occ >= self.high_occ:
            self.level = "HIGH"
        elif n >= self.low_count or occ >= self.low_occ:
            self.level = "MEDIUM"
        else:
            self.level = "LOW"

        return self.level, self.count, self.occupancy

    # ------------------------------------------------------------------
    def badge_text(self):
        """Short string for cv2 overlay, e.g. 'Density: HIGH  9v  28.3%'"""
        return f"Density: {self.level}  {self.count}v  {self.occupancy:.1f}%"

    def badge_color(self):
        """BGR tuple for cv2 rectangle."""
        return self.COLORS[self.level]

    def streamlit_text(self):
        """Markdown string for Streamlit stats panel."""
        emoji = self.EMOJI[self.level]
        return (
            f"**{emoji} {self.level}**\n\n"
            f"- Vehicles in zone: {self.count}\n"
            f"- Zone occupancy: {self.occupancy:.1f}%"
        )