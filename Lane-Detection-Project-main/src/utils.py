from __future__ import annotations
from typing import Optional
import cv2
import numpy as np


def preprocess(frame: np.ndarray) -> np.ndarray:
    """Convert a BGR frame to a softly blurred grayscale image."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def detect_edges(frame: np.ndarray) -> np.ndarray:
    """Return Canny edges after grayscale conversion and noise reduction."""
    blurred_gray = preprocess(frame)
    return cv2.Canny(blurred_gray, 50, 150)


def region_of_interest(frame: np.ndarray) -> np.ndarray:
    """Keep only a trapezium covering the road ahead in an edge image."""
    height, width = frame.shape[:2]
    polygon = np.array(
        [[
            (int(width * 0.08), height),
            (int(width * 0.43), int(height * 0.60)),
            (int(width * 0.57), int(height * 0.60)),
            (int(width * 0.92), height),
        ]],
        dtype=np.int32,
    )
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(frame, mask)


def detect_lines(frame: np.ndarray) -> Optional[np.ndarray]:
    """Find short line segments in a ROI-masked Canny edge image."""
    return cv2.HoughLinesP(
        frame,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=35,
        maxLineGap=120,
    )


def _make_line(frame_shape: tuple, slope: float, intercept: float) -> Optional[np.ndarray]:
    """Create one long lane line from its slope/intercept representation."""
    height = frame_shape[0]
    y_bottom = height
    y_top = int(height * 0.60)

    if abs(slope) < 1e-6:
        return None

    x_bottom = int((y_bottom - intercept) / slope)
    x_top = int((y_top - intercept) / slope)
    return np.array([x_bottom, y_bottom, x_top, y_top], dtype=np.int32)


def average_slope_intercept(
    lines: Optional[np.ndarray], frame_shape: Optional[tuple] = None
) -> list[np.ndarray]:
    """Separate Hough segments by slope and return averaged left/right lanes.

    ``frame_shape`` is optional for compatibility when this helper is used on
    its own.  When supplied, returned lines extend from the image bottom to
    the top of the road ROI.
    """
    if lines is None:
        return []

    left_fits: list[tuple[float, float]] = []
    right_fits: list[tuple[float, float]] = []

    for segment in lines:
        x1, y1, x2, y2 = segment.reshape(4)
        if x1 == x2:  
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
        if abs(slope) < 0.35 or abs(slope) > 3.0:
            continue
        if slope < 0:
            left_fits.append((slope, intercept))
        else:
            right_fits.append((slope, intercept))

    averaged_lines: list[np.ndarray] = []
    for fits in (left_fits, right_fits):
        if not fits:
            continue
        slope, intercept = np.mean(fits, axis=0)
        if frame_shape is None:
            averaged_lines.append(np.array([slope, intercept]))
        else:
            lane_line = _make_line(frame_shape, float(slope), float(intercept))
            if lane_line is not None:
                averaged_lines.append(lane_line)

    return averaged_lines


def draw_lines(frame: np.ndarray, lines: list[np.ndarray]) -> np.ndarray:
    """Draw left (blue) and right (red) lanes on a transparent overlay."""
    overlay = np.zeros_like(frame)
    for line in lines:
        x1, y1, x2, y2 = map(int, line.reshape(4))
        # A left lane slopes upward to the right (x_bottom < x_top).
        color = (255, 0, 0) if x1 < x2 else (0, 0, 255) 
        cv2.line(overlay, (x1, y1), (x2, y2), color, 10)
    return cv2.addWeighted(frame, 0.8, overlay, 1.0, 1.0)
