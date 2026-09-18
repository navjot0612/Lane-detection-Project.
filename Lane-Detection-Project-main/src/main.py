from __future__ import annotations
import time
from pathlib import Path
import cv2
import numpy as np

from utils import (
    average_slope_intercept,
    detect_edges,
    detect_lines,
    draw_lines,
    region_of_interest,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
INPUT_VIDEO = PROJECT_DIR / "data" / "input" / "test.mp4"
OUTPUT_VIDEO = PROJECT_DIR / "data" / "output" / "output2_video.mp4"


def smooth_lines(
    current_lines: list[np.ndarray], previous_lines: list[np.ndarray], alpha: float = 0.2
) -> list[np.ndarray]:
    """Use exponential averaging to reduce lane-line flicker between frames."""
    if not previous_lines:
        return current_lines
    if len(current_lines) != len(previous_lines):
        return current_lines
    return [
        (alpha * current + (1 - alpha) * previous).astype(np.int32)
        for current, previous in zip(current_lines, previous_lines)
    ]


def main() -> None:
    """Process the input video, display it, and save the annotated result."""
    if not INPUT_VIDEO.exists():
        raise FileNotFoundError(
            f"Input video was not found: {INPUT_VIDEO}\n"
            "Place test_video.mp4 in data/input and run this script again."
        )

    OUTPUT_VIDEO.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(INPUT_VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open the input video: {INPUT_VIDEO}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the output video. Check codec support and folder access.")

    previous_lines: list[np.ndarray] = []
    previous_time = time.perf_counter()
    print("Processing video... Press Q or Esc in the output window to stop.")

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            edges = detect_edges(frame)
            masked_edges = region_of_interest(edges)
            segments = detect_lines(masked_edges)
            current_lines = average_slope_intercept(segments, frame.shape)
            stable_lines = smooth_lines(current_lines, previous_lines)
            if stable_lines:
                previous_lines = stable_lines

            result = draw_lines(frame, stable_lines)

            now = time.perf_counter()
            fps = 1.0 / max(now - previous_time, 1e-6)
            previous_time = now
            cv2.putText(result, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            writer.write(result)
            cv2.imshow("Lane Detection using OpenCV", result)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"Done. Processed video saved to: {OUTPUT_VIDEO}")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Error: {error}")
