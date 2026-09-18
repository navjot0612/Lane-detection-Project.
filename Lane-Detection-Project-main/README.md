# Real-Time Lane Detection using OpenCV

## Overview

This project detects road lane boundaries from a recorded video using classical computer-vision methods. Each frame is processed to identify the left and right lane markings, draw them on the original frame, and save the annotated result as a new video. No machine learning, training data, or deep-learning model is used.

It is designed as a simple, modular Computer Vision project suitable for academic demonstration and viva examination.

## Features

- Processes `test_video.mp4` frame by frame
- Uses grayscale conversion, Gaussian blur, Canny edge detection, ROI masking, and Hough Line Transform
- Separates, averages, and smooths left and right lane lines
- Draws blue left-lane and red right-lane boundaries
- Displays processing FPS and a live output window
- Saves the processed result as an MP4 video

## Technologies Used

- Python 3
- OpenCV (`opencv-python`)
- NumPy
- Classical Computer Vision techniques: Canny edge detection, ROI masking, and Hough Line Transform

## Folder Structure

```text
lane_detection_project/
├── data/
│   ├── input/
│   │   └── test_video.mp4
│   └── output/
│       └── output_video.mp4
├── docs/
│   ├── REPORT_CONTENT.md
│   ├── DIAGRAMS.md
│   └── TESTING_AND_SUBMISSION.md
├── src/
│   ├── main.py
│   └── utils.py
├── statement.md
├── requirements.txt
└── README.md
```

## Installation and Execution

1. Open a terminal in the `lane_detection_project` folder.
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure that the input video is available at `data/input/test_video.mp4`.
4. Run the project:

   ```bash
   python src/main.py
   ```

Press `Q` or `Esc` in the output window to stop processing early.

## Output Description

The system displays the original road scene with detected lanes overlaid. The left lane is blue, the right lane is red, and current processing FPS is shown in green. The processed video is saved to `data/output/output_video.mp4`.

## Screenshots

Add screenshots after running the project.

| Input Video Frame | Detected Lane Output |
| --- | --- |
| `![Input frame](docs/screenshots/input_frame.png)` | `![Output frame](docs/screenshots/output_frame.png)` |

> Create `docs/screenshots/` and replace the placeholders with captured screenshots before final submission.

## Documentation

- [Problem statement](statement.md)
- [Full report content](docs/REPORT_CONTENT.md)
- [Diagram descriptions and Mermaid code](docs/DIAGRAMS.md)
- [Testing plan and GitHub submission checklist](docs/TESTING_AND_SUBMISSION.md)
