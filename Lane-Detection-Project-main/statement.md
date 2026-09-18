# Project Statement

## Problem Statement

Lane markings provide important visual guidance for safe road navigation. Manual identification of lane boundaries from every frame of a road video is impractical. This project develops a simple real-time system that detects the left and right lane boundaries in a road video and overlays them on each frame.

The system uses only classical computer-vision techniques. It converts each video frame to grayscale, reduces noise with Gaussian blur, extracts strong edges through Canny edge detection, selects the road region using a Region of Interest mask, and identifies line segments with the Hough Line Transform. The identified segments are averaged and smoothed to produce stable lane lines.

## Scope of the Project

The project accepts a pre-recorded road video named `test_video.mp4` and produces an annotated output video. It focuses on visible, reasonably clear lane markings under normal road and lighting conditions. It is intended for learning and academic use.

The scope does not include autonomous vehicle control, steering decisions, object detection, traffic-sign recognition, GPS integration, camera calibration, or machine-learning-based lane segmentation. Performance may reduce when lane markings are faded, heavily occluded, curved, or affected by severe weather and lighting conditions.

## Target Users

- Students learning Python, OpenCV, and Computer Vision
- Faculty members evaluating a basic image-processing project
- Beginners who want to understand lane detection without deep learning
- Developers using the project as a starting point for classical road-scene analysis

## High-Level Features

- Reads a road video from the specified input location
- Detects potential lane edges using Canny edge detection
- Restricts processing to the road area using a trapezium ROI
- Finds lane-like line segments using the Hough Line Transform
- Separates left and right lane candidates according to their slope
- Averages and smooths lane lines to improve visual stability
- Displays processing FPS and a live output window
- Saves the final annotated video to the output folder
