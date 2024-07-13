# Pose Detection with MediaPipe and OpenCV

This Python script uses MediaPipe and OpenCV to perform real-time pose detection using your computer's webcam. It detects the pose of a person and calculates the angle between the right shoulder, elbow, and wrist. It then compares this angle with a predefined threshold (45 degrees in this case) and displays the results on the video feed.

## Prerequisites

- Python 3.x
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)

## Usage

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>