# Eye Controlled Mouse Using Computer Vision

This project implements a real-time eye-controlled mouse system using OpenCV, MediaPipe FaceMesh, and PyAutoGUI.

## Features
- Cursor control using iris tracking
- Mouse click using eye blink detection
- Smooth cursor movement
- Cooldown-based click prevention

## Tech Stack
- Python
- OpenCV
- MediaPipe
- PyAutoGUI

## How It Works
- FaceMesh detects facial landmarks
- Iris landmarks are mapped to screen coordinates
- Blink is detected using eyelid distance threshold

## Run
```bash
pip install opencv-python mediapipe pyautogui
python main.py
