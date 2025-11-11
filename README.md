# Taekwondo Tutorial Hub

A Flask + Socket.IO web application that teaches Taekwondo forms with the help of real-time pose detection. The app pairs instructional videos with a webcam feed, uses a YOLOv8 pose model to compare the practitioner against reference poses, and gives instant feedback when a pose is matched.

## Features
- Guided tutorial flow that alternates between short demo clips and live practice.
- Real-time pose estimation via `ultralytics` YOLOv8n pose weights (`yolov8n-pose.pt`).
- Similarity scoring and celebratory overlays when a pose is mastered.
- Responsive Bootstrap UI with video carousel, progress indicators, and success messaging.
- WebSocket streaming of webcam frames for low-latency feedback powered by Eventlet.

## Project Structure
```
.
|-- app.py                 # Flask application + Socket.IO events
|-- pose_detector.py       # YOLO-based pose processing utilities
|-- templates/             # Jinja2 templates (landing page, tutorial flow)
|   |-- index.html
|   |-- tutorial.html
|-- static/
|   |-- assets/            # Branding images
|   |-- audio/             # Celebratory audio clip
|   |-- js/
|   |   |-- camera-client.js   # Browser-side webcam capture + socket client
|   |-- poses/             # Reference pose images loaded at startup
|   |-- poses_video/       # Tutorial video clips that precede each pose
|-- requirements.txt       # Python dependencies
|-- runtime.txt            # Python version hint for PaaS (e.g., Heroku)
|-- Procfile               # Gunicorn entry point for production
|-- yolov8n-pose.pt        # Pretrained YOLOv8n pose weights
```

## How It Works
1. `pose_detector.PoseDetector` loads the YOLOv8n pose model and calculates similarity between detected landmarks and reference poses.
2. At startup `app.py` scans `static/poses` and builds `tutorial_poses`, so new reference images are picked up automatically.
3. The tutorial page (`templates/tutorial.html`) plays a short video, then switches to the webcam and requests camera access.
4. `static/js/camera-client.js` captures frames at ~3 FPS, sends them over a Socket.IO channel, and displays the processed frames returned by the server.
5. When similarity exceeds the configured threshold (default 90%), the server emits a `pose_match_confirmed` event and the UI advances to the next pose.

## Requirements
- Python 3.10 (matches `runtime.txt`).
- pip + virtual environment management (recommended).
- A machine with a webcam and sufficient CPU/GPU resources to run YOLOv8 pose inference. GPU acceleration (CUDA) is optional but speeds up detection.

## Local Setup
```
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```
> Installing `torch` and `torchvision` can take time. On some platforms, you may need to install a wheel that matches your CUDA/cuDNN stack; see https://pytorch.org/get-started/locally for alternatives.

## Running the App
```
python app.py
```
- By default the server finds an open port starting at `10000`. Check the terminal logs for the chosen port and open `http://localhost:<port>` in your browser.
- Set the `PORT` environment variable to force a specific port (useful for cloud deployment).

For production-style hosting (e.g., Heroku) the included `Procfile` runs:
```
gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
```
Ensure the `yolov8n-pose.pt` weights file is present in the deployment slug.

## Using the Tutorial
1. Navigate to the tutorial page via the **Start Training** button.
2. Watch the demonstration clip for the current pose.
3. When prompted, allow camera access so the app can capture your movements.
4. Mirror the reference pose shown on screen; the similarity score is evaluated continuously.
5. On a successful match, the app congratulates you, saves the best frame server-side, and advances to the next drill.

## Customisation Tips
- **Thresholds**: Adjust `similarity_threshold` when instantiating `PoseDetector` in `app.py` to make the match stricter or more lenient.
- **Reference Library**: Add, rename, or remove images in `static/poses`. Files are sorted alphabetically, so prefix with numbers to control order.
- **Demo Videos**: Place new clips in `static/poses_video` and update the tutorial script if the naming scheme changes.
- **Frontend Tweaks**: Modify `templates/` and `static/js/camera-client.js` to refine the UX, messaging, or streaming cadence.

## Troubleshooting
- **Camera access denied**: Grant permission in the browser and reload the page.
- **Model not loading**: Verify that `yolov8n-pose.pt` exists alongside the code and that `ultralytics` can read it.
- **Slow/laggy inference**: Lower the frame rate (`FPS`) in `camera-client.js`, reduce `imgsz` in `pose_detector.py`, or leverage a GPU-enabled environment.
- **Torch install issues**: Use pre-built wheels from PyTorch's official website that match your Python version and platform.
