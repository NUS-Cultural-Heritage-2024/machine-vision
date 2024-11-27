from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2
import time
import argparse
import os

app = Flask(__name__)

# ------------------- Argument Parser -------------------
parser = argparse.ArgumentParser(description="YOLO Flask App")
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to the local YOLO model file. If not provided, the model will be downloaded from Hugging Face."
)
parser.add_argument(
    "--frame_rate",
    type=float,
    default=0.2,
    help="Time interval between processing frames in seconds. Default is 0.2 seconds."
)
args = parser.parse_args()

# ------------------- Load YOLO Model -------------------
if args.model_path:
    print(f"Loading model from provided path: {args.model_path}")
    model = YOLO(args.model_path)
else:
    print("No model path provided. Downloading model from Hugging Face...")
    model_path = hf_hub_download(repo_id="culturalheritagenus/YoloModelSticks4labels", filename="Yolov5l_4labels.pt")
    model = YOLO(model_path)

# Frame processing interval
#By default set to 0.2 seconds
frame_rate = 0.2
#Set Time interval
if args.frame_rate:
    frame_rate = args.frame_rate

# ------------------- Video Capture -------------------
cap = cv2.VideoCapture(0)
last_frame_time = time.time()  # For frame rate control
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ------------------- Helper Functions -------------------
def normalize_coordinates(x_pixel, y_pixel, frame_width, frame_height):
    """Normalize pixel coordinates to a custom range."""
    x_normalized = (x_pixel - frame_width / 2) / (frame_width / 16)
    y_normalized = -(y_pixel - frame_height / 2) / (frame_height / 9)
    return x_normalized, y_normalized

def process_frame(frame):
    """Process a single frame to extract YOLO detections and return JSON data."""
    results = model.predict(source=frame, conf=0.5)
    json_data = {
        "puppetArmTop": {"x": 0, "y": 0},
        "puppetArmBottom": {"x": 0, "y": 0},
        "puppetBodyTop": {"x": 0, "y": 0},
        "puppetBodyBottom": {"x": 0, "y": 0}
    }
    label_map = {
        0: "puppetBodyBottom",
        1: "puppetBodyTop",
        2: "puppetArmBottom",
        3: "puppetArmTop",
    }

    for result in results:
        boxes = result.boxes.xyxy
        class_ids = result.boxes.cls
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i].tolist()
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            class_id = int(class_ids[i])
            if class_id in label_map:
                x_normalized, y_normalized = normalize_coordinates(
                    x_center, y_center, frame_width, frame_height)
                json_data[label_map[class_id]] = {
                    "x": round(x_normalized, 2),
                    "y": round(y_normalized, 2)
                }
                cv2.circle(frame, (int(x_center), int(y_center)),
                           radius=5, color=(0, 255, 0), thickness=-1)
                cv2.putText(frame, f"{label_map[class_id]}",
                            (int(x_center), int(y_center) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame, json_data

@app.route("/")
def index():
    """Render the main page."""
    return render_template("video1.html")

def gen_frames():
    """Generate video frames for the video feed."""
    global last_frame_time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Control the frame processing rate
        if time.time() - last_frame_time >= frame_rate:
            last_frame_time = time.time()
            frame, _ = process_frame(frame)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route("/video_feed")
def video_feed():
    """Stream the video feed."""
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/stick_data")
def stick_data():
    """Return stick detection data in JSON."""
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "Failed to grab frame"})
    _, json_data = process_frame(frame)
    return jsonify(json_data)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000)
