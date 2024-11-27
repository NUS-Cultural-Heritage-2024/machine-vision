# machine-vision
Repository for the Sticks endpoint classification and detection

Model Test Script - yoloModelTestScript.py

This Python script utilizes a YOLO (You Only Look Once) model to perform real-time object detection and tracking, specifically designed for detecting and localizing four key points on a stick-like object. It generates normalized coordinates in JSON format for further use, such as puppet control systems.

## **Features**
- Real-time video capture and object detection.
- Normalizes detected object coordinates to a custom range.
- Outputs results in JSON format for easy integration.
- Supports model loading from a local path or downloading from Hugging Face.
- Visualizes the detections on the video feed.

## **Requirements(versions given in requirements.txt)**
- Python 3.11.4
- OpenCV
- PyTorch
- Ultralytics YOLO
- Hugging Face Hub
- Flask

## **Setup for testing the Model independently and get video feed**

1. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Script**:
   To use the script, run the following command:
   ```bash
   python yoloModelTestScript.py --model_path <path_to_model_file> --interval <time_interval>
   ```
   - Replace `<path_to_model_file>` with the path to your YOLO model file.
   - Replace `<time_interval>` (optional) with the desired interval (in seconds) between processing frames (default: 0.2 seconds).

   or Simply run the script as below. It will fetch the trained model Yolov5l_4labels.pt(Trained on Yolov5 large) from hugging face and use default interval of 0.2 seconds
   ```bash
   python yoloModelTestScript.py
   ```

## **Output Example**
The script generates normalized JSON data for four specific points:
```json
{
    "puppetArmTop": {"x": -1.23, "y": 3.45},
    "puppetArmBottom": {"x": 2.34, "y": -0.56},
    "puppetBodyTop": {"x": 0.78, "y": -1.23},
    "puppetBodyBottom": {"x": 0, "y": 0}  // Missing detection
}
```


## **Notes**
- If a model path is not provided, the script downloads a pre-trained model from Hugging Face.