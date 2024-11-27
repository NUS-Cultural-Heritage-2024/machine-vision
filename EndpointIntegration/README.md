
# **Stick Detection Flask Application**

This Flask-based application uses a YOLO (You Only Look Once) model to detect and classify objects (specifically parts of a stick) in real-time from a webcam feed. The application also provides the ability to retrieve detection details in JSON format.

## **Overview**
The app consists of:
- **Real-time video feed** with object detection overlaid (bounding boxes).
- **JSON API endpoints** that provide real-time detection data (coordinates, angles, and lengths) of the detected stick parts.
- **Model selection**: You can load the YOLO model from a local path or download it from Hugging Face if no model path is provided.

### **Flask Endpoints**

#### **1. `/` (Index)**
- **Method**: `GET`
- **Description**: The root route renders the `video1.html` page, which displays the real-time video feed and detected stick details.


#### **2. `/video_feed`**
- **Method**: `GET`
- **Description**: Streams the webcam video with YOLO-based detection applied, i.e., bounding boxes around detected parts of the stick.
- **Returns**: A continuous stream of the video feed with detection boxes.

#### **3. `/stick_data`**
- **Method**: `GET`
- **Description**: Returns the stick detection data as a JSON response. This data includes the normalized coordinates, angles, and lengths of two detected stick parts: `puppetArmTop`, `puppetArmBottom`, `puppetBodyTop`, and `puppetBodyBottom`.
- **Returns**: A JSON object with the following structure:
  ```json
  {
      "puppetArmTop": {"x": <x_value>, "y": <y_value>},
      "puppetArmBottom": {"x": <x_value>, "y": <y_value>},
      "puppetBodyTop": {"x": <x_value>, "y": <y_value>},
      "puppetBodyBottom": {"x": <x_value>, "y": <y_value>}
  }
  ```

### **Arguments and Configuration**

The app supports two command-line arguments:

1. **`--model_path`**:
   - **Type**: String
   - **Description**: Path to the local YOLO model file. If not provided, the model will be downloaded from Hugging Face.
   - **Example**: `--model_path "path/to/yolov5_model.pt"`

2. **`--frame_rate`**:
   - **Type**: Float
   - **Description**: The time interval (in seconds) between each processed frame. This is used to control the processing speed of the video feed.
   - **Default**: `0.2` seconds (i.e., process a frame every 0.2 seconds).
   - **Example**: `--frame_rate 0.5`

### **Model Loading**
- If the `--model_path` argument is provided, the app loads the model from the specified local path.
- If the `--model_path` argument is not provided, the app will download the model from Hugging Face using `hf_hub_download`:
  - **Model Downloaded**: `"Yolov5l_4labels.pt"` from the repository `culturalheritagenus/YoloModelSticks4labels`.

### **How to Run the Application**

1. **Install dependencies**:
   First, install the necessary Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Flask app**:
   You can run the app by providing the model path and frame rate or use the defaults.
   - To use a local model:
     ```bash
     python app.py --model_path /path/to/your/model.pt --frame_rate 0.5
     ```
   - To download the model from Hugging Face (if no model path is provided):
     ```bash
     python app.py --frame_rate 0.5
     ```

3. **Access the application**:
   Once the app is running, you can access it in your browser at:
   ```
   http://127.0.0.1:8000
   ```
