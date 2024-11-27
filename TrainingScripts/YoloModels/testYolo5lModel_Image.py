from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import time

def test_yolo_on_image(model_path, image_path):
    # Load the YOLO model

    model = YOLO(model_path)

    # Read the image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB for display

    # Run inference
    start_time = time.time()
    results = model.predict(source=img, conf=0.5)  # Adjust confidence threshold as needed
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time required to show the output:", time_taken, "seconds")
    # Display results
    result_image = results[0].plot()  # Get the image with bounding boxes

    cv2.imwrite("generated_image.jpg", result_image)
    # plt.imshow(result_image)
    # plt.axis('off')
    # plt.show()

# Add Variables here
model_path = "Yolov5l_4labels.pt"
img_path = r"C:\Users\Manav\Desktop\KeypointEstimation\codebase\testImagesLocal\\"
test_yolo_on_image(, img_path)
