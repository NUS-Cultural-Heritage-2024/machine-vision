import torch
import torchvision.transforms as T
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import time

# Function to test on a single image
def test_on_image(model, image_path):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    transform = T.ToTensor()
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        predictions = model(img_tensor)
    end_time = time.time()
    time_taken = end_time - start_time
    print("Time required to show the output:", time_taken, "seconds")

    # Process results and draw keypoints
    img = cv2.imread(image_path)
    for i, keypoint in enumerate(predictions[0]['keypoints'][0]):
        x, y, visibility = keypoint
        if visibility > 0:  # Only draw visible keypoints
            cv2.circle(img, (int(x), int(y)), radius=12, color=(0, 255, 0), thickness=-1)
            cv2.putText(img, f"K{i+1}", (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the result
    #img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("generated_image.jpg", img)

    # plt.imshow(img_rgb)
    # plt.axis("off")
    # plt.show()

# Example usage
#img_path = r"C:\Users\Manav\Desktop\KeypointEstimation\codebase\stick_endpoint_images\frame_0670.jpg"
#img_path=  r"C:\Users\Manav\Pictures\Camera Roll\image2.jpg"
#img_path =  r"C:\Users\Manav\Desktop\KeypointEstimation\codebase\testImagesLocal\image1.jpeg"
#test_on_image(model, img_path)

## Add Variables here!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Load the trained model
model_path = "PoseModelResnet50.pth"
img_path = "image1.jpg"

model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
if device.type == "cpu":
    model = torch.load(model_path,map_location=torch.device('cpu'))
elif device.type == "cuda":
    model = torch.load(model_path)

model = model.to(device)
model.eval()

test_on_image(model, img_path)

        