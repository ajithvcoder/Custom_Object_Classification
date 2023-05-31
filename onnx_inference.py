import onnx
import onnxruntime
import numpy as np
import torchvision.transforms as transforms
import cv2
import os

# Load the ONNX model
onnx_model = onnx.load("./savedModels/custommodel.onnx")

# Create an ONNX runtime session
ort_session = onnxruntime.InferenceSession("./savedModels/custommodel.onnx")

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229,0.224,0.225]),
])

def get_folder_names(directory):
    folder_names = []
    for root, dirs, files in os.walk(directory):
        for dir_name in dirs:
            folder_names.append(dir_name)
    return folder_names

# Give the dataset directory
class_label = get_folder_names("./data/Fruits/train/")

# img = cv2.imread("./test_data/Image_29.jpg")
img = cv2.imread("./test_data/Image_32.jpg")

img = cv2.resize(img, (224,224))
trans_img = val_transform(img)
trans_img = trans_img.unsqueeze(0)

# Run the ONNX model
inputs = {"input": trans_img.numpy()}
outputs = ort_session.run(None, inputs)

# Get the predicted class
probs = outputs[0][0]
class_idx = np.argmax(probs)

# Print the predicted class
print(f"Predicted class: {class_label[class_idx]}")

#Write predictions to image
# Define the text and position
text = class_label[class_idx]
position = (10, 20)  # (x, y) coordinates of the starting point of the text

# Define the font properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
color = (255, 0, 0)  # BGR color format (Blue, Green, Red)
thickness = 2

# Write the text on the image
cv2.putText(img, text, position, font, font_scale, color, thickness)
cv2.imwrite("output/predict_image_3.png",img)