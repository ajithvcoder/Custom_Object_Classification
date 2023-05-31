import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import cv2 
import os
from model import CustomConvNet


# Import custom model from model.py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
model = CustomConvNet(num_classes=9).to(device, non_blocking=True)

# Load the model weights
model.load_state_dict(torch.load("./savedModels/net_20.pth", map_location=torch.device('cpu')))

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
# reshape to 1x3x224x224 to feed model as input
trans_img = trans_img.unsqueeze(0)

model.eval()

outputs = model(trans_img)
_, preds = torch.max(outputs,1)
print(class_label[preds[0]])

#Write predictions to image
# Define the text and position
text = class_label[preds[0]]
position = (10, 20)  # (x, y) coordinates of the starting point of the text

# Define the font properties
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1.0
color = (255, 0, 0)  # BGR color format (Blue, Green, Red)
thickness = 2

# Write the text on the image
cv2.putText(img, text, position, font, font_scale, color, thickness)
cv2.imwrite("output/predict_image_2.png",img)
