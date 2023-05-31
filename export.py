import torch
from torchvision import transforms
from model import CustomConvNet

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
import cv2

img = cv2.imread("./test_data/Image_32.jpg")

img = cv2.resize(img, (224,224))
img = val_transform(img)
img = img.unsqueeze(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img.to(device)
model.to(device)

# Now we will save this model.
import torch.onnx
torch.onnx.export(model,
                  img,
                  "./savedModels/custommodel.onnx",
                  export_params=True,
                  opset_version=10,
                  verbose=True,              # Print verbose output
                  input_names=['input'],     # Names for input tensor
                  output_names=['output'])

