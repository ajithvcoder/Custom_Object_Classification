from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch
import cv2
import os

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p=0.6),
    transforms.RandomRotation(degrees=30),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229,0.224,0.225]),
])
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229,0.224,0.225]),
])

class TrainDataset(Dataset):
    def __init__(self, train_dir):
        self.foldernames = [name for name in os.listdir(train_dir)]
        self.data = []
        for label in self.foldernames:
            for file in os.listdir(train_dir+label):
                #print(file,label)
                self.data.append([train_dir+label+"/"+file,label])
        self.classmap ={element: index for index, element in enumerate(self.foldernames)}
        self.img_dim = (224,224)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.classmap[class_name]
        img_tensor = train_transform(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id

class ValDataset(Dataset):
    def __init__(self, val_dir):
        self.foldernames = [name for name in os.listdir(val_dir)]
        self.data = []
        for label in self.foldernames:
            for file in os.listdir(val_dir+label):
                #print(file,label)
                self.data.append([val_dir+label+"/"+file,label])
        self.classmap ={element: index for index, element in enumerate(self.foldernames)}
        self.img_dim = (224,224)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_path, class_name = self.data[idx]
        img = cv2.imread(img_path)
        img = cv2.resize(img, self.img_dim)
        class_id = self.classmap[class_name]
        img_tensor = val_transform(img)
        class_id = torch.tensor(class_id)
        return img_tensor, class_id
