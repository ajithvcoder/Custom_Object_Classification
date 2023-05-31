import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import os
from dataloader import TrainDataset, ValDataset
from model import CustomConvNet
#Optimizer
import torch.optim as optim
from torch.optim import lr_scheduler

train_dir = "./data/Fruits/train/"
val_dir = "./data/Fruits/test/"

#Load train and val dataset
train_data = TrainDataset(train_dir)
val_data = ValDataset(val_dir)

# Import custom model from model.py
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu" )
model = CustomConvNet(num_classes=9).to(device, non_blocking=True)


dataLoaders = {
    "train": DataLoader(train_data, batch_size = 64, shuffle=True),
    "valid": DataLoader(val_data, batch_size = 64, shuffle=True),
}

num_epochs = 200

# Loss function
criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def save_model(model, epoch_num):
    save_filename = "net_%s.pth"%epoch_num
    save_path=os.path.join("./savedModels",save_filename)
    torch.save(model.cpu().state_dict(), save_path)


for epoch in range(num_epochs):
    print("Epoch No -", epoch)

    model.train()
    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in dataLoaders["train"]:
        # Feeding input and labels to device
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
        running_loss += loss.item()* inputs.size(0)
        #calculate accuracy
        running_corrects += torch.sum(preds == labels.data)
    #scheduler step
    exp_lr_scheduler.step()
    # Calculate average loss and acc for a epoch
    epoch_loss = running_loss/len(train_data)
    epoch_acc = running_corrects.double()/len(train_data)

    print('Loss:{} , Acc{}'.format(epoch_loss, epoch_acc))
    # Saving model every five epoch
    if (epoch%5 == 0):
        save_model(model,epoch)
