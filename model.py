# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
Created on Sun Dec 09 2018 11:01:00 

@author: aruroxx31 and palak
"""

from __future__ import print_function
from __future__ import division
import copy
import os
import time
import torch
import torchvision
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.image as mpimg
from torchvision import datasets, models
from skimage import io, transform
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)


"""
Image Size of Every Image
(768 * 1024 * 3)
"""

data_dir = "./dataset"

model_name = "alexnet"

num_classes = 2

batch_size = 4

num_epochs = 15

"""
We donot want to train the whole model
Just the reshaped parameters, so we set this parameter true
"""
feature_extract = True

"""
Pretrained alexnet on Imagenet
"""
alexnet = torchvision.models.alexnet(pretrained=True)



def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


"""
By default require_grad=True, but we donot change gradients
"""
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



"""
Parameters of the model according to the given image size
Image Input given (768, 1024, 3)

--> After 1st Conv Layer
(190, 255, 64)

--> After first MaxPooling Layer
(92, 125, 64)

--> After 2nd Conv Layer
(91, 124, 192)

--> After second MaxPooling Layer
(43, 69, 192)

--> After 3rd Conv Layer
(43, 69, 384)

--> After 4th Conv Layer
(43, 69, 256)

--> After 5th Conv Layer
(43, 69, 256)

--> After third MaxPooling Layer
(20, 33, 256)

Total 1D features in fully connected layers
20*33*256 = 168960
"""


"""
Resahping Fully Connected Layers According to our Image Size
Reinitializing the layers and then
changing the parameters
First Reinitialize the layers
and then changing the parameters
"""


# Data augmentation and normalization for training
# Just normalization for validation

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}

dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'val']}



class MyDataset(Dataset):
    def __init__(self, ...):
        # stuff
        ...
        # (2) One way to do it is define transforms individually
        # When you define the transforms it calls __init__() of the transform
        self.center_crop = transforms.CenterCrop(100)
        self.to_tensor = transforms.ToTensor()
        
        # (3) Or you can still compose them like 
        self.transformations = \
            transforms.Compose([transforms.CenterCrop(100),
                                transforms.ToTensor()])
        
    def __getitem__(self, index):
        # stuff
        ...
        data = # Some data read from a file or image
        
        # When you call the transform for the second time it calls __call__() and applies the transform 
        data = self.center_crop(data)  # (2)
        data = self.to_tensor(data)  # (2)
        
        # Or you can call the composed version
        data = self.trasnformations(data)  # (3)
        
        # Note that you only need one of the implementations, (2) or (3)
        return (img, label)

    def __len__(self):
        return count # of how many data(images?) you have
        
if __name__ == '__main__':
    # Call the dataset
    custom_dataset = MyCustomDataset(...)



