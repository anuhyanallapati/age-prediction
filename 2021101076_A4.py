!pip install timm

import numpy as np
import pandas as pd
from glob import glob
from os.path import join
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch.optim as optim

import torch.nn.functional as F
from torchvision.transforms import Grayscale
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18
from torchvision.transforms import ColorJitter
import random  # Add this line to import the random module
from torchvision.transforms import RandomAffine, RandomHorizontalFlip, RandomRotation

import timm
import torchvision.transforms as transforms
from torchvision.transforms import Compose, ColorJitter, RandomAffine, RandomErasing, Resize, CenterCrop, ToTensor, Normalize
from torch.optim.lr_scheduler import StepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, data_path, annot_path, train=True):
        super(AgeDataset, self).__init__()

        self.annot_path = annot_path
        self.data_path = data_path
        self.train = train

        self.ann = pd.read_csv(annot_path)
        self.files = self.ann['file_id']
        if train:
            self.ages = self.ann['age']
            self.ages = (self.ages - self.ages.min()) / (self.ages.max() - self.ages.min())
        self.transform = self._transform(224)

    @staticmethod    
    def _convert_image_to_rgb(image):
        return image.convert("RGB")


    def _transform(self, n_px):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # mean = [0.485]
        # std = [0.229]
        return Compose([
#             Resize(n_px),
#             torchvision.transforms.Resize(n_px),
            torchvision.transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            torchvision.transforms.RandomRotation(degrees=(-10, 10)),  # Randomly rotate the image by -10 to 10 degrees
#             torchvision.transforms.CenterCrop(n_px),  # Crop the image to n_px size at the center
            ColorJitter(contrast=0.2, brightness=0.2, saturation=0.2, hue=0.2),  # Adjust contrast with a random factor in the range [1-0.5, 1+0.5]
            # RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3)),  # Randomly erase patches of the image
            # random blurring
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),
#             RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-10, 10)),
            self._convert_image_to_rgb,
            # Grayscale(num_output_channels=1),
            ToTensor(),
            Normalize(mean, std),
        ])

    def read_img(self, file_name):
        im_path = join(self.data_path,file_name)   
        img = Image.open(im_path)
        img = self.transform(img)
        return img
    
    def __getitem__(self, index):
        file_name = self.files[index]
        img = self.read_img(file_name)
        if self.train:
            age = self.ages[index]
            return img, age
        else:
            return img
    
    def __len__(self):
        return len(self.files)

train_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train'
train_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/train.csv'

annotations = pd.read_csv(train_ann)
ages = annotations['age']
min_age = ages.min()
max_age = ages.max()

train_dataset = AgeDataset(train_path, train_ann, train=True)


test_path = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/test'
test_ann = '/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv'
test_dataset = AgeDataset(test_path, test_ann, train=False)

train_size = int(0.9 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

model = timm.create_model('efficientnet_b4', pretrained=True)

filters_count = model.classifier.in_features

model.classifier = nn.Sequential(
    nn.Linear(filters_count, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 1),
)

model.to(device)


criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

def train(model, train_loader, optimizer, criterion, num_epochs=10):
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, ages in tqdm(train_loader):
            images, ages = images.to(device), ages.to(device).float()

            outputs = model(images)
            loss = criterion(outputs.squeeze(), ages)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

        scheduler.step()

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, ages in val_loader:
                images, ages = images.to(device), ages.to(device).float()

                outputs = model(images)
                descaled_outputs = outputs * (max_age - min_age) + min_age
                desc_ages = ages * (max_age - min_age) + min_age

                loss = criterion(descaled_outputs.squeeze(), desc_ages)
                running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(val_loader.dataset)
        print(f"Val Loss: {epoch_loss:.4f}")

train(model, train_loader, optimizer, criterion)

###### SUBMISSION CSV FILE #####

@torch.no_grad
def predict(loader, model):
    model.eval()
    predictions = []

    for img in tqdm(loader):
        img = img.to(device)

        pred = model(img)
        predictions.extend(pred.flatten().detach().tolist())

    return predictions

preds = predict(test_loader, model)

def descale_age(scaled_age):
    return scaled_age * (max_age - min_age) + min_age

descaled_preds = [descale_age(pred) for pred in preds]


submit = pd.read_csv('/kaggle/input/smai-24-age-prediction/content/faces_dataset/submission.csv')
submit['age'] = descaled_preds
submit.head()

submit.to_csv('baseline-o4.csv',index=False)
