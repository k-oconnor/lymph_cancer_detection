from pickle import TRUE
import numpy as np
import h5py
import os as os
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px

import torch.nn as nn


IN_PATH = os.path.join("data", "train.h5")
IN_PATH2 = os.path.join("data", "train_labels.h5")
h5_label = h5py.File(IN_PATH2, 'r')
ys = h5_label['y']
dist = torch.utils.data.DataLoader(ys)


# Making a pie chart of the distribution of cancer positives and negatives from the 262144 data points..
def show_pie(dist):
    i=0
    dict = {"No_Tumor":0, "Yes_Tumor":0}
    iterator = iter(dist)
    while i < 262144:
        sample = iterator.next()
        print(sample)
        if sample.item() == 0:
            dict['No_Tumor'] += 1
            i += 1
        else:
            dict['Yes_Tumor'] += 1
            i += 1
    df = pd.DataFrame.from_dict(dict,orient ='index',columns=['Tumor'])
    fig = px.pie(df, values = 'Tumor',names =df.index, title= 'Tumor Distribution in Training Data')
    fig.show()


# Initialize the .H5 datasets, and join the labels with the images
class PaCAM(torch.utils.data.Dataset):
    def __init__(self, IN_PATH, IN_PATH2, batch_size=32, shuffle=TRUE):
        super(PaCAM, self).__init__()
        h5_train = h5py.File(IN_PATH, 'r')
        h5_label = h5py.File(IN_PATH2, 'r')
        self.X = h5_train['x']
        self.y = h5_label['y']
        self.batch_size = batch_size
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.4981579 , 0.53774446, 0.68968743],[0.18233, 0.20106581, 0.16525005])])
    
    def __getitem__(self, item):
        idx = item % self.__len__()
        _slice = slice(idx*self.batch_size, (idx + 1) * self.batch_size)
        images = self._transform(self.X[_slice])
        labels = torch.tensor(self.y[_slice].astype(np.float32)).view(-1, 1)
        return {'images': images, 'labels': labels}

    def __len__(self):
        return len(self.X) // self.batch_size

    def _transform(self, images):
        tensors = []
        for image in images:
            tensors.append(self.transform(image))
        return torch.stack(tensors)

# Load the dataset
train_dataset = PaCAM(IN_PATH=IN_PATH, IN_PATH2=IN_PATH2)
train_loader = torch.utils.data.DataLoader(train_dataset)


# Display image and label.
def show_image(train_loader):
    i=0
    while i < 5:
        iterator = iter(train_loader)
        sample = iterator.next()
        train_features =sample['images'][0][:]
        train_labels = sample['labels']
        img = train_features[i].squeeze().permute(1,2,0)
        tense = train_labels[0][i]
        if tense.item() == 0.0:
            label = "Tumor Not Detected"
        else:
            label = "Tumor Detected"
        plt.imshow(img)
        plt.title(label)
        plt.show()
        i += 1


# Calculating sample mean and STD for image normalization
def normalize_stats(train_loader):
    image_mean = []
    image_std = []
    i=0
    while i < 1000:
        iterator = iter(train_loader)
        sample = iterator.next()
        train_features =sample['images'][0][:]
        img = train_features[i].squeeze().permute(1,2,0)
        npimg = np.array(img)
        image_mean.append(np.mean(npimg, axis = (0,1)))
        image_std.append(np.std(npimg, axis =(0,1)))
        i += 1
    channel_mean = np.mean(image_mean, axis=0)
    channel_std = np.mean(image_std, axis=0)
    return(channel_mean,channel_std)
# (array([0.6981579 , 0.53774446, 0.68968743], dtype=float32), array([0.18233, 0.20106581, 0.16525005], dtype=float32))
# These results will be inputs for the transforms.normalize method


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
    

        self.avg = nn.AvgPool2d(8)
        self.fc = nn.Linear(512 * 1 * 1, 2)
        
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.avg(x)
        x = x.view(-1, 512 * 1 * 1)
        x = self.fc(x)
        return x

## We tell torch to run the model on the GPU if possible.
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = CNN().to(device)


## Next, we have to define our loss function
criterion = nn.CrossEntropyLoss()
learning_rate = .002
optimizer = torch.optim.Adam(model.parameters(),lr=.002)



iterator = iter(train_loader)
sample = iterator.next()
epochs = 10
steps = len(train_loader)
for epoch in range(epochs):
    train_loss = 0.0
    valid_loss = 0.0

    i =0
    while i < len(iterator):
        sample = iterator.next()
        images = sample['images'][0][:]
        labels = sample['labels'][0]
        images = images.to(device)
        labels = labels.to(device)
        output=model(images)
        loss = criterion(output, labels)
        train_loss += loss.item()*images.size(0)
        train_loss = train_loss/len(train_loader.sampler)

        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch+1, num_epochs, i+1, total_step, loss.item()))