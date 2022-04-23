import numpy as np
import h5py
import os as os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

IN_PATH = os.path.join("data", "train.h5")
IN_PATH2 = os.path.join("data", "train_labels.h5")


# Initialize the .H5 datasets, and join the labels with the images
class PaCAM(torch.utils.data.Dataset):
    def __init__(self, IN_PATH, IN_PATH2, batch_size=10):
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
show_image(train_loader)

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


