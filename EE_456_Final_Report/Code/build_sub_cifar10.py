# prerequisites
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from torchvision.utils import save_image
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import imageio
import os
from torch.utils.data import DataLoader, Subset, Dataset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    return image

# datasets (string): mnist, fashion_mnist, cifar10
# element_colors (dictionary): assign specific colors to certain elements by labels, random colors will be assigned if None, colors must be integers within 0-9
# retina (boolean): whether the elements from this dataset are placed into a retina
# element_locations (dictionary): assign which side of the retina a group of elements will be placed, random sides will be assigned if None
bs = 10
train_dataset = datasets.CIFAR10(root='./cifar_data/', train=True ,transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='./cifar_data/', train=False, transform=transforms.ToTensor(), download=False)

selected_indices = {i: [] for i in range(10)}

# Iterate through the dataset and select 'samples_per_class' number of samples for each class
count = 0
for idx, (img, label) in enumerate(train_dataset):
    if len(selected_indices[label]) < 1:
        selected_indices[label].append(idx)
    elif len(selected_indices[label]) == 1:
        selected_indices[label][0] = idx

    count += 1
    # Check if we have selected enough samples for each class
    if all(len(indices) == 1 for indices in selected_indices.values()) and count >= 100:
        break
# Concatenate the selected indices for all classes
final_indices = np.concatenate([indices for indices in selected_indices.values()])

print(len(final_indices))
# Create a subset of the dataset using the selected indices
subset_trainset = torch.utils.data.Subset(train_dataset, final_indices)
torch.save(subset_trainset, 'sub1020_cifar10.pth')
#train_dataset = torch.load('sub_cifar10.pth')

#train_loader_noSkip = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True)
