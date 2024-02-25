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

global colorlabels
numcolors = 0

colornames = ["red", "blue", "green", "purple", "yellow", "cyan", "orange", "brown", "pink", "white"]
colorlabels = np.random.randint(0, 10, 1000000)
colorrange = .1
colorvals = [
    [1 - colorrange, colorrange * 1, colorrange * 1],
    [colorrange * 1, 1 - colorrange, colorrange * 1],
    [colorrange * 2, colorrange * 2, 1 - colorrange],
    [1 - colorrange * 2, colorrange * 2, 1 - colorrange * 2],
    [1 - colorrange, 1 - colorrange, colorrange * 2],
    [colorrange, 1 - colorrange, 1 - colorrange],
    [1 - colorrange, .5, colorrange * 2],
    [.6, .4, .2],
    [1 - colorrange, 1 - colorrange * 3, 1 - colorrange * 3],
    [1-colorrange,1-colorrange,1-colorrange]
]

#comment this
def Colorize_func(img):
    global numcolors,colorlabels  

    thiscolor = colorlabels[numcolors]  # what base color is this?

    rgb = colorvals[thiscolor];  # grab the rgb for this base color
    numcolors += 1  # increment the index
    r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
    g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
    b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange
    np_img = np.array(img, dtype=np.uint8)
    np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
    backup = np_img
    np_img = np_img.astype(np.uint8)
    img = Image.fromarray(np_img, 'RGB')
    return img

def thecolorlabels(datatype):
    colornumstart = 0
    coloridx = range(colornumstart, len(datatype))
    labelscolor = colorlabels[coloridx]
    return torch.tensor(labelscolor)

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
def dataset_builder(data_set_flag, bs, element_colors = {}, retina = False, element_locations = {}):
    retina_size = 100
    train_dataset = None
    test_dataset = None
    train_skip = None

    class Colorize_specific:
        def __init__(self, col):
            self.col = col
        def __call__(self, img):
            # col: an int index for which base color is being used
            rgb = colorvals[self.col]  # grab the rgb for this base color

            r_color = rgb[0] + np.random.uniform() * colorrange * 2 - colorrange  # generate a color randomly in the neighborhood of the base color
            g_color = rgb[1] + np.random.uniform() * colorrange * 2 - colorrange
            b_color = rgb[2] + np.random.uniform() * colorrange * 2 - colorrange

            np_img = np.array(img, dtype=np.uint8)
            np_img = np.dstack([np_img * r_color, np_img * g_color, np_img * b_color])
            np_img = np_img.astype(np.uint8)
            img = Image.fromarray(np_img, 'RGB')

            return img
        
    class translate_to_right:
        def __init__(self, max_width):
            self.max_width = max_width
            self.pos = torch.zeros((100))
        def __call__(self, img):
            padding_left = random.randint(self.max_width // 2, self.max_width - img.size[0])
            padding_right = self.max_width - img.size[0] - padding_left
            padding = (padding_left, 0, padding_right, 0)
            pos = self.pos.clone()
            pos[padding_left] = 1
            return ImageOps.expand(img, padding), pos

    class translate_to_left:
        def __init__(self, max_width):
            self.max_width = max_width
            self.pos = torch.zeros((100))
        def __call__(self, img):
            padding_left = random.randint(0, (self.max_width // 2) - img.size[0])
            padding_right = self.max_width - img.size[0] - padding_left
            padding = (padding_left, 0, padding_right, 0)
            pos = self.pos.clone()
            pos[padding_left] = 1
            return ImageOps.expand(img, padding), pos
    
    class CustomMNIST(Dataset):
        def __init__(self, dataset, right_targets, color_targets=None):
            self.dataset = dataset
            self.right_targets = right_targets
            self.translate_left = PadAndPosition(translate_to_left(retina_size))
            self.translate_right = PadAndPosition(translate_to_right(retina_size))
            self.color_dict = None

            if color_targets is not None:
                colors = {}
                for color in color_targets:
                    for target in color_targets[color]:
                        colors[target] = color

                self.color_dict = colors

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            image, target = self.dataset[index]
            if target in self.right_targets:
                if self.color_dict is None or target not in self.color_dict:
                    transform = transforms.Compose([Colorize_func, self.translate_right])
                    return transform(image), target
                else:
                    transform = transforms.Compose([Colorize_specific(col = self.color_dict[target]), self.translate_right])
                    return transform(image), target

            else:
                if self.color_dict is None or target not in self.color_dict:
                    transform = transforms.Compose([Colorize_func, self.translate_left])
                    return transform(image), target
                else:
                    transform = transforms.Compose([Colorize_specific(col = self.color_dict[target]), self.translate_left])
                    return transform(image), target

    class PadAndPosition:
        def __init__(self, transform):
            self.transform = transform
        def __call__(self, img):
            new_img, position = self.transform(img)
            return transforms.ToTensor()(new_img), transforms.ToTensor()(img), position
            
    if data_set_flag == 'mnist':
        if retina == True:
            mnist_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform = None, download=True)
            train_dataset = CustomMNIST(mnist_dataset, element_locations['right'], element_colors)
            test_dataset = CustomMNIST(mnist_dataset, element_locations['right'], element_colors)
        else:
            train_dataset = datasets.MNIST(root='./mnist_data/', train=True, transform = transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)
            test_dataset = datasets.MNIST(root='./mnist_data/', train=False, transform = transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)
        # skip connection datasets
        train_skip = datasets.MNIST(root='./mnist_data/', train=True,
                                    transform=transforms.Compose([Colorize_func,transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8), transforms.ToTensor()]), download=True)
        
        test_skip = datasets.MNIST(root='./mnist_data/', train=False,
                                    transform=transforms.Compose([Colorize_func,transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8), transforms.ToTensor()]), download=True)
    
    elif data_set_flag == 'fashion_mnist':
        if retina == True:
            fmnist_dataset = datasets.MNIST(root='./fashionmnist_data/', train=True, transform = None, download=True)
            train_dataset = CustomMNIST(fmnist_dataset, element_locations['right'], element_colors)
            test_dataset = CustomMNIST(fmnist_dataset, element_locations['right'], element_colors)
        else:
            train_dataset = datasets.MNIST(root='./fashionmnist_data/', train=True, transform = transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)
            test_dataset = datasets.MNIST(root='./fashionmnist_data/', train=False, transform = transforms.Compose([Colorize_func, transforms.ToTensor()]), download=True)

        # skip connection dataset
        train_skip = datasets.FashionMNIST(root='./fashionmnist_data/', train=True,
                transform=transforms.Compose([Colorize_func, transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8),transforms.ToTensor()]), download=True)
        
        train_skip = datasets.FashionMNIST(root='./fashionmnist_data/', train=True,
                transform=transforms.Compose([Colorize_func, transforms.RandomRotation(90), transforms.RandomCrop(size=28, padding= 8),transforms.ToTensor()]), download=True)

    elif data_set_flag == 'cifar10':
        def dataset_with_indices(cls):
            """
            Modifies the given Dataset class to return a tuple data, target, index
            instead of just data, target.
            """

            def __getitem__(self, index):
                data, target = cls.__getitem__(self, index)
                return data, target, index

            return type(cls.__name__, (cls,), {
            '   __getitem__': __getitem__,})

        #makes a new data set that returns indices
        CIFAR10windicies = dataset_with_indices(datasets.CIFAR10)

        train_dataset = CIFAR10windicies(root='./cifar_data/', train=True ,transform=transforms.ToTensor(), download=True)
        test_dataset = CIFAR10windicies(root='./cifar_data/', train=False, transform=transforms.ToTensor(), download=False)

    train_loader_noSkip = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True)
    test_loader_noSkip = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=True, drop_last=True)

    if data_set_flag != 'cifar10':
        train_loader_skip = torch.utils.data.DataLoader(dataset=train_skip, batch_size=bs, shuffle=True,  drop_last= True)
        test_loader_skip = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False,  drop_last=True)

        return train_loader_noSkip, train_loader_skip, test_loader_noSkip, test_loader_skip
    else:
        return train_loader_noSkip, test_loader_noSkip