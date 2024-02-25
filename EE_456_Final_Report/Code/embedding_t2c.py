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
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import imageio
import os
from torch.utils.data import DataLoader, Subset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

cimg_names = {'airplane':0, 'automobile':1, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9, 'neither':10}

ngram_vectorizer = CountVectorizer(ngram_range=(1,4),
                                     token_pattern=r'\b\w+\b', min_df=1)
onegram_vectorizer = CountVectorizer(ngram_range=(1,1),
                                     token_pattern=r'\b\w+\b', min_df=1)
corpus = [
    "an airplane flies over a cat",
    "a truck passes a horse",
    "a bird sings near an automobile",
    "a deer runs beside a ship",
    "a dog chases a bird",
    "a horse jumps over a frog",
    "an automobile overtakes a truck",
    "a cat climbs onto a bird",
    "a frog hops near a deer",
    "a ship sails past a horse",
    "an airplane passes a dog",
    "a truck overtakes a deer",
    "a bird flies over a frog",
    "a cat jumps onto a horse",
    "a deer runs beside an airplane",
    "a dog chases a frog",
    "a horse jumps over a truck",
    "an automobile overtakes a bird",
    "a frog hops near a cat",
    "a ship sails past a dog",
    "an airplane flies over a truck",
    "a truck passes a frog",
    "a bird sings near a deer",
    "a deer runs beside a cat",
    "a dog chases a horse",
    "a horse jumps over an automobile",
    "an automobile overtakes a ship",
    "a cat climbs onto a frog",
    "a frog hops near an airplane",
    "a ship sails past a bird",
    "an airplane flies over a horse",
    "a truck passes a cat",
    "a bird sings near a dog",
    "a deer runs beside a frog",
    "a dog chases a cat",
    "a horse jumps over a bird",
    "an automobile overtakes a deer",
    "a cat climbs onto a horse",
    "a frog hops near a truck",
    "a ship sails past a cat",
    "an airplane flies over a frog",
    "a truck passes a bird",
    "a bird sings near a horse",
    "a deer runs beside a dog",
    "a dog chases a horse",
    "a horse jumps over an automobile",
    "an automobile overtakes a bird",
    "a cat climbs onto a frog",
    "a frog hops near a deer",
    "a ship sails past a dog",
    "an airplane flies over a truck",
    "a truck passes a frog",
    "a bird sings near a cat",
    "a deer runs beside a horse",
    "a dog chases a bird",
    "a horse jumps over a ship",
    "an automobile overtakes a frog",
    "a cat climbs onto a horse",
    "a frog hops near an airplane",
    "a ship sails past a bird",
    "an airplane flies over a horse",
    "a truck passes a dog",
    "a bird sings near an automobile",
    "a deer runs beside a ship",
    "a dog chases a frog",
    "a horse jumps over a cat",
    "an automobile overtakes a truck",
    "a cat climbs onto a bird",
    "a frog hops near a deer",
    "a ship sails past a horse",
    "an airplane flies over a cat",
    "a truck passes a horse",
    "a bird sings near an automobile",
    "a deer runs beside a ship",
    "a dog chases a bird",
    "a horse jumps over a frog",
    "an automobile overtakes a truck",
    "a cat climbs onto a bird",
    "a frog hops near a deer",
    "a ship sails past a dog",
    "an airplane flies over a truck",
    "a truck passes a frog",
    "a bird sings near a deer",
    "a deer runs beside a cat",
    "a dog chases a horse",
    "a horse jumps over an automobile",
    "an automobile overtakes a ship",
    "a cat climbs onto a frog",
    "a frog hops near an airplane",
    "a ship sails past a bird",
    "an airplane flies over a horse",
    "a truck passes a cat",
    "a bird sings near a dog",
    "a deer runs beside a frog",
    "a dog chases a cat",
    "a horse jumps over a bird",
    "an automobile overtakes a deer",
    "a cat climbs onto a horse",
    "a frog hops near a truck",
    "a ship sails past a cat",
    "a car speeds by a truck",
    "a car drives next to a cat",
    "car overtakes a car"
]

X = ngram_vectorizer.fit_transform(corpus)
X = onegram_vectorizer.fit_transform(corpus)
# load a saved vae checkpoint
def load_t2c(filepath):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(filepath,device)
    t2c.load_state_dict(checkpoint['state_dict'])
    return t2c

# model training data set and dimensions
data_set_flag = 'text2c' # mnist, cifar10, padded_mnist, padded_cifar10

output_size = 11
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = 33 #len(ngram_vectorizer.vocabulary_)
h_dim1 = x_dim//2
bs = 10

class T2C_NN(nn.Module):
    def __init__(self, x_dim, h_dim1):
        super(T2C_NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(x_dim, h_dim1*2)
        self.fc2 = nn.Linear(h_dim1*2, h_dim1)
        self.fc3 = nn.Linear(h_dim1, output_size)

    def forward(self, x):
        h = onegram_vectorizer.transform([x]).toarray()
        h = torch.tensor(h).float().cuda()
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))

        return h

# function to build an  actual model instance
def t2c_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1):
    t2c = T2C_NN(x_dim, h_dim1)

    folder_path = f'sample_{vae_type}_{data_set_flag}'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return t2c

########Actually build it
t2c = t2c_builder()

#######what optimizer to use:
# learning rate = 0.01, weight decay of 0.001
optimizer = torch.optim.SGD(t2c.parameters(), lr=0.01)

# cross entropy loss
def loss_function(y, t):
    criterion = nn.CrossEntropyLoss()
    CE = criterion(y, t)
    return CE

t2c.cuda()

# data: ["text", 1-hot tensor]
def train(epoch):
    t2c.train()
    train_data = []
    train_targets = []
    for word in onegram_vectorizer.vocabulary_:
        data = [word]
        if word in cimg_names:
            t = cimg_names[word]
            data += [torch.tensor([t]).cuda()]
        else:
            data += [torch.tensor([10]).cuda()]

        optimizer.zero_grad()

        recon = t2c(data[0])
        loss = loss_function(recon, data[1])
        loss.backward()

        i, y = torch.max(recon, dim=1)
        train_data += [y.cpu().detach().numpy()]
        train_targets += [data[1].cpu().detach().numpy()]
        optimizer.step()

    if epoch % 5 == 0:
        acc = accuracy_score(train_data, train_targets)
        print(f"{epoch}: t2c accuracy: {acc}")
    return loss.item