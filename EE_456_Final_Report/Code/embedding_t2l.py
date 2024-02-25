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

cimg_names = {'airplane':0, 'automobile':1, 'car':1, 'bird':2, 'cat':3, 'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}

limg_data = {'flies': [1,4], 'over': [2,3], 'passes': [2,1], 'sings': [3,4],
            'near': [2,4], 'runs': [2,3], 'beside': [3,4], 'chases': [3,4], 'jumps': [2,3],
            'overtakes': [2,3], 'climbs': [2,3], 'onto': [2,4], 'hops': [2,4],
            'sails': [4,1], 'past': [4,1], 'speeds': [4,3],
            'drives': [2,1], 'next': [4,3], 'to': [4,1]}

def build_loc(Q):
    if Q == 1:
        x = random.randint(5,20)
        y = random.randint(58,63)
    elif Q == 2:
        x = random.randint(50,60)
        y = random.randint(58,63)
    elif Q == 3:
        x = random.randint(5,18)
        y = random.randint(4,17)
    elif Q == 4:
        x = random.randint(50,60)
        y = random.randint(4,17)
    return [x,y]      

ngram_vectorizer = CountVectorizer(ngram_range=(1,4),
                                     token_pattern=r'\b\w+\b', min_df=1)

trigram_vectorizer = CountVectorizer(ngram_range=(6,6),
                                     token_pattern=r'\b\w+\b', min_df=1)
analyze = ngram_vectorizer.build_analyzer()
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
X = trigram_vectorizer.fit_transform(corpus)
'''print(ngram_vectorizer.vocabulary_)
h = ngram_vectorizer.transform(["car overtakes a car"]).toarray()
print(torch.tensor(h))
print(torch.tensor(h).size())'''

# load a saved vae checkpoint
def load_t2l(filepath):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(filepath,device)
    t2l.load_state_dict(checkpoint['state_dict'])
    return t2l

# model training data set and dimensions
data_set_flag = 'text2l' # mnist, cifar10, padded_mnist, padded_cifar10

output_size = 4
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = 261
h_dim1 = x_dim//2
bs = 10

class T2L_NN(nn.Module):
    def __init__(self, x_dim, h_dim1):
        super(T2L_NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(x_dim, h_dim1)
        self.fc2 = nn.Linear(h_dim1, h_dim1//2)
        self.fc3 = nn.Linear(h_dim1//2, output_size)

    def forward(self, x):
        h = ngram_vectorizer.transform([x]).toarray()
        h = torch.tensor(h).float().cuda()
        h = self.relu(self.fc1(h))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))

        return h

# function to build an  actual model instance
def t2l_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1):
    t2l = T2L_NN(x_dim, h_dim1)

    folder_path = f'sample_{vae_type}_{data_set_flag}'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return t2l

########Actually build it
t2l = t2l_builder()

#######what optimizer to use:
optimizer = optim.Adam(t2l.parameters())

# cross entropy loss
def loss_function(recon_x, x):
    x = x.clone()
    x = x.cuda()
    BCE = F.mse_loss(recon_x, x) # mse loss used for output range [0,64]
    return BCE 

t2l.cuda()

# data: ["text", tensor[x,y]]
def train(epoch):
    t2l.train()
    for words in trigram_vectorizer.vocabulary_:
        data = [words]
        locs = []
        for word in words.split():
            if word in limg_data:
                Qs = limg_data[word]
                for Q in Qs:
                    xy = build_loc(Q)
                    locs += [xy[0]]
                    locs += [xy[1]]

        data += [torch.tensor(locs[:4][::-1]).float().cuda()]

        optimizer.zero_grad()

        recon = t2l(data[0])
        loss = loss_function(recon, data[1])
        loss.backward()

        optimizer.step()

    if epoch % 1 == 0:
        print(f"{epoch}: t2l loss: {loss.item()}")
    return loss.item()