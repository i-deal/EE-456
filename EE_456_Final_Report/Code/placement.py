
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
from torch.utils.data import DataLoader, Subset

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

# load a saved vae checkpoint
def load_pos(filepath):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(filepath,device)
    pos.load_state_dict(checkpoint['state_dict'])
    return pos

# model training data set and dimensions
data_set_flag = 'cifar10_pos' # mnist, cifar10, padded_mnist, padded_cifar10
imgsize = 32
output_size = 96
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = imgsize * imgsize * 3
h_dim1 = 256
h_dim2 = 128
z_dim = 5
l_dim = 100
#bs = 10

# must call the dataset_builder function from a seperate .py file

# padded cifar10, 2d retina, sight word latent, 


#CNN VAE
#this model takes in a single cropped image and a location 1-hot vector  (to be replaced by an attentional filter that determines location from a retinal image)
#there are three latent spaces:location, shape and color and 6 loss functions
#loss functions are: shape, color, location, retinal, cropped (shape + color combined), skip

class POS_NN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim):
        super(POS_NN, self).__init__()
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(output_size**2, imgsize**2)
        self.fc6 = nn.Linear((imgsize) * 2, 288*2) #//2
        self.fc7 = nn.Linear(288*2, 288*4)
        self.fc8 = nn.Linear(288*4, 288)

    def forward(self, x, l):
        bs = x.size()[0]
        x, l = x.cuda(), l.cuda()
        l = l.view(bs,-1)
        l = self.relu(self.fc5(l))
        l = l.view(-1,1,32,32)
        l = l.expand(-1, 3, 32,32) 
        # combine image, position dist.
        h = torch.cat([x,l], dim = 3)

        h = self.relu(self.fc6(h))
        h = self.relu(self.fc7(h))
        h = self.fc8(h).view(-1,3,output_size,output_size)

        return torch.sigmoid(h)

# function to build a model instance
def vae_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1, h_dim2 = h_dim2, z_dim = z_dim, l_dim = l_dim):
    pos = POS_NN(x_dim, h_dim1, h_dim2, z_dim, l_dim)

    folder_path = f'sample_{vae_type}_{data_set_flag}'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return pos, z_dim

########Actually build it
pos, z_dim = vae_builder()

#######what optimizer to use:
optimizer = optim.Adam(pos.parameters())

pos.cuda()

######the loss functions
#Pixelwise loss for the entire retina (dimensions are cropped image height x retina_size)
def loss_function(recon_x, x):
    x = x.clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, output_size * output_size * 3), reduction='sum')
    # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE 

# test recreate img with different features
def progress_out(data, epoch = 0, count = 0, filename = None):
    sample_size = 10
    pos.eval()
    #make a filename if none is provided
    if filename == None:
        filename = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png'
    sample_x, sample_l = data[1], data[2]
    with torch.no_grad():
        shape_color_dim = output_size
        recon = pos(sample_x, sample_l)
        utils.save_image(
        torch.cat([data[0].view(sample_size, 3, output_size, shape_color_dim).cuda(), recon.view(sample_size, 3, output_size, shape_color_dim).cuda()], 0),filename,nrow=sample_size, normalize=False, range=(-1, 1),)

def train(epoch, train_loader):
    pos.train()
    train_loss = 0
    dataiter = iter(train_loader)
    count = 0
    loader=train_loader
    for i in loader:
        data = dataiter.next()
        data = data[0]

        count += 1

        optimizer.zero_grad()

        recon_batch = pos(data[1], data[2])
        loss = loss_function(recon_batch, data[0])
        loss.backward()
        

        train_loss += loss.item()
        optimizer.step()

        if epoch % 4000 == 0:
            progress_out(data, epoch, count)
    return loss.item()