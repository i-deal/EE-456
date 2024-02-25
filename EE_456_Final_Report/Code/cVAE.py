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
def load_cvae(filepath):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    checkpoint = torch.load(filepath,device)
    vae.load_state_dict(checkpoint['state_dict'])
    return vae

# model training data set and dimensions
data_set_flag = 'padded_mnist_rg' # mnist, cifar10, padded_mnist, padded_cifar10
imgsize = 32
vae_type_flag = 'CNN' # must be CNN or FC
x_dim = imgsize * imgsize * 3
h_dim1 = 256
h_dim2 = 128
z_dim = 5
l_dim = 100

# must call the dataset_builder function from a seperate .py file
#to be moved
# padded cifar10, 2d retina, sight word latent, 


#CNN VAE
#this model takes in a single cropped image and a location 1-hot vector  (to be replaced by an attentional filter that determines location from a retinal image)
#there are three latent spaces:location, shape and color and 6 loss functions
#loss functions are: shape, color, location, retinal, cropped (shape + color combined), skip

class VAE_CNN(nn.Module):
    def __init__(self, x_dim, h_dim1, h_dim2, z_dim, l_dim):
        super(VAE_CNN, self).__init__()
        self.relu = nn.ReLU()
        # encoder part
        self.l_dim = l_dim
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(16)  # Latent vectors mu and sigma
        
        self.fc2 = nn.Linear(int(imgsize / 4) * int(imgsize / 4) * 16, h_dim2)
        self.fc_bn2 = nn.BatchNorm1d(h_dim2) # remove
        # bottle neck part
        self.fc31 = nn.Linear(h_dim2, z_dim)  # latent
        self.fc32 = nn.Linear(h_dim2, z_dim)

        # decoder part
        self.fc4s = nn.Linear(z_dim, h_dim2)  # latent

        self.fc5 = nn.Linear(h_dim2, int(imgsize/4) * int(imgsize/4) * 16)
        self.fc8 = nn.Linear(h_dim2, h_dim2)
                
        self.conv5 = nn.ConvTranspose2d(16, 64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn8 = nn.BatchNorm2d(3)

    def encoder(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.relu(self.bn3(self.conv3(h)))
        h = self.relu(self.bn4(self.conv4(h)))
        h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
        h = self.relu(self.fc_bn2(self.fc2(h)))

        return self.fc31(h), self.fc32(h) # mu, log_var

    def sampling(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z_rep):
        h = F.relu(self.fc4s(z_rep)) * 1
        h = F.relu(self.fc5(h)).view(-1, 16, int(imgsize / 4), int(imgsize / 4))
        h = self.relu(self.bn5(self.conv5(h)))
        h = self.relu(self.bn6(self.conv6(h)))
        h = self.relu(self.bn7(self.conv7(h)))
        h = self.conv8(h).view(-1, 3, imgsize, imgsize)
        return torch.sigmoid(h)

    def forward_layers(self, l1, l2, layernum, whichdecode):
        if layernum == 1:
            h = F.relu(self.bn2(self.conv2(l1)))
            h = self.relu(self.bn3(self.conv3(h)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 2:
            h = self.relu(self.bn3(self.conv3(l2)))
            h = self.relu(self.bn4(self.conv4(h)))
            h = h.view(-1, int(imgsize / 4) * int(imgsize / 4) * 16)
            h = self.relu(self.fc_bn2(self.fc2(h)))
            hskip = self.fc8(h)
            mu_shape = self.fc31(h)
            log_var_shape = self.fc32(h)
            mu_color = self.fc33(h)
            log_var_color = self.fc34(h)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        elif layernum == 3:
            hskip = self.fc8(l1)
            mu_shape = self.fc31(l1)
            log_var_shape = self.fc32(l1)
            mu_color = self.fc33(l1)
            log_var_color = self.fc34(l1)
            z_shape = self.sampling(mu_shape, log_var_shape)
            z_color = self.sampling(mu_color, log_var_color)

        if (whichdecode == 'cropped'):
            output = self.decoder_cropped(z_shape, z_color, 0, hskip)
        elif (whichdecode == 'skip_cropped'):
            output = self.decoder_skip_cropped(z_shape, z_color, 0, hskip)

        return output, mu_color, log_var_color, mu_shape, log_var_shape

    def forward(self, x, return_z = 0):
        mu_rep, log_var_rep = self.encoder(x)
        z_rep = self.sampling(mu_rep, log_var_rep)

        output = self.decoder(z_rep)

        if return_z == 1:
            return z_rep
        else:
            return output, mu_rep, log_var_rep

# function to build an  actual model instance
def vae_builder(vae_type = vae_type_flag, x_dim = x_dim, h_dim1 = h_dim1, h_dim2 = h_dim2, z_dim = z_dim, l_dim = l_dim):
    vae = VAE_CNN(x_dim, h_dim1, h_dim2, z_dim, l_dim)

    folder_path = f'sample_{vae_type}_{data_set_flag}'

    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    
    return vae, z_dim

########Actually build it
vae, z_dim = vae_builder()

#######what optimizer to use:
optimizer = optim.Adam(vae.parameters())

vae.cuda()

######the loss functions
#Pixelwise loss for the entire retina (dimensions are cropped image height x retina_size)
def loss_function(recon_x, x, mu, log_var):
    x = x.clone()
    x = x.cuda()
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, imgsize * imgsize * 3), reduction='sum')
    # KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE 

# test recreate img with different features
def progress_out(data, epoch = 0, count = 0, filename = None):
    sample_size = 10
    vae.eval()
    #make a filename if none is provided
    if filename == None:
        filename = f'sample_{vae_type_flag}_{data_set_flag}/{str(epoch + 1).zfill(5)}_{str(count).zfill(5)}.png'
    sample = data
    with torch.no_grad():
        shape_color_dim = imgsize
        recon, mu, log_var = vae(sample)
        utils.save_image(
        torch.cat([sample.view(sample_size, 3, imgsize, shape_color_dim).cuda(), recon.view(sample_size, 3, imgsize, shape_color_dim).cuda()], 0),filename,nrow=sample_size, normalize=False, range=(-1, 1),)

def train(epoch, train_loader):
    vae.train()
    train_loss = 0
    dataiter = iter(train_loader)
    count = 0
    loader=tqdm(train_loader)
    for i in loader:
        data = dataiter.next()
        data = data[0].cuda()

        count += 1

        optimizer.zero_grad()

        recon_batch, mu, log_var = vae(data)
        loss = loss_function(recon_batch, data, mu, log_var)
        loss.backward()
        

        train_loss += loss.item()
        optimizer.step()
        loader.set_description(
            (
                f'epoch: {epoch}; mse: {loss.item():.5f};'
            )
        )
        if epoch % 20 == 0:
            progress_out(data, epoch, count)
    return loss.item()

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))

# shape label network
class VAElabels(nn.Module):
    def __init__(self, xlabel_dim, hlabel_dim, h2label_dim, zlabel_dim):
        super(VAElabels, self).__init__()

        # encoder part
        self.fc1label = nn.Linear(xlabel_dim, hlabel_dim)
        #self.fc2label = nn.Linear(hlabel_dim, h2label_dim)
        self.fc21label= nn.Linear(hlabel_dim,  zlabel_dim) #mu
        self.fc22label = nn.Linear(hlabel_dim, zlabel_dim) #log-var


    def sampling_labels (self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_labels):
        h = F.relu(self.fc1label(x_labels))
        #h = F.relu(self.fc2label(h))
        mu_label = self.fc21label(h)
        log_var_label=self.fc22label(h)
        z_label = self.sampling_labels(mu_label, log_var_label)
        return  z_label