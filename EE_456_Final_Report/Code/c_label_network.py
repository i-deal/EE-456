from cVAE import vae, VAElabels, progress_out, imgsize
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, utils
from torchvision.utils import save_image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

bs = 10

vae_labels= VAElabels(xlabel_dim=10, hlabel_dim=10, h2label_dim=18,  zlabel_dim=5)

if torch.cuda.is_available():
    vae.cuda()
    vae_labels.cuda()
    print('CUDA')

def image_recon(z_labels):
    with torch.no_grad():
        vae.eval()
        output=vae.decoder_noskip(z_labels)
    return output

def load_labels(filepath):
    checkpoint = torch.load(filepath)
    vae_labels.load_state_dict(checkpoint['state_dict_labels'])
    for parameter in vae_labels.parameters():
        parameter.requires_grad = False
    vae_labels.eval()
    return vae_labels


optimizer = optim.Adam(vae.parameters())

optimizer_labels= optim.Adam(vae_labels.parameters())

def loss_label(label_act,image_act):

    criterion=nn.MSELoss(reduction='sum')
    e=criterion(label_act,image_act)

    return e

def train_labels(epoch, train_loader):
    train_loss_label = 0
    vae_labels.train()

    dataiter = iter(train_loader)

    for i in tqdm(range(len(train_loader))):
        optimizer_labels.zero_grad()

        image, labels = dataiter.next()            
              
        image = image.cuda()
        labels = labels.cuda()
        input_oneHot = F.one_hot(labels, num_classes=10) 
        input_oneHot = input_oneHot.float()
        input_oneHot = input_oneHot.cuda()

        z_label = vae_labels(input_oneHot)

        z_rep = vae(image, return_z = 1)

        # train shape label net
        loss_of_labels = loss_label(z_label, z_rep)
        loss_of_labels.backward(retain_graph = True)
        train_loss_label += loss_of_labels.item()

        optimizer_labels.step()

        if epoch % 150 == 0:
            vae_labels.eval()
            vae.eval()

            with torch.no_grad():
                filename = f"label_sample/sample_{epoch}_{i}.png"
                recon_imgs = vae.decoder(z_rep)

                recon_labels = vae.decoder(z_label)

                sample_size = 10
                orig_imgs = image
                recon_labels = recon_labels
                utils.save_image(
                torch.cat([orig_imgs.view(sample_size, 3, imgsize, imgsize).cuda(), recon_labels.view(sample_size, 3, imgsize, imgsize).cuda()], 0),filename,nrow=sample_size, normalize=False, range=(-1, 1),)

    return loss_of_labels.item()


    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss_label / (len(train_loader.dataset) / bs)))