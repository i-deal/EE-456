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
from sklearn.feature_extraction.text import CountVectorizer
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from joblib import dump, load
import copy

#import and load trained model components
from cVAE import load_cvae
from c_label_network import load_labels
from placement import load_pos
from embedding_t2c import load_t2c, corpus, cimg_names
from embedding_t2l import load_t2l

i = random.randint(0,1)
vae = load_cvae("output/checkpoint_cvae500.pth")
vae_labels = load_labels(f"output_label_net/checkpoint_labels{i}8000.pth")
pos = load_pos("output/checkpoint_pos150000.pth")
t2c = load_t2c("output_t2c/checkpoint_t2c499.pth")
t2l = load_t2l("output_t2l/checkpoint_t2l100.pth")

vae.eval()
vae_labels.eval()
pos.eval()
t2c.eval()
t2l.eval()

def close_place(L1,L2):
    x = L1[0]+32
    y = L1[1]+32

    if L2[0] in list(range(L1[0],x)):
        L2[0] += random.randint(30,64-L2[0])
    elif L2[1] in list(range(L1[1],y)):
        L2[1] += random.randint(30,64-L2[1])
    return L2



def generate(input):
    print("Is input in corpus:", input in corpus)

    ngram_vectorizer = CountVectorizer(ngram_range=(1,4),
                                        token_pattern=r'\b\w+\b', min_df=1)

    trigram_vectorizer = CountVectorizer(ngram_range=(3,3),
                                        token_pattern=r'\b\w+\b', min_df=1)

    analyze = trigram_vectorizer.build_analyzer()

    X = ngram_vectorizer.fit_transform(corpus)
    X = trigram_vectorizer.fit_transform(corpus)

    # text embeddings
    clist = []
    for word in input.split():
        y = t2c(word)
        i, y = torch.max(y, dim=1)
        if y.cpu().detach().numpy() <= 9:
            clist += [y]

    loc_inds = t2l(input)
    loc_inds = loc_inds.round()

    imglist = []
    prev = [0,0]
    for i in range(len(clist)):
        loc = torch.zeros((96,96))
        ind = clist[i]
        loc_ind = [int(loc_inds[0][0+(2*i)].item()), int(loc_inds[0][1+(2*i)].item())]
        if i >= 1:
            loc_ind = close_place(prev,loc_ind)

        prev = loc_ind
        c_onehot = F.one_hot(ind, num_classes=10).float().cuda()
        loc[loc_ind[0], loc_ind[1]] = 1

        latent = vae_labels(c_onehot)
        img = vae.decoder(latent)

        img = pos(img, loc)
        imglist += [img]

    out_img = imglist[0]
    for i in range(1,len(imglist)):
        out_img = torch.clamp(out_img + imglist[i], 0, 255)

    filename =input + "_generated.png"
    utils.save_image( out_img,filename,nrow=1, normalize=False, range=(-1, 1),)

inputs = ["an automobile overtakes a truck", "a ship sails past a horse", "an airplane flies past a horse", "car overtakes a car", "a deer runs beside a ship",]

for text in inputs:
    generate(text)