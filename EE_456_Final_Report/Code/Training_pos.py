# prerequisites
import torch
import os
from placement import train, pos, optimizer
from dataset_builder import dataset_builder
import random
from torchvision import datasets, transforms, utils
from PIL import Image, ImageOps, ImageEnhance, __version__ as PILLOW_VERSION
from torch.utils.data import Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

checkpoint_folder_path = 'output' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')
tensor_to_pil = transforms.ToPILImage()

class pad:
    def __init__(self, max_width):
        self.max_width = max_width
        self.pos = torch.zeros((96,96))
    def __call__(self, img):
        padding_left = random.randint(0, self.max_width - img.size[0])
        padding_right = self.max_width - img.size[0] - padding_left
        padding_top = random.randint(0, self.max_width - img.size[0])
        padding_bottom = self.max_width - img.size[0] - padding_top
        padding = (padding_left, padding_top, padding_right, padding_bottom)
        pos = self.pos.clone()
        pos[padding_left][padding_top] = 1
        return ImageOps.expand(img, padding), pos

class PadAndPosition:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, img):
        img = tensor_to_pil(img)
        new_img, position = self.transform(img)
        return transforms.ToTensor()(new_img), transforms.ToTensor()(img), position

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

bs=10
data_set_flag = 'padded_mnist' # set to desired data set
#train_loader, test_loader = dataset_builder('cifar10', bs) 
train_dataset = torch.load('sub_cifar10.pt')
train_dataset1 = torch.load('sub1020_cifar10.pth')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset+train_dataset1, batch_size=bs, shuffle=True,  drop_last= True)

train_dataset = CustomDataset(train_dataset+train_dataset1, transform =PadAndPosition(pad(96)))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True, drop_last= True)
max_epoch = 10000

loss_lst = []
for epoch in tqdm(range(1, max_epoch+1)):
    loss = train(epoch, train_loader)
    loss_lst += [loss]
    if epoch in [max_epoch*2,max_epoch*4]:
        checkpoint =  {
                 'state_dict': pos.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_pos{str(epoch)}.pth')

plt.plot(loss_lst, label='Training Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.title('Placement Training Error')
plt.legend()
plt.show()