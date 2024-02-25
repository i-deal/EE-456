# prerequisites
import torch
import torch.nn as nn
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# detect whether a cuda device is available
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

imgsize = 32
bs = 100

# load the training and validation CIFAR-10 datasets
train_dataset = datasets.CIFAR10(root='./cifar_data/', train=True ,transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='./cifar_data/', train=False, transform=transforms.ToTensor(), download=False)

# pass the data sets into loaders for easier training
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=500, shuffle=True, drop_last=True)

# residual block class
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.identity = nn.Sequential() # skip connection
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(out_channels))
            
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        h += self.identity(x)
        h = self.relu(h)
        return h

# ResNet class
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc1 = nn.Linear(512, num_classes)

    # assemble a new residual block given the output dimensions of the blocks before and after
    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avgpool(h)
        h = self.fc1(h.view(-1,512))
        h = self.relu(h)
        return h

cnn_classifier = ResNet(ResBlock, [3, 4, 6, 3]).to(device)

# learning rate = 0.01, weight decay of 0.001
optimizer = torch.optim.SGD(cnn_classifier.parameters(), lr=0.01, weight_decay = 0.001, momentum = 0.9)

# cross entropy loss
def loss_function(y, t):
    criterion = nn.CrossEntropyLoss()
    CE = criterion(y, t)
    return CE

def train(epoch):
    cnn_classifier.train()
    dataiter = iter(train_loader)
    test_iter = iter(test_loader)
    loader=tqdm(train_loader, total=200)

    for i, j in enumerate(loader):
        optimizer.zero_grad()
        data,t = dataiter.next()
        data,t = data.to(device), t.to(device)
        y = cnn_classifier(data) # pass training data through network

        loss = loss_function(y,t) # compute loss
        loss.backward() # compute gradient
        optimizer.step() # apply backprop
        loader.set_description((f'epoch: {epoch}; BCE: {loss.item():.5f};'))

        if i + 1 == 200:
            break

    cnn_classifier.eval() # prevent a gradient from being computed
    test_data, test_t = test_iter.next()
    test_data,test_t = test_data.to(device), test_t.to(device)
    test_y = cnn_classifier(test_data)

    test_loss = loss_function(test_y,test_t)

    i, test_y = torch.max(test_y, dim=1)
    i, y = torch.max(y, dim=1) 

    # test accuracy of network
    test_acc = accuracy_score(test_y.cpu().detach().numpy(), test_t.cpu().detach().numpy())
    train_acc = accuracy_score(y.cpu().detach().numpy(), t.cpu().detach().numpy())
    print(test_acc,train_acc)
    print(test_y[:10]) # predictions of the 10 random test images saved below
    utils.save_image(
            test_data[:10],
            'test_sample.png',
            nrow=10, normalize=False, range=(-1, 1),)

    return loss.item(), test_loss.item(), test_acc, train_acc, [test_t.cpu().detach().numpy(),test_y.cpu().detach().numpy()]
