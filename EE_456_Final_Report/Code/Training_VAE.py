# prerequisites
import torch
import os
from cVAE import train, vae, optimizer
from dataset_builder import dataset_builder
import matplotlib.pyplot as plt

checkpoint_folder_path = 'output' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

bs=10
data_set_flag = 'padded_mnist' # set to desired data set
#train_loader, test_loader = dataset_builder('cifar10', bs) 
train_dataset = torch.load('sub_cifar10.pt')
#train_dataset1 = torch.load('sub1020_cifar10.pth')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True)

loss_lst = []
for epoch in range(1, 501):
    loss = train(epoch, train_loader)
    loss_lst += [loss]
    if epoch in [500]:
        checkpoint =  {
                 'state_dict': vae.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_fvae{str(epoch)}.pth')

plt.plot(loss_lst, label='Training Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.title('cVAE Training Error')
plt.legend()
plt.show()




