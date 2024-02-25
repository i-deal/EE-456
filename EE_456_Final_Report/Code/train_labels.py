from c_label_network import vae_labels, optimizer_labels, train_labels
import torch
from cVAE import vae, load_cvae
from dataset_builder import dataset_builder
import matplotlib.pyplot as plt

load_cvae('output/checkpoint_cvae500.pth')
bs = 10
train_dataset = torch.load('sub_cifar10.pt')

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bs, shuffle=True,  drop_last= True)

loss_lst = []
for epoch in range (1,1001):
   
    loss = train_labels(epoch, train_loader)
    loss_lst += [loss]
    if epoch in [6000]:
        checkpoint =  {
                 'state_dict_labels': vae_labels.state_dict(),

                 'optimizer_labels' : optimizer_labels.state_dict(),

                      }
        torch.save(checkpoint,f'output_label_net/checkpoint_zzlabels'+str(epoch)+'.pth')

plt.plot(loss_lst, label='Training Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.title('Label Network Training Error')
plt.legend()
plt.show()