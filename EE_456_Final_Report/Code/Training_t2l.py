# prerequisites
import torch
import os
from embedding_t2l import train, t2l, optimizer
from dataset_builder import dataset_builder
from tqdm import tqdm
import matplotlib.pyplot as plt

checkpoint_folder_path = 'output_t2l' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

loss_lst = []
for epoch in (range(1, 101)):
    loss = train(epoch)
    loss_lst += [loss]
    if epoch in [100]:
        checkpoint =  {
                 'state_dict': t2l.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_t2l{str(epoch)}.pth')

plt.plot(loss_lst, label='Training Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.title('t2l Training Error')
plt.legend()
plt.show()