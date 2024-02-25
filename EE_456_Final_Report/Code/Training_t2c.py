# prerequisites
import torch
import os
from embedding_t2c import train, t2c, optimizer
from dataset_builder import dataset_builder
from tqdm import tqdm
import matplotlib.pyplot as plt

checkpoint_folder_path = 'output_t2c' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

# to resume training an existing model checkpoint, uncomment the following line with the checkpoints filename
# load_checkpoint('CHECKPOINTNAME.pth')

loss_lst = []
for epoch in (range(1, 501)):
    loss = train(epoch)
    loss_lst += [loss]
   
    if epoch in [499,1001,5000,10000]:
        checkpoint =  {
                 'state_dict': t2c.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_t2c{str(epoch)}.pth')

plt.plot(loss_lst, label='Training Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.title('t2c Training Error')
plt.legend()
plt.show()