import os
import torch
from classifier import train, cnn_classifier, optimizer

checkpoint_folder_path = 'output' # the output folder for the trained model versions

if not os.path.exists(checkpoint_folder_path):
    os.mkdir(checkpoint_folder_path)

data_lst = []
for epoch in range(1, 201):
    loss, test_loss, test_acc, train_acc, cm_data = train(epoch)
    data_lst += [[loss, test_loss, test_acc, train_acc]]
    torch.save(data_lst, 'cnn_data.pt')
    torch.save(cm_data, 'cm_data.pt')
   
    if epoch% 5 == 0:
        checkpoint =  {
                 'state_dict': cnn_classifier.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                      }
        torch.save(checkpoint,f'{checkpoint_folder_path}/checkpoint_{str(epoch)}.pth')