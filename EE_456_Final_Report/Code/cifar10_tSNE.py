
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import random
from cVAE import load_cvae

class_names = {0: 'airplane', 1: 'car', 2: 'bird', 3: 'cat',
    4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse',
    8: 'ship', 9: 'truck'}

vae = load_cvae("output/checkpoint_cvaex500.pth")
vae.eval()
dataset = torch.load('sub800_cifar10.pth')

z_reps = []
for data_tup in tqdm(dataset):
    #print(data_tup[0].size())
    z_rep = vae(data_tup[0].view(1,3,32,32).cuda(), return_z = 1)
    z_reps += [(z_rep,data_tup[1])]

print('cifar done')

data= z_reps
print(len(data))

tensors = [item[0][0].cpu().detach().numpy() for item in data]
tensors_array = np.array(tensors)

tsne = TSNE(n_components=2, perplexity=30)
embedded_data = tsne.fit_transform(tensors_array)

labels = [item[1] for item in data]
unique_classes = list(set(labels))
class_to_color = {cls: plt.cm.jet(i / len(unique_classes)) for i, cls in enumerate(unique_classes)}

plt.figure(figsize=(10, 8))
for i, cls in enumerate(unique_classes):
    class_indices = [j for j, label in enumerate(labels) if label == cls]
    plt.scatter(embedded_data[class_indices, 0], embedded_data[class_indices, 1], 
                label=f'{class_names[cls]}', color=class_to_color[cls], marker='o')

    representative_index = class_indices[random.randint(0,799)]
    plt.annotate(f'{class_names[cls]}', (embedded_data[representative_index, 0], embedded_data[representative_index, 1]), fontsize=20)
plt.legend()
plt.title('tSNE Plot of Latent Representations of CIFAR10 Images')
plt.show()





