import matplotlib.pyplot as plt
import torch
from sklearn.metrics import confusion_matrix

data = torch.load('cnn_data.pt')
data = data[:14]

loss, test_loss, test_acc, train_acc = [], [], [], []
for sub in data:
    loss += [sub[0]]
    test_loss += [sub[1]]
    train_acc += [sub[2]]
    test_acc += [sub[3]]

plt.plot(loss, label='Training Error')
plt.plot(test_loss, label='Validation Error')
plt.ylabel('Error')
plt.xlabel('Epochs of Training')
plt.legend()
plt.show()

plt.plot(test_acc, label='Training Data Accuracy')
plt.plot(train_acc, label='Validation Data Accuracy')
plt.ylabel('Classification Accuracy')
plt.xlabel('Epochs of Training')
plt.legend()
plt.show()


data = torch.load('cm_data.pt')
# Create the confusion matrix
cm = confusion_matrix(data[1], data[0])

# Plot the confusion matrix
plt.imshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
plt.show()