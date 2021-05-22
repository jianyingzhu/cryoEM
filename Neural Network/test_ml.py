import numpy as np
import torch
import mrcfile
import torchvision

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

# input
res_corr = np.load('/home/zhujianying/Compressed_Sensing/test/zhujianying/res_corr.npy', allow_pickle = True)
res_wrong = np.load('/home/zhujianying/Compressed_Sensing/test/zhujianying/res_wrong.npy', allow_pickle = True)
res_all = np.concatenate((res_corr, res_wrong), axis=0)

label_all = np.empty(res_all.shape[0]) # 1000 * 1
label_all[:res_corr.shape[0]] = 1 # True
label_all[res_corr.shape[0]: res_all.shape[0]] = 0 # False

res_all = torch.tensor(res_all)
res_all = res_all.unsqueeze(1) # 1000 * 160 * 160 ->  1000 * 1 * 160 * 160, add a single channel to the 'grey scale' image
label_all = torch.tensor(label_all)
label_all = label_all.long()

print(res_all.dtype)
# if torch.cuda.is_available():
#   tensor = tensor.to('cuda')

# dataset
class NoiselessDataset(torch.utils.data.Dataset):
    def __init__(self, res, labels):
        super().__init__() # delete also ok?
        self.images = res
        self.labels = labels
    def __len__(self):
        assert self.images.shape[0] == len(self.labels)
        return len(self.labels)
    def __getitem__(self, i):
        return (self.images[i, :, :, :], self.labels[i])

# k-fold
k = 10
dataset = NoiselessDataset(res_all, label_all)

train_size = int(len(dataset) * (1 - 1 / k))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

# Dataloader
batch_size = 10

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

# network
class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 25) # 10 * 1 * 160 * 160 -> 10 * 6 * 136 * 136, 10: batch size
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 35)
        self.fc1 = nn.Linear(16 * 17 * 17, 240)
        self.fc2 = nn.Linear(240, 48)
        self.fc3 = nn.Linear(48, 2)
    def forward(self, x):
        # x = self.conv1(x) # 10 * 1 * 160 * 160 -> 10 * 6 * 136 * 136, 10: batch size
        # x = F.relu(x) # 10 * 6 * 136 * 136 -> 10 * 6 * 136 * 136
        # x = self.pool(x) # 10 * 6 * 136 * 136 -> 10 * 6 * 68 * 68
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x))) # 10 * 6 * 68 * 68 -> 10 * 16 * 34 * 34 -> 10 * 16 * 34 * 34 -> 10 * 16 * 17 * 17
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = CNN().double()

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# training
epoch = 15
for epoch in range(epoch):
    train_loss = 0.0
    for i, data in enumerate(train_loader):
        (inputs, labels) = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        if i % 20 == 19:    # print every 20 mini-batches
            print(f'epoch: {epoch + 1: d}, data_num: {i + 1: d}, train_loss: {train_loss / i: .3f}')
            running_loss = 0.0

print('Finished Training')

PATH = '/home/zhujianying/Compressed_Sensing/output/cnn_net.pth'
torch.save(net.state_dict(), PATH)


# test
net = CNN().double()
net.load_state_dict(torch.load(PATH))

total = 0
correct = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images = images.double()
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: ', f'{100 * correct / total:.2f}')


