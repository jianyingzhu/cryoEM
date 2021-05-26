import torch
import torch.nn
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from time import time

__all__ = ['run']

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        assert len(images) == len(labels)
        super().__init__()
        self.images = images
        self.labels = labels
    def __len__(self):
        return len(self.images)
    def __getitem__(self, i):
        return self.images[i], self.labels[i]

class CNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, 25)
        self.conv2 = torch.nn.Conv2d(6, 16, 35)
        self.fc1 = torch.nn.Linear(16 * 17 * 17, 240)
        self.fc2 = torch.nn.Linear(240, 48)
        self.fc3 = torch.nn.Linear(48, 2)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))     # (B, 1, 160, 160) -> (B, 6, 136, 136) -> (B, 6, 68, 68)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))     # (B, 6, 68, 68) -> (B, 16, 34, 34) -> (B, 16, 17, 17)
        x = torch.flatten(x, 1)                             # (B, 16, 17, 17) -> (B, 16 * 17 * 17)
        x = F.relu(self.fc1(x))                             # (B, 16 * 17 * 17) -> (B, 240)
        x = F.relu(self.fc2(x))                             # (B, 240) -> (B, 48)
        x = self.fc3(x)                                     # (B, 48) -> (B, 2)
        return x

def run(images, labels, k_fold = 10, batch_size = 10, epoch_num = 1):
    print('Preparing data.')
    dataset = ImageDataset(images, labels)
    size = len(dataset)

    print('Splitting dataset into training set (90%) and validation set (10%).')
    training_size = int(size - size / k_fold)
    validation_size = size - training_size
    training_set, validation_set = random_split(dataset, [training_size, validation_size])
    training_loader = DataLoader(dataset = training_set, batch_size = batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_set, batch_size = batch_size)

    print('Defining network, loss and optimizer.')
    net = CNN().double()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = 1e-3, momentum = 0.9)

    print('Starting training.')
    time1 = time()
    for epoch in range(epoch_num):
        running_loss = 0.
        for i, data in enumerate(training_loader):
            images, labels = data
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 10 == 9:
                print(f'  epoch: {epoch + 1}, data_num: {i - 9} to {i + 1}, running_loss: {running_loss / 10:.3f}')
                running_loss = 0.
    time2 = time()
    print(f'Training completed in {time2 - time1:.4f}s.')

    print('Testing network by validation dataset.')
    correct = 0
    total = 0
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, dim = 1)
            correct += (predictions == labels).sum().item()
            total += batch_size
    print(f'Accuracy of the network on the validation dataset is {correct / total * 100:.2f}.')
