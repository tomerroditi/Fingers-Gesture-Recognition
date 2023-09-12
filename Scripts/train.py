import pandas as pd
import random
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.append('../')

import numpy as np
import Source.fgr.models as models

from Source.fgr.pipelines import Data_Pipeline
from Source.fgr.data_manager import Data_Manager


# specify model
class simpleCNN(torch.nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
            super().__init__()
            self.dropout_rate = dropout_rate

            self.conv_1 = nn.Conv2d(1, 2, 2, padding='same')
            self.batch_norm_1 = nn.BatchNorm2d(2)
            self.conv_2 = nn.Conv2d(2, 4, 2, padding='same')
            self.batch_norm_2 = nn.BatchNorm2d(4)
            self.fc_1 = nn.Linear(4*4*4, 40)  # 4*4 from image dimension, 4 from num of filters
            self.batch_norm_3 = nn.BatchNorm1d(40)
            self.fc_2 = nn.Linear(40, 20)
            self.batch_norm_4 = nn.BatchNorm1d(20)
            self.fc_3 = nn.Linear(20, num_classes)

    def forward(self, x):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:  # % chance to add noise to the batch (adjust to your needs)
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))  # up to 10% noise
            # move noise to the same device as x - super important!
            noise = noise.to(x.device)
            # add the noise to x
            x = x + noise
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = functional.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = functional.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.batch_norm_3(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_2(x)
        x = self.batch_norm_4(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_3(x)
        x = functional.softmax(x, dim=1)
        return x

class simpleMLP(torch.nn.Module):
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.fc_1 = nn.Linear(4*4, 40)
        self.batch_norm_1 = nn.BatchNorm1d(40)
        self.fc_2 = nn.Linear(40, 20)
        self.batch_norm_2 = nn.BatchNorm1d(20)
        self.fc_3 = nn.Linear(20, num_classes)
    
    def forward(self, x):
        # add white gaussian noise to the input only during training
        if self.training and random.random() < 0:
            noise = torch.randn(x.shape) * 0.1 * (float(torch.max(x)) - float(torch.min(x)))
            noise = noise.to(x.device)
            x = x + noise
        
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.batch_norm_1(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_2(x)
        x = self.batch_norm_2(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.fc_3(x)
        x = functional.softmax(x, dim=1)
        return x


# train function
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader): # data: (batch_size, 3, 512, 512)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # set gradient to zero
        output = model(data) # output: (batch_size, 10)
        loss = torch.nn.functional.cross_entropy(output, target) # loss: (batch_size)
        loss.backward() # back propagation
        optimizer.step() # update parameters

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format( # print loss
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad(): # no gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += torch.nn.functional.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # sum up correct predictions
            total += target.size(0)
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.00f}%)\n'.format(test_loss, correct, total, (correct/total)*100) )# print loss and accuracy

    return correct/total

# main function
def main():

    data_dir = '../data/doi_10/emg'



    # get data
    train_transform = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.ToTensor(),
                ])
            
    # dataset = datasets.ImageFolder(data_dir+'/001_1', transform=train_transform)
    
    # pipeline definition and data manager creation
    data_path = Path('../../data/doi_10')
    pipeline = Data_Pipeline(base_data_files_path=data_path)  # configure the data pipeline you would like to use (check pipelines module for more info)
    subject = 1
    dm = Data_Manager([subject], pipeline)
    print(dm.data_info())

    dataset = dm.get_dataset(experiments=[f'{subject:03d}_*_*'])
    data = dataset[0]
    labels = dataset[1]

    data = torch.Tensor(data)
    labelencoder = LabelEncoder()
    labels = labelencoder.fit_transform(np.char.strip(labels, '_0123456789'))
    labels = torch.Tensor(labels).to(torch.int64)
  
    dataset = TensorDataset(data, labels)

    # split data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # data loader
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    print(len(train_dataset))
    # for i in range(100):
    #     #generate images for the first 10 samples
    #     Path(f'../../data/doi_10/emg_test/001_1/{train_dataset[i][1]}').mkdir(parents=True, exist_ok=True)
    #     img = train_dataset[i][0].numpy().reshape(4,4)
    #     image = Image.fromarray(img, mode="L")
    #     image.save(f'../../data/doi_10/emg_test/001_1/{train_dataset[i][1]}/{i}.png')


    # specify device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # specify model
    model = simpleMLP().to(device)

    # specify optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # train
    for epoch in range(1, 100):
        train(model, device, train_loader, optimizer, epoch)
        test(model, test_loader, device=device)



if __name__ == '__main__':
    main()
