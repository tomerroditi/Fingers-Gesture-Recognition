import torch.nn as nn
import torch as torch
import torch.nn.functional as F


class Net(nn.Module):
    L2_weight = 0.0001
    dropout_rate = 0.3
    input_shape = (1, 4, 4)

    def __init__(self, num_classes: int = 10):
        super().__init__()
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
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = self.batch_norm_3(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.fc_2(x)
        x = self.batch_norm_4(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.fc_3(x)
        x = F.softmax(x, dim=1)
        return x


def train(model: torch.nn.Module, train_dataloader, test_dataloader, epochs, optimizer, loss_function):
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        model.float()
        model.train()
        train_loop(train_dataloader, model, loss_function, optimizer)
        model.eval()
        test_loop(test_dataloader, model, loss_function)
        print("Done!")


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.float())
        loss = loss_fn(pred, y.long())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            test_loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")