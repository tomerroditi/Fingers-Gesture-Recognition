import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from tqdm import tqdm
from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.plotting import figure


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
        x = F.relu(x)
        x = self.batch_norm_1(x)
        x = self.conv_2(x)
        x = F.relu(x)
        x = self.batch_norm_2(x)
        x = torch.flatten(x, 1)
        x = self.fc_1(x)
        x = F.relu(x)
        x = self.batch_norm_3(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.fc_2(x)
        x = F.relu(x)
        x = self.batch_norm_4(x)
        x = F.dropout(x, p=self.dropout_rate)
        x = self.fc_3(x)
        x = F.softmax(x, dim=1)
        return x


def train(model: torch.nn.Module, train_dataloader, test_dataloader, epochs: int, optimizer, loss_function):
    global y_loss, y_accu
    # initialize the variables for the loss and accuracy plotting
    y_loss = {'train': [], 'val': []}  # loss history
    y_accu = {'train': [], 'val': []}
    x_epoch = list(range(epochs))

    for epoch in tqdm(range(epochs), desc='training model', unit='epoch'):
        model.float()
        model.train()
        train_loop(train_dataloader, model, loss_function, optimizer)
        model.eval()
        test_loop(test_dataloader, model, loss_function)

    fig_1 = figure(title='Training Loss', x_axis_label='Epoch', y_axis_label='Loss')
    fig_1.line(x_epoch, y_loss['train'], legend_label='Train Loss', color='blue')
    fig_1.line(x_epoch, y_loss['val'], legend_label='Validation Loss', color='red')

    fig_2 = figure(title='Training Accuracy', x_axis_label='Epoch', y_axis_label='Accuracy')
    fig_2.line(x_epoch, y_accu['train'], legend_label='Train Accuracy', color='blue')
    fig_2.line(x_epoch, y_accu['val'], legend_label='Validation Accuracy', color='red')

    show(row(fig_1, fig_2))
    print("Done Training!")


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

    train_loss, train_accu = calc_loss_accu(dataloader, model, loss_fn)
    global y_loss, y_accu
    y_loss['train'].append(train_loss)  # average loss per batch
    y_accu['train'].append(train_accu * 100)  # accuracy in percent


def test_loop(dataloader, model, loss_fn):
    test_loss, test_accu = calc_loss_accu(dataloader, model, loss_fn)
    global y_loss, y_accu
    y_loss['val'].append(test_loss)  # average loss per batch
    y_accu['val'].append(test_accu * 100)  # accuracy in percent


def calc_loss_accu(dataloader, model, loss_fn):
    loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.float())
            loss += loss_fn(pred, y.long()).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    accu = correct / len(dataloader.dataset)
    loss = loss / len(dataloader)
    return loss, accu

