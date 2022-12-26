import torch.nn as nn
import torch as torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.plotting import figure
from sklearn.model_selection import train_test_split


class Net(nn.Module):
    L2_weight = 0.0001
    dropout_rate = 0.3
    input_shape = (1, 4, 4)

    def __init__(self, num_classes: int = 10, label_encoder=None):
        super().__init__()
        self.label_encoder = label_encoder

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

    def classify(self, data: np.array):
        """classify the data according to the model's label encoder"""
        predictions = self(torch.from_numpy(data).float())
        predictions = torch.argmax(predictions, dim=1)
        predictions = predictions.detach().numpy()
        return predictions


def create_dataloaders(train_data: np.array, train_targets: np.array, test_data: np.array, test_targets: np.array,
                       batch_size: int = 64) -> (DataLoader, DataLoader):
    """
    This function created data loaders from numpy arrays of data and targets (train and test)
    inputs:
        train_data: numpy array of data for training
        train_targets: numpy array of targets for training
        test_data: numpy array of data for testing
        test_targets: numpy array of targets for testing
        batch_size: int, the size of the batches to be used in the data loaders
    outputs:
        train_loader: DataLoader object, contains the training data
        test_loader: DataLoader object, contains the testing data
    """
    train_dataset_torch = TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets))
    test_dataset_torch = TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_targets))

    train_dataloader = DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True,
                                  drop_last=True)
    test_dataloader = DataLoader(test_dataset_torch, batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader


def train_test_split_by_gesture(*arrays, labels: np.array = None, test_size: float = 0.2, seed: int = 42):
    """
    split data into train test sets, each set will contain data from different gestures repetitions, e.g. if we have
    10 repetitions of "fist" gestures from an experiment ('001_1_1_fist_1', '001_1_1_fist_2', ..., '001_1_1_fist_10')
    inputs:
        arrays: numpy arrays, each array should have the same number of rows (samples) as the labels array
        labels: numpy array of labels, each label should be a string
        test_size: float, the percentage of the data to be used for testing
    outputs:
        train_test_split: list of numpy arrays, each array contains the data for one of the sets (train, test)
    """
    unique_labels = np.unique(labels)
    unique_labels_no_num = np.char.rstrip(unique_labels, '_0123456789')
    train_gestures, test_gestures = train_test_split(unique_labels, stratify=unique_labels_no_num, test_size=test_size,
                                                     random_state=seed)
    train_arrays = []
    test_arrays = []
    for array in arrays:
        train_arrays.append(array[np.isin(labels, train_gestures)])
        test_arrays.append(array[np.isin(labels, test_gestures)])

    train_labels = labels[np.isin(labels, train_gestures)]
    test_labels = labels[np.isin(labels, test_gestures)]

    if len(arrays) == 1:
        return train_arrays[0], test_arrays[0], train_labels, test_labels
    else:
        return train_arrays, test_arrays, train_labels, test_labels


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

