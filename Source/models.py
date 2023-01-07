import copy
import random

import sklearn.preprocessing
import torch.optim
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import torch.nn.functional as functional
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


class pre_training_utils:
    @staticmethod
    def train_test_split_by_gesture(*arrays: np.array, labels: np.array = None, test_size: float = 0.2, seed: int = 42)\
            -> (np.array, np.array, np.array, np.array) or (list[np.array], list[np.array], np.array, np.array):
        """
        split data into train test sets, each set will contain data from different gestures repetitions, e.g. if we have
        10 repetitions of "fist" gestures from an experiment ('001_1_1_fist_1', '001_1_1_fist_2', ..., '001_1_1_fist_10')
        then the fist 8 repetitions will be in the train set and the last 2 will be in the test set.
        inputs:
            arrays: numpy arrays, each array should have the same number of rows (samples) as the labels array
            labels: numpy array of labels, each label should be a string
            test_size: float, the percentage of the data to be used for testing
        outputs:
            train_arrays: list of numpy arrays, each array contains the train data (in the same order as the input
                          arrays), if only one array was given as input then a single numpy array will be returned
            test_arrays: list of numpy arrays, each array contains the test data (in the same order as the input
                         arrays), if only one array was given as input then a single numpy array will be returned
            train_labels: numpy array, contains the train labels
            test_labels: numpy array, contains the test labels
        """
        unique_labels = np.unique(labels)
        unique_labels_no_num = np.char.rstrip(unique_labels, '_0123456789')

        # in case there is a gesture with one repetition we will discard it since we cant split it to train and test
        only_one_rep = np.array([np.sum(unique_labels_no_num == label) == 1 for label in unique_labels_no_num])
        print(f'number of gestures with only one repetition: {np.sum(only_one_rep)}\n'
              f'Their names are: {unique_labels_no_num[only_one_rep]}')
        unique_labels = unique_labels[np.logical_not(only_one_rep)]
        unique_labels_no_num = unique_labels_no_num[np.logical_not(only_one_rep)]

        # split each gesture of each subject to train and test
        train_gestures, test_gestures = train_test_split(unique_labels, stratify=unique_labels_no_num, test_size=test_size,
                                                         random_state=seed)

        # now spit the data itself to train and test
        train_arrays = []
        test_arrays = []
        for array in arrays:
            train_arrays.append(array[np.isin(labels, train_gestures)])
            test_arrays.append(array[np.isin(labels, test_gestures)])

        # split the labels to train and test
        train_labels = labels[np.isin(labels, train_gestures)]
        test_labels = labels[np.isin(labels, test_gestures)]

        # in case we have more than one array we will return a list of arrays, otherwise we will return a single array
        if len(arrays) == 1:
            return train_arrays[0], test_arrays[0], train_labels, test_labels
        else:
            return train_arrays, test_arrays, train_labels, test_labels

    @staticmethod
    def folds_split_by_gesture(labels: np.array = None, num_folds: int = 5, seed: int = 42) -> np.array:
        """
        split data into k folds, each fold will contain data from different gestures repetitions, e.g. if we have
        10 repetitions of "fist" gestures from an experiment ('001_1_1_fist_1', '001_1_1_fist_2', ..., '001_1_1_fist_10')
        and we want 5 folds than each fold will contain 2 repetitions of "fist" gestures.
        inputs:
            labels: numpy array of labels, each label should be a string
            num_folds: int, the number of folds to split the data to
        outputs:
            fold_idx: a numpy array with the fold number for each sample
        """
        unique_labels = np.unique(labels)
        unique_labels_no_num = np.char.strip(unique_labels, '_0123456789')

        # in case there is a gesture with less than num_folds repetition we will discard it since we cant split it in a
        # stratified way.
        too_less_rep = np.array([np.sum(unique_labels_no_num == label) < num_folds for label in unique_labels_no_num])
        print(f'number of gestures with less than {num_folds} repetition: {np.sum(too_less_rep)}\n'
              f'Their names are: {unique_labels_no_num[too_less_rep]}')
        unique_labels = unique_labels[np.logical_not(too_less_rep)]
        unique_labels_no_num = unique_labels_no_num[np.logical_not(too_less_rep)]

        # split the unique labels (subjects gestures with repetition) to folds according to the gesture name (without
        # the repetition and subject number)
        skf = StratifiedKFold(num_folds, random_state=seed, shuffle=True)
        fold_idx_unique_labels = np.zeros(unique_labels.shape)
        for i, (_, test_idx) in enumerate(skf.split(unique_labels, unique_labels_no_num)):
            fold_idx_unique_labels[test_idx] = i
        fold_idx_unique_labels = np.array(fold_idx_unique_labels, dtype=int)

        # now split the data itself to folds
        fold_idx = np.zeros(labels.shape)
        for i, label in zip(fold_idx_unique_labels, unique_labels):
            fold_idx[labels == label] = i

        return fold_idx

    @staticmethod
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


class simple_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_encoder = LabelEncoder()
        self.accu_vals = {}
        self.loss_vals = {}
        self.train_data = np.empty(0)
        self.train_labels = np.empty(0)
        self.test_data = np.empty(0)
        self.test_labels = np.empty(0)

    def cv_fit_model(self, data: np.array, labels: np.array, num_folds: int = 5, batch_size: int = 64,
                     lr: float = 0.001, l2_weight: float = 0.0001, num_epochs: int = 200) -> \
            (list[nn.Module], list[float]):
        fold_idx = pre_training_utils.folds_split_by_gesture(labels, num_folds)
        # fit the label encoder
        self.label_encoder.fit(np.char.strip(labels, '_0123456789'))
        # cv training loop
        cv_scores = []
        models = []
        for fold in np.unique(fold_idx):
            curr_model = copy.deepcopy(self)  # create a copy of the model for each fold
            # split the data to train and test of the current fold
            train_idx = fold_idx != fold
            test_idx = fold_idx == fold
            curr_model.train_data = data[train_idx]
            curr_model.train_labels = labels[train_idx]
            curr_model.test_data = data[test_idx]
            curr_model.test_labels = labels[test_idx]
            # train the model, notice that we are using the labels without the repetition and subject number
            train_labels = self.label_encoder.transform(np.char.strip(curr_model.train_labels, '_0123456789'))
            test_labels = self.label_encoder.transform(np.char.strip(curr_model.test_labels, '_0123456789'))
            train_loader, test_loader = \
                pre_training_utils.create_dataloaders(curr_model.train_data, train_labels, curr_model.test_data,
                                                      test_labels, batch_size)
            optimizer = torch.optim.Adam(curr_model.parameters(), lr=lr, weight_decay=l2_weight)
            loss_func = nn.CrossEntropyLoss()
            curr_model.train_model(train_loader, test_loader, num_epochs, optimizer, loss_func)
            models.append(curr_model)
            cv_scores.append(curr_model.accu_vals['val'][-1])

        return models, cv_scores

    def fit_model(self, data: np.array, labels: np.array, test_size: float = 0.2, batch_size: int = 64, lr=0.001,
                  l2_weight: float = 0.0001, num_epochs: int = 200) -> None:
        """This function trains a model and returns the train and test loss and accuracy"""
        # split the data to train and test
        train_data, test_data, train_labels, test_labels = \
            pre_training_utils.train_test_split_by_gesture(data, labels=labels, test_size=test_size)

        # save the data and labels for later use
        self.train_data = train_data
        self.test_data = test_data
        self.train_labels = train_labels
        self.test_labels = test_labels

        # strip the labels from the numbers - the model will predict the gesture name and not the repetition number or
        # subject number (the numbers are used for data splitting)
        train_labels = np.char.strip(train_labels, '_0123456789')
        test_labels = np.char.strip(test_labels, '_0123456789')

        train_labels = self.label_encoder.fit_transform(train_labels)
        test_labels = self.label_encoder.transform(test_labels)

        # create data loaders
        train_dataloader, test_dataloader = pre_training_utils.create_dataloaders(train_data, train_labels, test_data,
                                                                                  test_labels, batch_size=batch_size)
        # train the model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=l2_weight)
        loss_func = nn.CrossEntropyLoss()
        self.train_model(train_dataloader, test_dataloader, num_epochs, optimizer, loss_func)

    def train_model(self, train_dataloader: DataLoader, test_dataloader: DataLoader, num_epochs: int,
                    optimizer, loss_function):
        # TODO: add type hints for optimizer and loss_function
        # initialize the variables for the loss and accuracy plotting
        self.loss_vals = {'train': [], 'val': []}  # loss history
        self.accu_vals = {'train': [], 'val': []}
        x_epoch = []  # epoch history

        # gui
        plt.ion()

        # create a plot for the loss and accuracy starting from epoch 0
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        ax1.set_title('Loss'), ax1.set_xlabel('Epoch'), ax1.set_ylabel('Loss')
        ax2.set_title('Accuracy'), ax2.set_xlabel('Epoch'), ax2.set_ylabel('Accuracy')
        loss_train_line, = ax1.plot(x_epoch, self.loss_vals['train'], label='train')
        loss_val_line, = ax1.plot(x_epoch, self.loss_vals['val'], label='val')
        accu_train_line, = ax2.plot(x_epoch, self.accu_vals['train'], label='train')
        accu_val_line, = ax2.plot(x_epoch, self.accu_vals['val'], label='val')
        plt.show()

        for epoch in tqdm(range(num_epochs), desc='training model', unit='epoch'):
            self.float()
            self.train()
            self._train_loop(train_dataloader, loss_function, optimizer)
            self.eval()
            self._test_loop(test_dataloader, loss_function)
            x_epoch.append(epoch)
            if epoch % 30 == 0: # update the plot with the new loss and accuracy
                loss_train_line.set_xdata(x_epoch)
                loss_train_line.set_ydata(self.loss_vals['train'])
                loss_val_line.set_xdata(x_epoch)
                loss_val_line.set_ydata(self.loss_vals['val'])
                accu_train_line.set_xdata(x_epoch)
                accu_train_line.set_ydata(self.accu_vals['train'])
                accu_val_line.set_xdata(x_epoch)
                accu_val_line.set_ydata(self.accu_vals['val'])
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()

        plt.ioff()
        plt.show()

        print("Done Training!")

    def _train_loop(self, dataloader, loss_fn, optimizer) -> None:
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            pred = self(X.float())
            loss = loss_fn(pred, y.long())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss, train_accu = self._calc_loss_accu(dataloader, loss_fn)
        self.loss_vals['train'].append(train_loss)  # average loss per batch
        self.accu_vals['train'].append(train_accu * 100)  # accuracy in percent

    def _test_loop(self, dataloader, loss_fn) -> None:
        test_loss, test_accu = self._calc_loss_accu(dataloader, loss_fn)
        self.loss_vals['val'].append(test_loss)  # average loss per batch
        self.accu_vals['val'].append(test_accu * 100)  # accuracy in percent

    def _calc_loss_accu(self, dataloader, loss_fn) -> (float, float):
        loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                pred = self(X.float())
                loss += loss_fn(pred, y.long()).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        accu = correct / len(dataloader.dataset)
        loss = loss / len(dataloader)
        return loss, accu

    def evaluate_model(self, data: np.array, labels: np.array, cm_title: str = 'unnamed') -> float:
        predictions = self.classify(data)  # get the gesture of each sample
        unique_labels = np.unique(labels)
        unique_labels_pred = []
        for label in unique_labels:
            curr_predictions = predictions[labels == label]  # get the predictions of the current label
            # predict the gesture by the most common string in the curr_prediction array

            unique, pos = np.unique(curr_predictions, return_inverse=True)  # Finds unique elements and their positions
            pred = unique[np.bincount(pos).argmax()]
            unique_labels_pred.append(pred)

        # strip the gesture number and experiment name from the true unique labels
        unique_labels_striped = np.char.strip(unique_labels, '_0123456789')

        # calculate the accuracy
        accuracy = np.sum(unique_labels_pred == unique_labels_striped) / len(unique_labels_striped)

        # create and display a confusion matrix
        cm_disp = sklearn.metrics.ConfusionMatrixDisplay.from_predictions(unique_labels_striped, unique_labels_pred)
        cm_disp.ax_.set_title(f'Confusion Matrix - {cm_title} - Accuracy: {accuracy:.3f}')
        return accuracy

    def classify(self, data: np.array):
        """classify the data according to the model's label encoder"""
        scores = self(torch.from_numpy(data).float())
        predictions = torch.argmax(scores, dim=1)
        predictions = predictions.detach().numpy()
        predictions = self.label_encoder.inverse_transform(predictions)
        return predictions


class Net(simple_CNN):
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
        # add white gaussian noise to the input only during training with probability 0.5
        if self.training:
            if random.random() < 0.3:
                x = x + torch.randn(x.shape) * 0.1 * (torch.max(x) - torch.min(x))  # up to 20% of the data range
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
        x = functional.dropout(x, p=self.dropout_rate)
        x = self.fc_2(x)
        x = self.batch_norm_4(x)
        x = functional.relu(x)
        x = functional.dropout(x, p=self.dropout_rate)
        x = self.fc_3(x)
        x = functional.softmax(x, dim=1)
        return x


