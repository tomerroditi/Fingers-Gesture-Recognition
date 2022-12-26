import sklearn
from Source.pipelines import Data_Pipeline
from Source.data_manager import Data_Manager
import Source.models as models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import torch.nn as nn
import numpy as np
import torch
from importlib import reload

# %% pipeline definition and data manager creation
pipeline = Data_Pipeline()
dm = Data_Manager([5], pipeline)
print(dm.data_info())

# %% extract datasets from the data manager and convert into torch dataloaders
dataset = dm.get_dataset(experiments=['005_1_2', '005_2_2'], include_synthetics=False)
print('finished extracting the dataset')
data = dataset[0]
labels = dataset[1]  # labels of not synthetic data includes the gesture number and experiment name as well
# %% train test split and convert to torch dataloaders
# we split the data according to the gesture, gesture number and experiment name, we want all segments of the same
# gesture and experiment to be in the same set. the labels format is '<experiment name>_<gesture name>_<gesture number>'
# the splitting function bellow takes care of that...
train_data, test_data, train_labels, test_labels = models.train_test_split_by_gesture(data, labels=labels,
                                                                                      test_size=0.2, seed=42)
# add synthetic data to the training set (optional, enrich the training data)
train_data, train_labels = dm.add_synthetics(train_data, train_labels, 6)

# reshape the data to fit the model
train_data = train_data.reshape(train_data.shape[0], 1, 4, 4)
test_data = test_data.reshape(test_data.shape[0], 1, 4, 4)

# remove the experiment name and the gesture number from the labels - prepare labels for the model training
train_labels_striped = np.char.strip(train_labels, '_0123456789')
test_labels_striped = np.char.strip(test_labels, '_0123456789')

# convert the labels to integers
label_encoder = preprocessing.LabelEncoder()
train_targets = label_encoder.fit_transform(train_labels_striped)
test_targets = label_encoder.transform(test_labels_striped)

# create the dataloaders
train_loader, test_loader = models.create_dataloaders(train_data, train_targets, test_data, test_targets, batch_size=64)

# %% model training
# set training Hyper-parameters:
L2_penalty = 0.0001
lr = 1e-3  # learning rate
num_epochs = 1000

# set the model, loss function and optimizer and train the model
model = models.Net(label_encoder=label_encoder)
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_penalty)
models.train(model, train_loader, test_loader, num_epochs, optimizer, loss_func)

# %% model evaluation
# test accuracy - we check how many gestures are correctly classified using majority voting on the segments of each gesture
predictions = model.classify(test_data)

test_labels_unique = np.unique(test_labels)
test_labels_unique_pred = np.empty(0, dtype=int)
for gesture in test_labels_unique:
    gesture_test_labels = test_labels[test_labels == gesture]
    gesture_predictions = predictions[test_labels == gesture]
    # predict the gesture by the most common prediction
    gesture_prediction = np.bincount(gesture_predictions).argmax()
    test_labels_unique_pred = np.append(test_labels_unique_pred, gesture_prediction)

# convert the predictions to the original labels
test_labels_unique_pred = label_encoder.inverse_transform(test_labels_unique_pred)

# strip the gesture number and experiment name from the true labels
test_labels_unique_striped = np.char.strip(test_labels_unique, '_0123456789')

# calculate the accuracy
accuracy = np.sum(test_labels_unique_pred == test_labels_unique_striped) / len(test_labels_unique_striped)
print('test accuracy: ', accuracy)

# create and display a confusion matrix
confusion_matrix = sklearn.metrics.confusion_matrix(test_labels_unique_striped, test_labels_unique_pred, labels=label_encoder.classes_)
sklearn.metrics.ConfusionMatrixDisplay(confusion_matrix, display_labels=label_encoder.classes_).plot()

