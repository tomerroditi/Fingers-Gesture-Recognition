from Source.pipelines import Data_Pipeline
from Source.data_manager import Data_Manager
import Source.models as models
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from importlib import reload
import matplotlib
from matplotlib import pyplot as plt
import torch.nn as nn
import torch

# %% pipeline definition and data manager creation
pipeline = Data_Pipeline()
dm = Data_Manager([7], pipeline)
print(dm.data_info())

# %% extract datasets from the data manager and convert into torch dataloaders
dataset = dm.get_dataset(experiments=['007_*_2'], include_synthetics=True)
data_dataset = dataset[0]
labels_dataset = dataset[1]
# %%
# split the dataset into train and test datasets using stratified splitting
train_data, test_data, train_labels, test_labels = train_test_split(data_dataset, labels_dataset, test_size=0.2, stratify=labels_dataset)

label_encoder = preprocessing.LabelEncoder()
train_targets = label_encoder.fit_transform(train_labels)
test_targets = label_encoder.transform(test_labels)

# %%
train_dataset_torch = torch.utils.data.TensorDataset(torch.from_numpy(train_data), torch.from_numpy(train_targets))
test_dataset_torch = torch.utils.data.TensorDataset(torch.from_numpy(test_data), torch.from_numpy(test_targets))

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset_torch, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset_torch, batch_size=batch_size, shuffle=False)

# %% model training
# some training Hyper-parameters:
L2_penalty = 0.0001
lr = 1e-3  # learning rate
num_epochs = 1000

model = models.Net()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_penalty)

models.train(model, train_dataloader, test_dataloader, num_epochs, optimizer, loss_func)

# %% model evaluation
# test accuracy
predictions = model(torch.from_numpy(test_data).float())
predictions = torch.argmax(predictions, dim=1)
predictions = predictions.detach().numpy()
accuracy = sum(predictions == test_targets) / len(test_targets)



