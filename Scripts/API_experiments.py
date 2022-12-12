from Source.pipelines import Data_Pipeline
from Source.data_manager import Data_Manager
import Source.models as models
from sklearn import preprocessing
import torch.nn as nn
import torch


# %% pipeline definition and data manager creation
pipeline = Data_Pipeline()
dm = Data_Manager([1], pipeline)
print(dm.data_info())


# %% extract datasets from the data manager and convert into torch dataloaders
train_dataset = dm.get_dataset(experiments=['001_*_1', '001_*_2'], include_synthetics=False)
test_dataset = dm.get_dataset(experiments=['001_*_3'], include_synthetics=False)

label_encoder = preprocessing.LabelEncoder()
train_targets = label_encoder.fit_transform(train_dataset[1])
test_targets = label_encoder.transform(test_dataset[1])

# %%
train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_dataset[0]), torch.from_numpy(train_targets))
test_dataset = torch.utils.data.TensorDataset(torch.from_numpy(test_dataset[0]), torch.from_numpy(test_targets))

batch_size = 64
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# %% model training
# some training Hyper-parameters:
L2_penalty = 0.0001
lr = 1e-3  # learning rate
num_epochs = 1000

model = models.Net()
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=L2_penalty)

models.train(model, train_dataloader, test_dataloader, num_epochs, optimizer, loss_func)





